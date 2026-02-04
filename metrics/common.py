"""Common utilities for metrics computation."""
import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def normalize_label_path(path_str: str) -> str:
    """Normalize label paths by removing leading ./ and /."""
    p = path_str.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p


def normalize_pairwise_path(path_str: str) -> str:
    """Normalize pairwise result paths, stripping container mount prefixes."""
    p = path_str.replace("\\", "/")
    if p.startswith("/build/"):
        p = p[len("/build/"):]
    elif p.startswith("build/"):
        p = p[len("build/"):]
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p


def extract_distance(value) -> float:
    """Handle distance provided as a number, string, or as an object like { "distance": num }."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    if isinstance(value, dict):
        dist_val = value.get("distance")
        if isinstance(dist_val, (int, float)):
            return float(dist_val)
        if isinstance(dist_val, str):
            return float(dist_val)
    raise ValueError(f"Unsupported distance format: {value!r}")


def load_labels(labels_path: str) -> Dict[str, Set[str]]:
    """
    Load series labels from JSON file.
    Returns dict: series_name -> set of normalized image paths.
    """
    with open(labels_path, "r") as f:
        data = json.load(f)
    series_to_images: Dict[str, Set[str]] = {}
    for series, paths in data.items():
        if not isinstance(paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list of paths.")
        normalized: Set[str] = set()
        for idx, path in enumerate(paths):
            if not isinstance(path, str):
                raise ValueError(
                    f"Labels for series '{series}' must be strings; "
                    f"found {type(path).__name__} at index {idx}."
                )
            normalized.add(normalize_label_path(path))
        series_to_images[series] = normalized
    return series_to_images


def load_pairwise(pairwise_path: str) -> List[dict]:
    """
    Load pairwise results from JSON or CSV, skipping entries with missing/invalid distance.
    Returns list of dicts with 'image1', 'image2', and 'distance' fields.
    """
    pairwise_file = Path(pairwise_path)
    
    # Detect format by extension
    if pairwise_file.suffix.lower() == '.csv':
        return _load_pairwise_csv(pairwise_path)
    else:
        return _load_pairwise_json(pairwise_path)


def _load_pairwise_json(pairwise_path: str) -> List[dict]:
    """Load pairwise results from JSON format."""
    with open(pairwise_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Pairwise results must be a JSON list.")
    normalized_entries: List[dict] = []
    skipped_count = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Pairwise entry at index {idx} must be a JSON object.")
        if "image1" not in item or "image2" not in item:
            raise ValueError(
                f"Pairwise entry at index {idx} must have 'image1' and 'image2'."
            )
        a = item.get("image1")
        b = item.get("image2")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(
                f"Pairwise entry at index {idx} must have 'image1' and 'image2' as strings."
            )
        try:
            d = extract_distance(item.get("distance"))
        except (ValueError, KeyError, TypeError):
            skipped_count += 1
            continue
        normalized_entries.append({"image1": a, "image2": b, "distance": d})
    
    if skipped_count > 0:
        print(
            f"Warning: Skipped {skipped_count} pairwise entries due to missing/invalid distances. "
            "This may result in some images having fewer or no neighbors."
        )
    return normalized_entries


def _load_pairwise_csv(pairwise_path: str) -> List[dict]:
    """Load pairwise results from CSV format."""
    normalized_entries: List[dict] = []
    skipped_count = 0
    
    with open(pairwise_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        required_cols = {"image1", "image2", "distance"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must have columns: {required_cols}. Found: {reader.fieldnames}"
            )
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            a = row.get("image1", "").strip()
            b = row.get("image2", "").strip()
            distance_str = row.get("distance", "").strip()
            
            if not a or not b or not distance_str:
                skipped_count += 1
                continue
            
            # Handle dict string format like "{'distance': '0.123'}" or "{'distance': 0.123}"
            if distance_str.startswith("{") and ("'distance'" in distance_str or '"distance"' in distance_str):
                try:
                    import ast
                    distance_dict = ast.literal_eval(distance_str)
                    if isinstance(distance_dict, dict):
                        distance_str = str(distance_dict.get("distance", distance_str))
                except (ValueError, SyntaxError):
                    # Try to extract manually if ast fails
                    import re
                    match = re.search(r"'distance'[\s:]+['\"]?([\d.]+)", distance_str)
                    if match:
                        distance_str = match.group(1)
            
            try:
                d = extract_distance(distance_str)
            except (ValueError, KeyError, TypeError):
                skipped_count += 1
                continue
            
            normalized_entries.append({"image1": a, "image2": b, "distance": d})
    
    if skipped_count > 0:
        print(
            f"Warning: Skipped {skipped_count} pairwise entries due to missing/invalid distances. "
            "This may result in some images having fewer or no neighbors."
        )
    return normalized_entries


def load_anon_id_map(anon_map_path: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Load a path->id mapping JSON and invert it to id->path.
    Returns None if anon_map_path is None.
    """
    if not anon_map_path:
        return None
    with open(anon_map_path, "r") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("Anonymous ID map must be a JSON object of {path: id}.")
    id_to_path: Dict[str, str] = {}
    for path_str, anon_id in mapping.items():
        if not isinstance(path_str, str) or not isinstance(anon_id, str):
            raise ValueError("Anonymous ID map must use string keys and string values.")
        if anon_id in id_to_path:
            raise ValueError(
                f"Anonymous ID map must be 1:1; {anon_id} is used more than once."
            )
        id_to_path[anon_id] = path_str
    return id_to_path


def collect_pairwise_paths(
    entries: List[dict], id_to_path: Optional[Dict[str, str]] = None
) -> Set[str]:
    """
    Collect all normalized image paths present in pairwise entries, applying
    anon-ID mapping if provided.
    Preconditions: entries have been validated by load_pairwise().
    """
    found: Set[str] = set()
    for item in entries:
        img1 = item["image1"]
        img2 = item["image2"]
        src1 = id_to_path.get(img1, img1) if id_to_path else img1
        src2 = id_to_path.get(img2, img2) if id_to_path else img2
        found.add(normalize_pairwise_path(src1))
        found.add(normalize_pairwise_path(src2))
    return found


def validate_inputs(
    series_to_images: Dict[str, Set[str]],
    pairwise_entries: List[dict],
    id_to_path: Optional[Dict[str, str]] = None,
) -> None:
    """
    Ensure all labeled images are present in the pairwise results (as either image1 or image2),
    accounting for '/build/' prefix on container paths.
    """
    pairwise_paths = collect_pairwise_paths(pairwise_entries, id_to_path)

    missing: List[str] = []
    for series, images in series_to_images.items():
        for p in images:
            if p not in pairwise_paths:
                missing.append(p)
    if missing:
        missing_preview = "\n  ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "Validation failed: the following labeled images are not present in pairwise results "
            "(after normalizing container '/build/' prefixes):\n  "
            f"{missing_preview}{more}"
        )


def build_image_to_series_map(series_to_images: Dict[str, Set[str]]) -> Dict[str, str]:
    """
    Build reverse mapping from image path to series name.
    Returns dict: image_path -> series_name.
    """
    image_to_series: Dict[str, str] = {}
    for series, images in series_to_images.items():
        for img in images:
            image_to_series[img] = series
    return image_to_series


def validate_series_metadata_exists(labels_path: str = None) -> None:
    """Validate series metadata file exists and has valid data."""
    if labels_path is None:
        labels_path = "resources/labels/images_series_labels.json"
    
    if not Path(labels_path).exists():
        raise ValueError(f"Series metadata not found: {labels_path}")
    
    series_to_images = load_labels(labels_path)
    
    if not series_to_images:
        raise ValueError(f"Series metadata is empty: {labels_path}")
    
    invalid = [f"{s} ({len(imgs)} images)" for s, imgs in series_to_images.items() if len(imgs) < 2]
    if invalid:
        raise ValueError(f"Series must have ≥2 images: {', '.join(invalid)}")
    
    print(f"✓ Series metadata validated: {len(series_to_images)} series found")


def load_and_validate_data(
    labels_path: str,
    pairwise_path: str,
    anon_map_path: Optional[str] = None
) -> Tuple[Dict[str, Set[str]], List[dict], Optional[Dict[str, str]]]:
    """
    Load and validate labels, pairwise, and optional anon map.
    Returns: (series_to_images, pairwise_entries, id_to_path_map)
    """
    series_to_images = load_labels(labels_path)
    entries = load_pairwise(pairwise_path)
    id_to_path = load_anon_id_map(anon_map_path)
    validate_inputs(series_to_images, entries, id_to_path)
    return series_to_images, entries, id_to_path


def classify_pairwise_by_series(
    series_to_images: Dict[str, Set[str]],
    pairwise_entries: List[dict],
    id_to_path: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, bool]]:
    """
    Classify each pairwise distance as same-series (True) or different-series (False).
    
    Args:
        series_to_images: Mapping from series name to set of image paths
        pairwise_entries: List of dicts with 'image1', 'image2', 'distance' fields
        id_to_path: Optional mapping from anonymous IDs to actual paths
    
    Returns:
        List of (distance, is_same_series) tuples for all valid pairs
    """
    image_to_series = build_image_to_series_map(series_to_images)
    
    classified: List[Tuple[float, bool]] = []
    
    for entry in pairwise_entries:
        # Apply anon-ID mapping if provided
        img1_src = id_to_path.get(entry["image1"], entry["image1"]) if id_to_path else entry["image1"]
        img2_src = id_to_path.get(entry["image2"], entry["image2"]) if id_to_path else entry["image2"]
        
        # Normalize paths to handle container mount prefixes
        img1 = normalize_pairwise_path(img1_src)
        img2 = normalize_pairwise_path(img2_src)
        
        # Skip if either image is not in any series
        if img1 not in image_to_series or img2 not in image_to_series:
            continue
        
        same_series = image_to_series[img1] == image_to_series[img2]
        classified.append((float(entry["distance"]), same_series))
    
    return classified


def ensure_output_dir(output_path: str) -> Path:
    """Create parent directories for output path if they don't exist."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
