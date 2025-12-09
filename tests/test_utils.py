import os
import json
import binascii
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from evaluate import image_input_dir
from tests.path_id_store import PathIdStore

ANON_ENV_FLAG = "DEANONYMIZE_IMAGE_PATHS"


# ----------------------------
# Path normalization helpers (refactored from map.py)
# ----------------------------


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
    # Strip leading container mount prefix if present
    if p.startswith("/build/"):
        p = p[len("/build/") :]
    elif p.startswith("build/"):
        p = p[len("build/") :]
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p


# ----------------------------
# I/O helpers (refactored from map.py)
# ----------------------------


def load_labels(labels_path: str) -> Dict[str, Set[str]]:
    """
    Load series labels from JSON file.
    Returns dict: series_name -> set of normalized image paths.
    """
    with open(labels_path, "r") as f:
        data = json.load(f)
    # Expecting a dict: series_name -> list of image paths
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
    Load pairwise results, skipping entries with missing/invalid distance.
    Returns list of dicts with 'image1', 'image2', and 'distance' fields.
    """
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
        # Skip entries with missing/invalid distance rather than fail
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


# ----------------------------
# Validation (refactored from map.py)
# ----------------------------


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


# ----------------------------
# Distance handling and rankings (refactored from map.py)
# ----------------------------


def extract_distance(value) -> float:
    """
    Handle distance provided as a number, string, or as an object like { "distance": num }.
    """
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


def build_rankings(
    entries: List[dict], id_to_path: Optional[Dict[str, str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a neighbor ranking for each image:
    - For each pair (a, b, d), add (b, d) to a's list and (a, d) to b's list.
    - If duplicate pairs occur, keep the smallest distance.
    - Sort neighbor lists by ascending distance.
    Returns dict: image_path -> list of (neighbor_path, distance) tuples.
    """
    neighbors: Dict[str, Dict[str, float]] = {}

    for item in entries:
        a_raw = item["image1"]
        b_raw = item["image2"]
        d = item["distance"]
        # Apply anon-ID mapping before normalization, if provided
        a_src = id_to_path.get(a_raw, a_raw) if id_to_path else a_raw
        b_src = id_to_path.get(b_raw, b_raw) if id_to_path else b_raw
        a = normalize_pairwise_path(a_src)
        b = normalize_pairwise_path(b_src)

        if a == b:
            continue
        if a not in neighbors:
            neighbors[a] = {}
        if b not in neighbors:
            neighbors[b] = {}

        # Keep the minimum distance if duplicates appear
        if b not in neighbors[a] or d < neighbors[a][b]:
            neighbors[a][b] = d
        if a not in neighbors[b] or d < neighbors[b][a]:
            neighbors[b][a] = d

    # Convert to sorted lists
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for img, nbrs in neighbors.items():
        rankings[img] = sorted(nbrs.items(), key=lambda kv: kv[1])
    return rankings


# ----------------------------
# Series metadata validation
# ----------------------------


def validate_series_metadata_exists(labels_path: str = None) -> None:
    """
    Validate that series metadata file exists and contains valid series data.
    Raises ValueError if no series metadata is provided or if it's invalid.
    """
    if labels_path is None:
        labels_path = "resources/labels/images_series_labels.json"
    
    labels_file = Path(labels_path)
    
    if not labels_file.exists():
        raise ValueError(
            f"Series metadata file not found at {labels_path}. "
            "Series labels are required to run evaluation metrics. "
            "Please provide a JSON file with series -> list of image paths mapping."
        )
    
    # Try to load and validate the structure
    try:
        series_to_images = load_labels(labels_path)
    except Exception as e:
        raise ValueError(
            f"Failed to load series metadata from {labels_path}: {e}"
        )
    
    if not series_to_images:
        raise ValueError(
            f"Series metadata file {labels_path} is empty. "
            "At least one series with images is required."
        )
    
    # Check that each series has at least 2 images (needed for MAP computation)
    invalid_series = []
    for series, images in series_to_images.items():
        if len(images) < 2:
            invalid_series.append(f"{series} (has {len(images)} image(s), needs at least 2)")
    
    if invalid_series:
        raise ValueError(
            f"Invalid series found in {labels_path}. Each series must have at least 2 images:\n  "
            + "\n  ".join(invalid_series)
        )
    
    print(f"✓ Series metadata validated: {len(series_to_images)} series found")


# ----------------------------
# Original test_utils functions
# ----------------------------


def get_image_files(image_dir=None):
    """Return sorted list of image file paths from the given directory (default: resources/images or $IMAGE_INPUT_DIR)."""
    if image_dir is None:
        image_dir = os.environ.get("IMAGE_INPUT_DIR", str(image_input_dir))
    return sorted([str(f) for f in Path(image_dir).iterdir() if f.is_file()])

def hash_image(evaluator, image_path):
    """Return the hash dict for a given image file using the Evaluator."""
    return evaluator.hash_local_content(image_path)

def match_image(evaluator, image_path):
    """Return the match result dict for a given image file using the Evaluator."""
    return evaluator.match_local_content(image_path)

def get_results_dir():
    """Get the results directory path based on timestamp, creating it if needed."""
    # Use TEST_RUN_TIMESTAMP if set (from parent process), otherwise generate new one
    timestamp = os.getenv("TEST_RUN_TIMESTAMP")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    results_dir = output_root / "evaluation_results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def _should_anonymize() -> bool:
    # Default ON; set DEANONYMIZE_IMAGE_PATHS=1 to disable anonymization
    return os.getenv(ANON_ENV_FLAG, "").strip() != "1"

def _collect_image_paths(results: list[dict]) -> set[str]:
    paths: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("image", "image1", "image2"):
            val = item.get(key)
            if isinstance(val, str):
                paths.add(val)
    return paths

def _anonymize_results(results: list[dict], id_map: dict[str, str]) -> list[dict]:
    anonymized: list[dict] = []
    for item in results:
        new_item = item.copy()
        for key in ("image", "image1", "image2"):
            val = new_item.get(key)
            if isinstance(val, str):
                if val not in id_map:
                    raise ValueError(f"Missing anonymized ID for key '{key}' with value '{val}'")
                new_item[key] = id_map[val]
        anonymized.append(new_item)
    return anonymized

def _validate_results_for_anonymization(results) -> None:
    if not isinstance(results, list):
        raise TypeError("Anonymization expects 'results' to be a list of dicts.")
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError(f"Anonymization expects dict items; found {type(item).__name__} at index {idx}.")

def write_results(results, filename):
    """Write results (list of dicts) to a JSON file in the timestamped results directory."""
    results_dir = get_results_dir()
    output_path = results_dir / filename
    results_to_write = results

    if _should_anonymize():
        _validate_results_for_anonymization(results)
        paths = _collect_image_paths(results)
        # Build IDs using persistent store if ANON_ID_MAP_FILEPATH is set, else in-memory
        store = PathIdStore.from_env()
        id_map = store.get_ids_for_paths(paths)
        results_to_write = _anonymize_results(results, id_map)
        if store.has_persistence:
            store.save()

    with open(output_path, 'w') as f:
        json.dump(results_to_write, f, indent=2)
    return str(output_path)

def decode_clip_hex(hex_string: str) -> list[float] | None:
    """
    Convert HMA-style hex string (representing float32 vector) to a float list.
    """
    try:
        byte_data = binascii.unhexlify(hex_string)
        float_array = np.frombuffer(byte_data, dtype=np.float32)
        return float_array.tolist()
    except Exception as e:
        print(f"[ERROR] Failed to decode CLIP hex string: {e}")
        return None

def cosine_distance_safe(vec1, vec2):
    """Compute cosine distance with safety fallback."""
    from scipy.spatial.distance import cosine
    try:
        return cosine(vec1, vec2)
    except Exception as e:
        print(f"[ERROR] Failed distance computation: {e}")
        return None


def decode_clip_hex_to_floats(hex_str: str) -> list[float]:
    print(f"[DEBUG] CLIP hex length: {len(hex_str)}")
    byte_data = bytes.fromhex(hex_str)
    decoded = np.frombuffer(byte_data, dtype=np.float32).tolist()
    print(f"[DEBUG] Decoded CLIP vector length: {len(decoded)}")
    return decoded