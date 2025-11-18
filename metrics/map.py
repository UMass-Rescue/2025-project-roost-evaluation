import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


# ----------------------------
# Path normalization helpers
# ----------------------------

def normalize_label_path(path_str: str) -> str:
    p = path_str.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p


def normalize_pairwise_path(path_str: str) -> str:
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
# I/O helpers
# ----------------------------

def load_labels(labels_path: str) -> Dict[str, Set[str]]:
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
    with open(pairwise_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Pairwise results must be a JSON list.")
    normalized_entries: List[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Pairwise entry at index {idx} must be a JSON object.")
        if "image1" not in item or "image2" not in item:
            raise ValueError(f"Pairwise entry at index {idx} must have 'image1' and 'image2'.")
        a = item.get("image1")
        b = item.get("image2")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(
                f"Pairwise entry at index {idx} must have 'image1' and 'image2' as strings."
            )
        if "distance" not in item:
            raise ValueError(f"Pairwise entry at index {idx} missing 'distance'.")
        d = extract_distance(item.get("distance"))
        normalized_entries.append({"image1": a, "image2": b, "distance": d})
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
            raise ValueError(f"Anonymous ID map must be 1:1; {anon_id} is used more than once.")
        id_to_path[anon_id] = path_str
    return id_to_path


# ----------------------------
# Validation
# ----------------------------

def collect_pairwise_paths(entries: List[dict], id_to_path: Optional[Dict[str, str]] = None) -> Set[str]:
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


def validate_inputs(series_to_images: Dict[str, Set[str]], pairwise_entries: List[dict], id_to_path: Optional[Dict[str, str]] = None) -> None:
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
# Distance handling and rankings
# ----------------------------

def extract_distance(value) -> float:
    """
    Handle distance provided as a number or as an object like { \"distance\": num }.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if isinstance(value["distance"], (int, float)):
            return float(value["distance"])
    raise ValueError(f"Unsupported distance format: {value!r}")


def build_rankings(entries: List[dict], id_to_path: Optional[Dict[str, str]] = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a neighbor ranking for each image:
    - For each pair (a, b, d), add (b, d) to a's list and (a, d) to b's list.
    - If duplicate pairs occur, keep the smallest distance.
    - Sort neighbor lists by ascending distance.
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
# AP / mAP computation
# ----------------------------

def average_precision_at_k(preds: List[str], positives: Set[str], k: int) -> float:
    """
    Compute AP@k for a single query.
    - Denominator = min(k, len(positives))
    - Sum precision at each rank where a positive is found, divided by denominator.
    """
    if not positives:
        return 0.0
    denom = min(k, len(positives))
    if denom == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    # Iterate over predictions until we see denom positives or exhaust preds
    for idx, p in enumerate(preds, start=1):
        if p in positives:
            hits += 1
            sum_precisions += hits / idx
            if hits == denom:
                break
        if idx == k:
            # We only care about top-k ranks; continue if k < denom (won't happen since denom<=k)
            break
    return sum_precisions / denom


def mean_average_precision_at_k(
    preds_by_image: Dict[str, List[str]],
    images: Set[str],
    k: int,
) -> float:
    """
    Compute mAP@k over a set of query images, given per-image ranked predictions.
    """
    if not images:
        raise ValueError("Cannot compute mAP@k for an empty set of images.")

    ap_values: List[float] = []
    for query in images:
        positives = set(images)
        positives.discard(query)
        preds = preds_by_image[query]
        ap = average_precision_at_k(preds, positives, k)
        ap_values.append(ap)
    return sum(ap_values) / len(ap_values)


def compute_series_map(
    series_to_images: Dict[str, Set[str]],
    rankings: Dict[str, List[Tuple[str, float]]],
    max_k: int,
) -> Dict[str, List[float]]:
    """
    For each series, compute mAP@k for k=1..max_k.
    For series with size s, AP uses denom=min(k, s-1) per query.
    """
    series_to_map: Dict[str, List[float]] = {}
    for series, images in series_to_images.items():
        s = len(images)
        if s == 0:
            raise ValueError(f"Series '{series}' has no images.")
        if s <= 1:
            raise ValueError(f"Series '{series}' has only one image.")
        # Pre-compute predictions per image (neighbor order only)
        preds_by_image: Dict[str, List[str]] = {}
        for img in images:
            ranked = rankings[img]
            preds_by_image[img] = [nbr for (nbr, _) in ranked if nbr != img]

        ap_by_k: List[float] = []
        for k in range(1, max_k + 1):
            series_map_k = mean_average_precision_at_k(preds_by_image, images, k)
            ap_by_k.append(series_map_k)
        series_to_map[series] = ap_by_k
    return series_to_map


# ----------------------------
# Output
# ----------------------------

def write_series_map_csv(series_to_map: Dict[str, List[float]], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Determine header size (max k)
    max_k = max((len(v) for v in series_to_map.values()), default=0)
    fieldnames = ["series"] + [f"map@{k}" for k in range(1, max_k + 1)]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write per-series rows and accumulate for averages
        sums = [0.0 for _ in range(max_k)]
        counts = [0 for _ in range(max_k)]
        for series, values in series_to_map.items():
            row = {"series": series}
            for i, v in enumerate(values, start=1):
                row[f"map@{i}"] = f"{v:.6f}"
                if i - 1 < max_k:
                    sums[i - 1] += v
                    counts[i - 1] += 1
            writer.writerow(row)
        # Append average row
        if max_k > 0 and any(c > 0 for c in counts):
            avg_row = {"series": "mean"}
            for i in range(max_k):
                avg = (sums[i] / counts[i]) if counts[i] > 0 else 0.0
                avg_row[f"map@{i+1}"] = f"{avg:.6f}"
            writer.writerow(avg_row)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute mAP@k from pairwise image distances and series labels.")
    parser.add_argument("--labels", required=True, help="Path to labels JSON (series -> list of image paths).")
    parser.add_argument("--pairwise", required=True, help="Path to pairwise results JSON from pairwise_test.py.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--anon_map", required=False, help="Optional path to anon ID map JSON ({path: id}); used to map IDs back to paths.")
    args = parser.parse_args()

    series_to_images = load_labels(args.labels)
    entries = load_pairwise(args.pairwise)
    id_to_path = load_anon_id_map(args.anon_map)

    # Validate inputs (single call as requested)
    validate_inputs(series_to_images, entries, id_to_path)

    # Build rankings and compute mAP
    rankings = build_rankings(entries, id_to_path)
    max_k = max(len(v) for v in series_to_images.values())
    series_to_map = compute_series_map(series_to_images, rankings, max_k)

    write_series_map_csv(series_to_map, args.output_csv)


if __name__ == "__main__":
    main()


