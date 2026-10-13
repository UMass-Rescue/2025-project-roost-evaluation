"""Mean Average Precision (mAP) computation for image retrieval."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from metrics.common import (
    load_and_validate_data, ensure_output_dir, normalize_pairwise_path,
    load_labels, load_anon_id_map, build_image_to_series_map
)
from metrics.retrieval_pr import load_retrieval_results_csv


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

        if b not in neighbors[a] or d < neighbors[a][b]:
            neighbors[a][b] = d
        if a not in neighbors[b] or d < neighbors[b][a]:
            neighbors[b][a] = d

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for img, nbrs in neighbors.items():
        rankings[img] = sorted(nbrs.items(), key=lambda kv: kv[1])
    return rankings


def build_predictions_by_image(
    images: Set[str],
    rankings: Dict[str, List[Tuple[str, float]]]
) -> Dict[str, List[str]]:
    """
    Build per-image prediction lists from rankings.
    Returns: dict mapping image -> list of neighbor paths (excluding self).
    """
    preds_by_image: Dict[str, List[str]] = {}
    for img in images:
        ranked = rankings.get(img, [])
        preds_by_image[img] = [nbr for (nbr, _) in ranked if nbr != img]
    return preds_by_image


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
    for idx, p in enumerate(preds, start=1):
        if p in positives:
            hits += 1
            sum_precisions += hits / idx
            if hits == denom:
                break
        if idx == k:
            break
    return sum_precisions / denom


def mean_average_precision_at_k(
    preds_by_image: Dict[str, List[str]],
    images: Set[str],
    k: int,
) -> float:
    """Compute mAP@k over a set of query images."""
    if not images:
        raise ValueError("Cannot compute mAP@k for an empty set of images.")

    ap_values: List[float] = []
    for query in images:
        positives = set(images) - {query}
        preds = preds_by_image[query]
        ap = average_precision_at_k(preds, positives, k)
        ap_values.append(ap)
    return sum(ap_values) / len(ap_values)


def compute_series_map(
    series_to_images: Dict[str, Set[str]],
    rankings: Dict[str, List[Tuple[str, float]]],
    max_k: int,
) -> Dict[str, List[float]]:
    """Compute mAP@k for k=1..max_k for each series."""
    series_to_map: Dict[str, List[float]] = {}
    
    for series, images in series_to_images.items():
        if len(images) <= 1:
            raise ValueError(f"Series '{series}' needs at least 2 images, has {len(images)}.")
        
        preds_by_image = build_predictions_by_image(images, rankings)
        
        ap_by_k: List[float] = []
        for k in range(1, max_k + 1):
            series_map_k = mean_average_precision_at_k(preds_by_image, images, k)
            ap_by_k.append(series_map_k)
        series_to_map[series] = ap_by_k
    
    return series_to_map


# ----------------------------
# Output
# ----------------------------


def write_series_map_csv(
    series_to_map: Dict[str, List[float]], output_csv: str
) -> None:
    """Write mAP results to CSV with mean row."""
    out_path = ensure_output_dir(output_csv)
    max_k = max((len(v) for v in series_to_map.values()), default=0)
    fieldnames = ["series"] + [f"map@{k}" for k in range(1, max_k + 1)]
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write per-series rows and accumulate for mean
        sums = [0.0] * max_k
        counts = [0] * max_k
        
        for series, values in series_to_map.items():
            row = {"series": series}
            for i, v in enumerate(values, start=1):
                row[f"map@{i}"] = f"{v:.6f}"
                sums[i - 1] += v
                counts[i - 1] += 1
            writer.writerow(row)
        
        # Write mean row
        if max_k > 0 and any(c > 0 for c in counts):
            avg_row = {"series": "mean"}
            for i in range(max_k):
                avg = (sums[i] / counts[i]) if counts[i] > 0 else 0.0
                avg_row[f"map@{i+1}"] = f"{avg:.6f}"
            writer.writerow(avg_row)


def compute_map_from_pairwise(
    labels_path: str, 
    pairwise_path: str, 
    output_csv: str, 
    anon_map_path: Optional[str] = None
) -> None:
    """Compute mAP@k from pairwise results and write to CSV."""
    series_to_images, entries, id_to_path = load_and_validate_data(
        labels_path, pairwise_path, anon_map_path
    )
    
    rankings = build_rankings(entries, id_to_path)
    max_k = max(len(v) for v in series_to_images.values())
    series_to_map = compute_series_map(series_to_images, rankings, max_k)
    
    write_series_map_csv(series_to_map, output_csv)


def compute_map_from_retrieval_csv(
    csv_path: str,
    labels_path: str,
    output_csv: str,
    result_type: str = "topk",
    anon_map_path: Optional[str] = None,
    content_id_to_image: Optional[Dict[str, str]] = None
) -> None:
    """
    Compute mAP@k from retrieval results (topk/threshold CSV).
    
    This measures actual index retrieval performance, not just distance quality.
    
    Args:
        csv_path: Path to retrieval results CSV (topk or threshold test output)
        labels_path: Path to labels JSON
        output_csv: Output CSV path for MAP results
        result_type: "topk" or "threshold"
        anon_map_path: Optional anon ID map
        content_id_to_image: Mapping from content_id to image path
    """
    # Load data
    series_to_images = load_labels(labels_path)
    image_to_series = build_image_to_series_map(series_to_images)
    id_to_path = load_anon_id_map(anon_map_path)
    
    if not content_id_to_image:
        content_id_to_image = {}
    
    # Load retrieval results (reuse existing function)
    results_by_query = load_retrieval_results_csv(csv_path, result_type)
    
    # Build rankings from retrieval results
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    
    for query_image, matches in results_by_query.items():
        query_src = id_to_path.get(query_image, query_image) if id_to_path else query_image
        query_normalized = normalize_pairwise_path(query_src)
        
        if query_normalized not in image_to_series:
            continue
        
        # Map content_ids to image paths
        ranked_neighbors: List[Tuple[str, float]] = []
        for match in matches:
            content_id = str(match["bank_content_id"])
            if content_id in content_id_to_image:
                matched_path = content_id_to_image[content_id]
                matched_normalized = normalize_pairwise_path(matched_path)
                ranked_neighbors.append((matched_normalized, match["distance"]))
        
        if ranked_neighbors:
            rankings[query_normalized] = ranked_neighbors
    
    # Compute MAP@k for each series
    max_k = max(len(v) for v in series_to_images.values())
    series_to_map = compute_series_map(series_to_images, rankings, max_k)
    
    # Write results
    write_series_map_csv(series_to_map, output_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mAP@k from pairwise image distances and series labels."
    )
    parser.add_argument(
        "--labels", required=True,
        help="Path to labels JSON (series -> list of image paths)."
    )
    parser.add_argument(
        "--pairwise", required=True,
        help="Path to pairwise results JSON."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Output CSV path."
    )
    parser.add_argument(
        "--anon_map",
        help="Optional path to anon ID map JSON."
    )
    args = parser.parse_args()
    
    compute_map_from_pairwise(args.labels, args.pairwise, args.output_csv, args.anon_map)


if __name__ == "__main__":
    main()
