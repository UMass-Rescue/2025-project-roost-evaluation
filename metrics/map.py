import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_utils import (
    normalize_label_path,
    normalize_pairwise_path,
    load_labels,
    load_pairwise,
    load_anon_id_map,
    collect_pairwise_paths,
    validate_inputs,
    extract_distance,
    build_rankings,
)


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
            ranked = rankings.get(img, [])
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


def write_series_map_csv(
    series_to_map: Dict[str, List[float]], output_csv: str
) -> None:
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


def compute_map_from_pairwise(labels_path: str, pairwise_path: str, output_csv: str, anon_map_path: Optional[str] = None) -> None:
    """Compute mAP@k from pairwise results and write to CSV."""
    series_to_images = load_labels(labels_path)
    entries = load_pairwise(pairwise_path)
    id_to_path = load_anon_id_map(anon_map_path)
    
    validate_inputs(series_to_images, entries, id_to_path)
    
    rankings = build_rankings(entries, id_to_path)
    max_k = max(len(v) for v in series_to_images.values())
    series_to_map = compute_series_map(series_to_images, rankings, max_k)
    
    write_series_map_csv(series_to_map, output_csv)


# ----------------------------
# Main
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute mAP@k from pairwise image distances and series labels."
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels JSON (series -> list of image paths).",
    )
    parser.add_argument(
        "--pairwise",
        required=True,
        help="Path to pairwise results JSON from pairwise_test.py.",
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--anon_map",
        required=False,
        help="Optional path to anon ID map JSON ({path: id}); used to map IDs back to paths.",
    )
    args = parser.parse_args()
    
    compute_map_from_pairwise(args.labels, args.pairwise, args.output_csv, args.anon_map)


if __name__ == "__main__":
    main()
