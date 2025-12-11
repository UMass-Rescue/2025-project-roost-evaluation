import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.common import load_and_validate_data, ensure_output_dir
from tests.test_utils import normalize_pairwise_path, build_image_to_series_map


def _collect_labeled_distances(
    series_to_images: Dict[str, set],
    pairwise_entries: List[dict],
    id_to_path: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, bool]]:
    """
    Return a list of (distance, is_positive) where positive = same series.
    """
    image_to_series = build_image_to_series_map(series_to_images)
    labeled: List[Tuple[float, bool]] = []

    for entry in pairwise_entries:
        img1_raw = entry["image1"]
        img2_raw = entry["image2"]
        img1_src = id_to_path.get(img1_raw, img1_raw) if id_to_path else img1_raw
        img2_src = id_to_path.get(img2_raw, img2_raw) if id_to_path else img2_raw

        img1 = normalize_pairwise_path(img1_src)
        img2 = normalize_pairwise_path(img2_src)

        if img1 not in image_to_series or img2 not in image_to_series:
            continue

        same_series = image_to_series[img1] == image_to_series[img2]
        labeled.append((float(entry["distance"]), same_series))

    return labeled


def _sweep_thresholds(
    labeled_distances: List[Tuple[float, bool]], thresholds: List[float]
) -> List[dict]:
    """
    Compute precision/recall/FPR at each distance threshold.
    """
    results: List[dict] = []
    total_pos = sum(1 for _, is_pos in labeled_distances if is_pos)
    total_neg = len(labeled_distances) - total_pos

    for t in thresholds:
        tp = fp = 0
        for dist, is_pos in labeled_distances:
            if dist <= t:
                if is_pos:
                    tp += 1
                else:
                    fp += 1
        fn = total_pos - tp
        tn = total_neg - fp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        results.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
            }
        )
    return results


def _auto_thresholds(labeled_distances: List[Tuple[float, bool]], num: int = 200) -> List[float]:
    """
    Generate evenly spaced thresholds over the observed distance range.
    """
    if not labeled_distances:
        return []
    distances = [d for d, _ in labeled_distances]
    d_min, d_max = min(distances), max(distances)
    if d_min == d_max:
        return [d_min]
    return np.linspace(d_min, d_max, num).tolist()


def _write_csv(rows: List[dict], output_csv: str) -> None:
    import csv

    out_path = ensure_output_dir(output_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["threshold", "precision", "recall", "fpr", "tp", "fp", "fn"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threshold": row["threshold"],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "fpr": f"{row['fpr']:.6f}",
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                }
            )


def _plot_pr(rows: List[dict], output_plot: str, signal_type: str) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return
    recalls = [r["recall"] for r in rows]
    precisions = [r["precision"] for r in rows]
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker="o", linewidth=1.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Classification Precision-Recall (threshold sweep) [{signal_type}]")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    out_path = ensure_output_dir(output_plot)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_precision_recall_from_pairwise(
    labels_path: str,
    pairwise_path: str,
    output_csv: str,
    output_plot: Optional[str] = None,
    anon_map_path: Optional[str] = None,
    signal_type: str = "",
) -> None:
    series_to_images, entries, id_to_path = load_and_validate_data(
        labels_path, pairwise_path, anon_map_path
    )
    labeled = _collect_labeled_distances(series_to_images, entries, id_to_path)
    thresholds = _auto_thresholds(labeled)
    rows = _sweep_thresholds(labeled, thresholds)
    _write_csv(rows, output_csv)
    if output_plot:
        _plot_pr(rows, output_plot, signal_type)


def main():
    parser = argparse.ArgumentParser(
        description="Compute classification PR (threshold sweep) from pairwise distances."
    )
    parser.add_argument("--labels", required=True, help="Path to labels JSON.")
    parser.add_argument("--pairwise", required=True, help="Path to pairwise results JSON.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--output_plot", help="Optional output plot path.")
    parser.add_argument("--anon_map", help="Optional anon ID map JSON.")
    parser.add_argument("--signal_type", default="", help="Signal type label for plots.")
    args = parser.parse_args()

    compute_precision_recall_from_pairwise(
        args.labels,
        args.pairwise,
        args.output_csv,
        args.output_plot,
        args.anon_map,
        args.signal_type,
    )


if __name__ == "__main__":
    main()

