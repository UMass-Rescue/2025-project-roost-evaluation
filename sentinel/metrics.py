"""
Precision/recall threshold sweep for Sentinel's rare_class_affinity_score.

Higher scores indicate more affinity to the rare (grooming) class, so a
conversation is predicted "grooming" when its score is >= threshold.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

POSITIVE_LABEL = "grooming"


def _sweep_thresholds(labeled_scores: List[Tuple[float, bool]], thresholds: List[float]) -> List[dict]:
    """Compute precision/recall/FPR at each score threshold.

    A score >= threshold is predicted positive (grooming).
    """
    results: List[dict] = []
    total_pos = sum(1 for _, is_pos in labeled_scores if is_pos)
    total_neg = len(labeled_scores) - total_pos

    for t in thresholds:
        tp = fp = 0
        for score, is_pos in labeled_scores:
            if score >= t:
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
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
            }
        )
    return results


def _auto_thresholds(labeled_scores: List[Tuple[float, bool]], num: int = 200) -> List[float]:
    """Generate evenly spaced thresholds over the observed score range."""
    if not labeled_scores:
        return []
    scores = [s for s, _ in labeled_scores]
    s_min, s_max = min(scores), max(scores)
    if s_min == s_max:
        return [s_min]
    return np.linspace(s_min, s_max, num).tolist()


def _write_csv(rows: List[dict], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["threshold", "precision", "recall", "fpr", "tp", "fp", "fn", "tn"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threshold": f"{row['threshold']:.6f}",
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "fpr": f"{row['fpr']:.6f}",
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "tn": row["tn"],
                }
            )


def _plot_pr(rows: List[dict], output_plot: str) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return

    recalls = np.array([r["recall"] for r in rows])
    precisions = np.array([r["precision"] for r in rows])

    # Add initial point at (0, 1.0) for recall=0.
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Interpolate to get monotonically decreasing precision (convex hull):
    # for each recall level, use the maximum precision from that point on.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Sentinel Precision-Recall (rare_class_affinity_score threshold sweep)")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    out_path = Path(output_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_precision_recall(
    labeled_scores: List[Tuple[float, str]],
    output_csv: str,
    output_plot: Optional[str] = None,
) -> List[dict]:
    """Compute a precision/recall threshold sweep from (score, ground_truth_label) pairs.

    ground_truth_label is expected to be "grooming" or "innocuous" (e.g. from
    sentinel.dataset.load_10l_eval_set()). Writes a CSV of threshold/
    precision/recall/fpr/tp/fp/fn/tn rows, and optionally a PR curve plot.
    Returns the same rows that were written to the CSV.
    """
    labeled_bools = [(score, label == POSITIVE_LABEL) for score, label in labeled_scores]
    thresholds = _auto_thresholds(labeled_bools)
    rows = _sweep_thresholds(labeled_bools, thresholds)
    _write_csv(rows, output_csv)
    if output_plot:
        _plot_pr(rows, output_plot)
    return rows
