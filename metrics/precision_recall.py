import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.common import load_and_validate_data, build_predictions_by_image, ensure_output_dir
from tests.test_utils import build_rankings


# ----------------------------
# Precision/Recall computation
# ----------------------------


def precision_at_k(preds: List[str], positives: Set[str], k: int) -> float:
    """Compute Precision@k = (# relevant in top-k) / k"""
    if k == 0:
        return 0.0
    relevant_count = sum(1 for p in preds[:k] if p in positives)
    return relevant_count / k


def recall_at_k(preds: List[str], positives: Set[str], k: int) -> float:
    """Compute Recall@k = (# relevant in top-k) / (total # relevant)"""
    if not positives:
        return 0.0
    relevant_count = sum(1 for p in preds[:k] if p in positives)
    return relevant_count / len(positives)


def compute_series_precision_recall(
    series_to_images: Dict[str, Set[str]],
    rankings: Dict[str, List[Tuple[str, float]]],
    max_k: int,
) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Compute Precision@k and Recall@k for k=1..max_k for each query in each series.
    Returns: {series: {query_image: [(prec, rec) for k in 1..max_k]}}
    """
    series_to_query_pr: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    
    for series, images in series_to_images.items():
        if len(images) <= 1:
            raise ValueError(f"Series '{series}' needs at least 2 images, has {len(images)}.")
        
        preds_by_image = build_predictions_by_image(images, rankings)
        query_to_pr: Dict[str, List[Tuple[float, float]]] = {}
        
        for query in images:
            positives = set(images) - {query}
            preds = preds_by_image[query]
            
            pr_by_k: List[Tuple[float, float]] = []
            for k in range(1, max_k + 1):
                prec = precision_at_k(preds, positives, k)
                rec = recall_at_k(preds, positives, k)
                pr_by_k.append((prec, rec))
            
            query_to_pr[query] = pr_by_k
        
        series_to_query_pr[series] = query_to_pr
    
    return series_to_query_pr


# ----------------------------
# Output
# ----------------------------


def plot_precision_recall_curves(
    series_to_query_pr: Dict[str, Dict[str, List[Tuple[float, float]]]], 
    output_path: str
) -> None:
    """Generate precision-recall curve plot showing individual query curves grouped by series."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define color palette for series
    colors = plt.cm.tab10(range(len(series_to_query_pr)))
    
    for (series, query_to_pr), color in zip(sorted(series_to_query_pr.items()), colors):
        # Plot each query in this series with same color but lighter
        for i, (query, pr_values) in enumerate(query_to_pr.items()):
            recalls = [rec for _, rec in pr_values]
            precisions = [prec for prec, _ in pr_values]
            
            # First query gets label for legend, others don't
            label = series if i == 0 else None
            plt.plot(recalls, precisions, color=color, alpha=0.4, linewidth=1.5, label=label)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (Individual Queries by Series)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    out_path = ensure_output_dir(output_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def write_precision_recall_csv(
    series_to_pr: Dict[str, List[Tuple[float, float]]], 
    output_csv: str
) -> None:
    """Write precision-recall data to CSV (series, k, precision, recall format)."""
    out_path = ensure_output_dir(output_csv)
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["series", "k", "precision", "recall"])
        writer.writeheader()
        
        for series, pr_values in sorted(series_to_pr.items()):
            for k, (prec, rec) in enumerate(pr_values, start=1):
                writer.writerow({
                    "series": series,
                    "k": k,
                    "precision": f"{prec:.6f}",
                    "recall": f"{rec:.6f}"
                })


def compute_precision_recall_from_pairwise(
    labels_path: str, 
    pairwise_path: str, 
    output_csv: str,
    output_plot: Optional[str] = None,
    anon_map_path: Optional[str] = None
) -> None:
    """Compute Precision-Recall@k from pairwise results and generate outputs."""
    series_to_images, entries, id_to_path = load_and_validate_data(
        labels_path, pairwise_path, anon_map_path
    )
    
    rankings = build_rankings(entries, id_to_path)
    max_k = max(len(v) for v in series_to_images.values())
    series_to_query_pr = compute_series_precision_recall(series_to_images, rankings, max_k)
    
    write_precision_recall_csv(series_to_query_pr, output_csv)
    
    if output_plot:
        plot_precision_recall_curves(series_to_query_pr, output_plot)


# ----------------------------
# CLI
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute Precision-Recall@k curves from pairwise distances."
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
        "--output_plot",
        help="Output plot path (PNG/PDF). If not provided, no plot is generated."
    )
    parser.add_argument(
        "--anon_map",
        help="Optional path to anon ID map JSON."
    )
    args = parser.parse_args()
    
    compute_precision_recall_from_pairwise(
        args.labels, 
        args.pairwise, 
        args.output_csv,
        args.output_plot,
        args.anon_map
    )


if __name__ == "__main__":
    main()
