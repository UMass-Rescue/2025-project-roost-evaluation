import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from metrics.common import load_and_validate_data, ensure_output_dir, classify_pairwise_by_series


def classify_pairwise_distances(
    series_to_images: Dict[str, Set[str]],
    pairwise_entries: List[dict],
    id_to_path: Optional[Dict[str, str]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Classify pairwise distances into same-series and different-series.
    Returns: (same_series_distances, different_series_distances)
    """
    labeled = classify_pairwise_by_series(series_to_images, pairwise_entries, id_to_path)
    
    same_series_distances = [distance for distance, is_same in labeled if is_same]
    different_series_distances = [distance for distance, is_same in labeled if not is_same]
    
    return same_series_distances, different_series_distances


def plot_distance_distributions(
    same_series_distances: List[float],
    different_series_distances: List[float],
    output_path: str,
    signal_type: str = ""
) -> None:
    """Generate histogram plot comparing same-series vs different-series distances."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = f" ({signal_type})" if signal_type else ""
    
    # Same-series histogram
    ax1.hist(same_series_distances, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Distance', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Same Series Pairs{title_suffix}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'n={len(same_series_distances)}', 
             transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Different-series histogram
    ax2.hist(different_series_distances, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Different Series Pairs{title_suffix}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.95, f'n={len(different_series_distances)}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Distance Distribution: Same vs Different Series{title_suffix}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = ensure_output_dir(output_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_distance_distribution(
    labels_path: str,
    pairwise_path: str,
    output_plot: str,
    signal_type: str = "",
    anon_map_path: Optional[str] = None
) -> None:
    """Compute and plot distance distributions from pairwise results."""
    series_to_images, entries, id_to_path = load_and_validate_data(
        labels_path, pairwise_path, anon_map_path
    )
    
    same_distances, diff_distances = classify_pairwise_distances(
        series_to_images, entries, id_to_path
    )
    
    print(f"Same-series pairs: {len(same_distances)}")
    print(f"Different-series pairs: {len(diff_distances)}")
    
    plot_distance_distributions(same_distances, diff_distances, output_plot, signal_type)


def main():
    parser = argparse.ArgumentParser(
        description="Generate distance distribution histograms from pairwise results."
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
        "--output_plot", required=True,
        help="Output plot path (PNG/PDF)."
    )
    parser.add_argument(
        "--signal_type", default="",
        help="Signal type label for plot title (e.g., 'clip', 'clip_float')."
    )
    parser.add_argument(
        "--anon_map",
        help="Optional path to anon ID map JSON."
    )
    args = parser.parse_args()
    
    compute_distance_distribution(
        args.labels,
        args.pairwise,
        args.output_plot,
        args.signal_type,
        args.anon_map
    )


if __name__ == "__main__":
    main()
