"""Retrieval Precision-Recall metrics for topk/threshold test results."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from metrics.common import (
    load_labels,
    load_anon_id_map,
    normalize_pairwise_path,
    ensure_output_dir,
    build_image_to_series_map,
)


def load_retrieval_results_csv(csv_path: str, result_type: str = "threshold") -> Dict[str, List[dict]]:
    """
    Load retrieval results from CSV (topk or threshold test output).
    
    Args:
        csv_path: Path to CSV file
        result_type: "threshold" or "topk"
    
    Returns:
        Dict mapping query_image -> list of matches with bank_content_id and distance
    """
    results_by_query: Dict[str, List[dict]] = {}
    
    with open(csv_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        
        required_cols = {"image", "bank_content_id", "distance"}
        param_col = "threshold" if result_type == "threshold" else "k"
        required_cols.add(param_col)
        
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must have columns: {required_cols}. Found: {reader.fieldnames}"
            )
        
        for row in reader:
            query_image = row["image"].strip()
            bank_content_id = row["bank_content_id"].strip()
            distance_str = row["distance"].strip()
            
            if not query_image or not bank_content_id or not distance_str:
                continue
            
            try:
                distance = float(distance_str)
            except ValueError:
                continue
            
            if query_image not in results_by_query:
                results_by_query[query_image] = []
            
            results_by_query[query_image].append({
                "bank_content_id": bank_content_id,
                "distance": distance,
                param_col: row.get(param_col, "")
            })
    
    # Sort matches by distance for each query
    for query_image in results_by_query:
        results_by_query[query_image].sort(key=lambda x: x["distance"])
    
    return results_by_query


def compute_retrieval_pr_curve(
    results_by_query: Dict[str, List[dict]],
    series_to_images: Dict[str, Set[str]],
    image_to_series: Dict[str, str],
    content_id_to_image: Dict[str, str],
    id_to_path: Optional[Dict[str, str]] = None,
    max_k: int = 100,
    result_type: str = "topk"
) -> List[dict]:
    """
    Compute retrieval Precision-Recall curve.
    
    Two retrieval strategies:
    - topk: Approximate search - returns top K matches regardless of distance
    - threshold: Exact search - returns all matches within distance threshold
    
    For each query image:
    - Positive: Images in the same series (excluding query itself)
    - Ranked results: Matches sorted by distance
    
    Args:
        results_by_query: Query image -> ranked list of matches
        series_to_images: Series name -> set of image paths
        image_to_series: Image path -> series name
        content_id_to_image: Mapping from content_id (string) to image path
        id_to_path: Optional mapping from anonymous IDs to actual paths
        max_k: Maximum k to evaluate (for topk) or max matches (for threshold)
        result_type: "topk" or "threshold"
    
    Returns:
        List of dicts with precision, recall, and k/threshold_idx values
    """
    all_precisions: List[List[float]] = []
    all_recalls: List[List[float]] = []
    k_values = list(range(1, max_k + 1))
    
    for query_image, matches in results_by_query.items():
        # Normalize query image path
        query_src = id_to_path.get(query_image, query_image) if id_to_path else query_image
        query_normalized = normalize_pairwise_path(query_src)
        
        # Get positive set (same series, excluding query)
        if query_normalized not in image_to_series:
            continue
        
        query_series = image_to_series[query_normalized]
        positives = series_to_images[query_series] - {query_normalized}
        
        if not positives:
            continue  # Skip if no other images in same series
        
        # Map content_ids to image paths, keeping track of distances
        matched_images: List[str] = []
        matched_distances: List[float] = []
        for match in matches[:max_k]:
            content_id = str(match["bank_content_id"])
            if content_id in content_id_to_image:
                matched_path = content_id_to_image[content_id]
                matched_normalized = normalize_pairwise_path(matched_path)
                matched_images.append(matched_normalized)
                matched_distances.append(match["distance"])
        
        if not matched_images:
            continue  # Skip if no valid matches
        
        if result_type == "threshold":
            # For threshold-based: evaluate at different distance thresholds
            if not matched_distances:
                continue
            
            # Generate thresholds from min to max distance
            min_dist = min(matched_distances)
            max_dist = max(matched_distances)
            num_thresholds = min(50, len(set(matched_distances)))  # Use up to 50 unique thresholds
            thresholds = np.linspace(min_dist, max_dist, num_thresholds).tolist()
            
            precisions = []
            recalls = []
            
            for threshold in thresholds:
                # Get all matches within threshold (sorted by distance)
                within_threshold = [img for i, img in enumerate(matched_images) if matched_distances[i] <= threshold]
                unique_matches = set(within_threshold)
                tp_k = sum(1 for img in unique_matches if img in positives)
                
                precision = tp_k / len(within_threshold) if len(within_threshold) > 0 else 0.0
                recall = tp_k / len(positives) if len(positives) > 0 else 0.0
                recall = min(recall, 1.0)
                
                precisions.append(precision)
                recalls.append(recall)
        else:
            # For topk-based: evaluate at different k values
            precisions = []
            recalls = []
            
            for k in k_values:
                if k > len(matched_images):
                    break
                
                # Count true positives in top-k (deduplicate matches)
                top_k_matches = matched_images[:k]
                unique_matches = set(top_k_matches)
                tp_k = sum(1 for img in unique_matches if img in positives)
                
                precision = tp_k / k if k > 0 else 0.0
                recall = tp_k / len(positives) if len(positives) > 0 else 0.0
                recall = min(recall, 1.0)
                
                precisions.append(precision)
                recalls.append(recall)
        
        if precisions:
            all_precisions.append(precisions)
            all_recalls.append(recalls)
    
    # Average across all queries
    if not all_precisions:
        return []
    
    # Handle variable-length lists by padding to max length
    max_len = max(len(p) for p in all_precisions) if all_precisions else 0
    if max_len == 0:
        return []
    
    # Pad all lists to same length with last value
    padded_precisions = []
    padded_recalls = []
    for prec, rec in zip(all_precisions, all_recalls):
        if len(prec) < max_len:
            prec = prec + [prec[-1]] * (max_len - len(prec)) if prec else [0.0] * max_len
            rec = rec + [rec[-1]] * (max_len - len(rec)) if rec else [0.0] * max_len
        padded_precisions.append(prec[:max_len])
        padded_recalls.append(rec[:max_len])
    
    avg_precisions = np.mean(padded_precisions, axis=0).tolist()
    avg_recalls = np.mean(padded_recalls, axis=0).tolist()
    
    results = []
    if result_type == "threshold":
        # For threshold, we need to map back to threshold values
        # Since we averaged, we'll use indices as approximate thresholds
        for idx, (prec, rec) in enumerate(zip(avg_precisions, avg_recalls)):
            results.append({
                "threshold_idx": idx,
                "precision": prec,
                "recall": rec
            })
    else:
        # For topk, use k values
        for k, prec, rec in zip(k_values[:len(avg_precisions)], avg_precisions, avg_recalls):
            results.append({
                "k": k,
                "precision": prec,
                "recall": rec
            })
    
    return results


def compute_retrieval_pr_from_csv(
    csv_path: str,
    labels_path: str,
    output_csv: str,
    output_plot: Optional[str] = None,
    result_type: str = "topk",
    anon_map_path: Optional[str] = None,
    signal_type: str = "",
    max_k: int = 100,
    content_id_to_image: Optional[Dict[str, str]] = None
) -> None:
    """
    Compute retrieval Precision-Recall from topk/threshold CSV results.

    Args:
        csv_path: Path to CSV with retrieval results
        labels_path: Path to labels JSON
        output_csv: Output CSV path
        output_plot: Optional output plot path
        result_type: "topk" or "threshold"
        anon_map_path: Optional anon ID map
        signal_type: Signal type label for plots
        max_k: Maximum k to evaluate
        content_id_to_image: Optional mapping of content_id -> image path (from upload)
    """
    # Load data
    results_by_query = load_retrieval_results_csv(csv_path, result_type)
    series_to_images = load_labels(labels_path)
    image_to_series = build_image_to_series_map(series_to_images)
    id_to_path = load_anon_id_map(anon_map_path)

    if content_id_to_image is None:
        content_id_to_image = {}
    
    # Compute PR curve
    pr_data = compute_retrieval_pr_curve(
        results_by_query,
        series_to_images,
        image_to_series,
        content_id_to_image,
        id_to_path,
        max_k,
        result_type
    )
    
    if not pr_data:
        print("Warning: No PR data computed. Check content_id -> image mapping.")
        return
    
    # Write CSV
    out_csv_path = ensure_output_dir(output_csv)
    fieldnames = ["threshold_idx", "precision", "recall"] if result_type == "threshold" else ["k", "precision", "recall"]
    with open(out_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pr_data:
            writer.writerow({
                fieldnames[0]: row.get("threshold_idx", row.get("k", "")),
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}"
            })
    
    # Plot if requested
    if output_plot and MATPLOTLIB_AVAILABLE:
        recalls = np.array([r["recall"] for r in pr_data])
        precisions = np.array([r["precision"] for r in pr_data])
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        # Apply convex hull interpolation (monotonically decreasing precision)
        # For each recall level, use the maximum precision from that point onward
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker="o", linewidth=1.5, markersize=3)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Retrieval Precision-Recall [{signal_type}] ({result_type})")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.grid(True, alpha=0.3)
        
        out_plot_path = ensure_output_dir(output_plot)
        plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute retrieval PR from topk/threshold CSV results."
    )
    parser.add_argument("--csv", required=True, help="Path to retrieval results CSV")
    parser.add_argument("--labels", required=True, help="Path to labels JSON")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--output_plot", help="Optional output plot path")
    parser.add_argument("--result_type", choices=["topk", "threshold"], default="topk",
                       help="Type of retrieval test")
    parser.add_argument("--anon_map", help="Optional anon ID map JSON")
    parser.add_argument("--signal_type", default="", help="Signal type label for plots")
    parser.add_argument("--max_k", type=int, default=100, help="Maximum k to evaluate")
    parser.add_argument("--content_id_map", help="Path to content_id -> image_path JSON (from upload)")
    args = parser.parse_args()

    content_id_to_image = {}
    if args.content_id_map:
        import json
        with open(args.content_id_map) as f:
            content_id_to_image = json.load(f)

    compute_retrieval_pr_from_csv(
        args.csv,
        args.labels,
        args.output_csv,
        args.output_plot,
        args.result_type,
        args.anon_map,
        args.signal_type,
        args.max_k,
        content_id_to_image
    )


if __name__ == "__main__":
    main()
