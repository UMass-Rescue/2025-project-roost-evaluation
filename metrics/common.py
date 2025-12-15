"""Common utilities for metrics computation."""
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_utils import (
    load_labels,
    load_pairwise,
    load_anon_id_map,
    validate_inputs,
    build_rankings,
    build_image_to_series_map,
    normalize_pairwise_path,
)


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
        img1_src = id_to_path.get(entry["image1"], entry["image1"]) if id_to_path else entry["image1"]
        img2_src = id_to_path.get(entry["image2"], entry["image2"]) if id_to_path else entry["image2"]
        
        img1 = normalize_pairwise_path(img1_src)
        img2 = normalize_pairwise_path(img2_src)
        
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
