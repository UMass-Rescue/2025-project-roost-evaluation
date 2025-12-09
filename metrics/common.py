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


def ensure_output_dir(output_path: str) -> Path:
    """Create parent directories for output path if they don't exist."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
