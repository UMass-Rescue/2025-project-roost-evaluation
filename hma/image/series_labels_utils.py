"""Image series labels: the shared logic in hma/series_labels_utils.py with the image file types."""

from pathlib import Path

from hma import series_labels_utils as shared

DEFAULT_LABELS_FILENAME = "images_series_labels.json"

_IMAGE_ROOT = Path(__file__).resolve().parent


def get_image_files(image_dir: Path) -> list[str]:
    """Return sorted list of image file paths (auto-recursive if subdirs exist)."""
    return shared.get_media_files(image_dir, shared.IMAGE_EXTENSIONS)


def ensure_labels_file(labels_path: Path, image_dir: Path) -> Path:
    default_labels_path = _IMAGE_ROOT / "resources/labels" / DEFAULT_LABELS_FILENAME
    return shared.ensure_labels_file(
        labels_path, image_dir, shared.IMAGE_EXTENSIONS, default_labels_path, _IMAGE_ROOT
    )
