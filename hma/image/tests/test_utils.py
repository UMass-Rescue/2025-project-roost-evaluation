import os
from pathlib import Path
from hma.image.series_labels_utils import get_image_files as _get_image_files

def get_image_files(image_dir=None):
    """Return sorted list of image file paths from the given directory (default: hma/image/resources/images or $IMAGE_INPUT_DIR)."""
    if image_dir is None:
        image_dir = Path(os.environ.get("IMAGE_INPUT_DIR", "./hma/image/resources/images"))
    else:
        image_dir = Path(image_dir)
    return _get_image_files(image_dir)

def hash_image(evaluator, image_path):
    """Return the hash dict for a given image file using the Evaluator."""
    return evaluator.hash_local_content(image_path)

def hash_images_batch(evaluator, image_paths, signal_type=None, batch_size=32):
    """Return list of hash dicts for multiple image files using batch hashing."""
    return evaluator.hash_local_content_batch(
        image_paths, signal_type=signal_type, batch_size=batch_size
    )

def match_image(evaluator, image_path):
    """Return the match result dict for a given image file using the Evaluator."""
    return evaluator.match_local_content(image_path)
