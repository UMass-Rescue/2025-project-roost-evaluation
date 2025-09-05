import os
import json
import binascii
import numpy as np
from pathlib import Path
from evaluate import image_input_dir

def get_image_files(image_dir=None):
    """Return sorted list of image file paths from the given directory (default: resources/images or $IMAGE_INPUT_DIR)."""
    if image_dir is None:
        image_dir = os.environ.get("IMAGE_INPUT_DIR", str(image_input_dir))
    return sorted([str(f) for f in Path(image_dir).iterdir() if f.is_file()])

def hash_image(evaluator, image_path):
    """Return the hash dict for a given image file using the Evaluator."""
    return evaluator.hash_local_content(image_path)

def match_image(evaluator, image_path):
    """Return the match result dict for a given image file using the Evaluator."""
    return evaluator.match_local_content(image_path)

def write_results(results, filename):
    """Write results (list of dicts) to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def decode_clip_hex(hex_string: str) -> list[float] | None:
    """
    Convert HMA-style hex string (representing float32 vector) to a float list.
    """
    try:
        byte_data = binascii.unhexlify(hex_string)
        float_array = np.frombuffer(byte_data, dtype=np.float32)
        return float_array.tolist()
    except Exception as e:
        print(f"[ERROR] Failed to decode CLIP hex string: {e}")
        return None

def cosine_distance_safe(vec1, vec2):
    """Compute cosine distance with safety fallback."""
    from scipy.spatial.distance import cosine
    try:
        return cosine(vec1, vec2)
    except Exception as e:
        print(f"[ERROR] Failed distance computation: {e}")
        return None


def decode_clip_hex_to_floats(hex_str: str) -> list[float]:
    print(f"[DEBUG] CLIP hex length: {len(hex_str)}")
    byte_data = bytes.fromhex(hex_str)
    decoded = np.frombuffer(byte_data, dtype=np.float32).tolist()
    print(f"[DEBUG] Decoded CLIP vector length: {len(decoded)}")
    return decoded