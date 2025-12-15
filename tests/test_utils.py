"""Test utilities for evaluation tests."""
import os
import json
import binascii
import numpy as np
from pathlib import Path
from datetime import datetime
from evaluate import image_input_dir
from tests.path_id_store import PathIdStore

ANON_ENV_FLAG = "DEANONYMIZE_IMAGE_PATHS"


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

def get_results_dir():
    """Get the results directory path based on timestamp, creating it if needed."""
    # Use TEST_RUN_TIMESTAMP if set (from parent process), otherwise generate new one
    timestamp = os.getenv("TEST_RUN_TIMESTAMP")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    results_dir = output_root / "evaluation_results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def _should_anonymize() -> bool:
    # Default ON; set DEANONYMIZE_IMAGE_PATHS=1 to disable anonymization
    return os.getenv(ANON_ENV_FLAG, "").strip() != "1"

def _collect_image_paths(results: list[dict]) -> set[str]:
    paths: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("image", "image1", "image2"):
            val = item.get(key)
            if isinstance(val, str):
                paths.add(val)
    return paths

def _anonymize_results(results: list[dict], id_map: dict[str, str]) -> list[dict]:
    anonymized: list[dict] = []
    for item in results:
        new_item = item.copy()
        for key in ("image", "image1", "image2"):
            val = new_item.get(key)
            if isinstance(val, str):
                if val not in id_map:
                    raise ValueError(f"Missing anonymized ID for key '{key}' with value '{val}'")
                new_item[key] = id_map[val]
        anonymized.append(new_item)
    return anonymized

def _validate_results_for_anonymization(results) -> None:
    if not isinstance(results, list):
        raise TypeError("Anonymization expects 'results' to be a list of dicts.")
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError(f"Anonymization expects dict items; found {type(item).__name__} at index {idx}.")

def write_results(results, filename):
    """Write results (list of dicts) to a JSON file in the timestamped results directory."""
    results_dir = get_results_dir()
    output_path = results_dir / filename
    results_to_write = results

    if _should_anonymize():
        _validate_results_for_anonymization(results)
        paths = _collect_image_paths(results)
        # Build IDs using persistent store if ANON_ID_MAP_FILEPATH is set, else in-memory
        store = PathIdStore.from_env()
        id_map = store.get_ids_for_paths(paths)
        results_to_write = _anonymize_results(results, id_map)
        if store.has_persistence:
            store.save()

    with open(output_path, 'w') as f:
        json.dump(results_to_write, f, indent=2)
    return str(output_path)

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