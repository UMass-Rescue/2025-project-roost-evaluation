import os
import json
import csv
import binascii
import numpy as np
from pathlib import Path
from datetime import datetime
from hma.path_id_store import PathIdStore

ANON_ENV_FLAG = "DEANONYMIZE_IMAGE_PATHS"

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

def save_content_id_mapping(content_id_to_image: dict, signal_type: str, name: str = "content_id_to_image") -> Path:
    """
    Save content_id -> image_path mapping to results directory.
    
    Args:
        content_id_to_image: Dict mapping content_id (str) -> image_path (str)
        signal_type: Signal type (e.g., 'clip_float')
        name: File name stem, so other evaluations can keep their own mapping
    
    Returns:
        Path to the saved mapping file
    """
    results_dir = get_results_dir()
    content_id_map_file = results_dir / f"{name}_{signal_type}.json"
    with open(content_id_map_file, 'w') as f:
        json.dump(content_id_to_image, f, indent=2)
    return content_id_map_file

def load_content_id_mapping(results_dir: Path, signal_type: str, name: str = "content_id_to_image") -> dict:
    """
    Load content_id -> image_path mapping from results directory.
    
    Args:
        results_dir: Path to results directory
        signal_type: Signal type (e.g., 'clip_float')
        name: File name stem used when the mapping was saved
    
    Returns:
        Dict mapping content_id (str) -> image_path (str), or empty dict if not found
    """
    content_id_map_file = results_dir / f"{name}_{signal_type}.json"
    if content_id_map_file.exists():
        try:
            with open(content_id_map_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            # Logging will be handled by caller
            return {}
    return {}

def _should_anonymize() -> bool:
    # Default ON; set DEANONYMIZE_IMAGE_PATHS=1 to disable anonymization
    return os.getenv(ANON_ENV_FLAG, "").strip() != "1"

def _item_keys(item_key: str) -> tuple[str, str, str]:
    """Result keys holding a file path: the item itself, or either side of a pair."""
    return (item_key, f"{item_key}1", f"{item_key}2")

def _collect_item_paths(results: list[dict], item_key: str = "image") -> set[str]:
    paths: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in _item_keys(item_key):
            val = item.get(key)
            if isinstance(val, str):
                paths.add(val)
    return paths

def _anonymize_results(results: list[dict], id_map: dict[str, str], item_key: str = "image") -> list[dict]:
    anonymized: list[dict] = []
    for item in results:
        new_item = item.copy()
        for key in _item_keys(item_key):
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

def read_csv_rows(path) -> list[dict]:
    """The rows of a CSV file with a header, as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows) -> None:
    """Write dict rows (at least one) to a CSV file whose header is the first row's keys."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(results, filename, item_key="image"):
    """Write results (list of dicts) to a CSV file in the timestamped results directory.

    item_key names the result key (and CSV column) holding the queried file path:
    "image" for the image evaluation, "video" for the video evaluation. Pairwise
    results use "<item_key>1" and "<item_key>2".
    """
    results_dir = get_results_dir()
    output_path = results_dir / filename.replace('.json', '.csv')
    results_to_write = results

    if _should_anonymize():
        _validate_results_for_anonymization(results)
        paths = _collect_item_paths(results, item_key)
        store = PathIdStore.from_env()
        id_map = store.get_ids_for_paths(paths)
        results_to_write = _anonymize_results(results, id_map, item_key)
        if store.has_persistence:
            store.save()

    # Write CSV instead of JSON
    is_pairwise = filename.startswith("pairwise_")
    if is_pairwise:
        key1, key2 = f"{item_key}1", f"{item_key}2"
        fieldnames = [key1, key2, "distance", "matched"]
        rows = []
        for r in results_to_write:
            distance = r.get("distance")
            # Extract distance value - handle dict format from compare API
            if isinstance(distance, dict):
                distance = distance.get("distance", distance)
            # Only include rows with valid distances
            if distance is not None and distance != "":
                distance_str = str(distance)
                rows.append({
                    key1: r.get(key1, ""), 
                    key2: r.get(key2, ""), 
                    "distance": distance_str, 
                    "matched": r.get("matched", False)
                })
    else:
        param_name = "threshold" if "threshold" in filename else "k"
        fieldnames = [item_key, param_name, "bank_content_id", "distance"]
        rows = []
        for r in results_to_write:
            for m in r.get("matches", []):
                rows.append({item_key: r.get(item_key, ""), param_name: r.get(param_name, ""),
                            "bank_content_id": m.get("bank_content_id", ""), 
                            "distance": m.get("distance", "")})
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
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
