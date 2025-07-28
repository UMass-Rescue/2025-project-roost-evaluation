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

def extract_matches(matches):
    """Helper function to extract and format matches from HMA API response."""
    processed_matches = []
    
    if isinstance(matches, dict):
        for bank_name, bank_matches in matches.items():
            if isinstance(bank_matches, list):
                for match in bank_matches:
                    if isinstance(match, dict):
                        try:
                            distance = float(match.get('distance', 'inf'))
                            content_id = match.get('bank_content_id')
                            
                            match_obj = {
                                'content_id': content_id,
                                'bank_name': bank_name,
                                'distance': distance
                            }
                                
                            processed_matches.append(match_obj)
                        except (ValueError, TypeError):
                            continue
    
    return processed_matches

def create_result(img, clip_hash, processed_matches, **kwargs):
    """Create a standardized result object."""
    result = {
        'image': os.path.basename(img),
        'clip_hash': clip_hash,
    }
    
    result['matches'] = processed_matches
    
    # Add any remaining kwargs
    result.update(kwargs)
    return result

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

def process_image(evaluator, img, process_function, **kwargs):
    try:
        match_result = match_image(evaluator, img)
        clip_hash = match_result.get('signal', '') if match_result.get('status') == 'success' else ''

        if match_result.get('status') == 'success' and 'matches' in match_result:
            matches = match_result['matches']
            processed_matches = process_function(matches, **kwargs)

            for match in processed_matches:
                if match.get('content_id'):
                    content_id = match['content_id']
                    content_result = evaluator.get_signal_from_contentid(content_id, "TEST_BANK_DATA")
                    match['clip_hash'] = content_result.get('data', {}).get('signals', {}).get('clip', '') if content_result.get('status') == 'success' else ''

            return create_result(img, clip_hash, processed_matches, **kwargs)
        else:
            return create_result(img, clip_hash, [], error=match_result.get('error', 'No matches found'), **kwargs)

    except Exception as e:
        print(f"[WARN] Failed to process {img}: {e}")
        return create_result(img, '', [], error=str(e), **kwargs)