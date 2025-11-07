import os
import itertools
import json
from test_utils import get_image_files, hash_image, write_results, decode_clip_hex_to_floats
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pairwise_clip_compare.json")
SIGNAL_TYPE = "clip"

def main():
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    # Use print for important messages so they're captured by parent process
    print(f"[INFO] Found {len(image_files)} images. Starting pairwise CLIP hash comparison...")
    _log_info(f"Found {len(image_files)} images. Starting pairwise CLIP hash comparison...")

    # Cache hashes
    image_hashes = {}
    print("[INFO] Caching image hashes...")
    _log_info("Caching image hashes...")
    for img in image_files:
        resp = hash_image(evaluator, img)
        if isinstance(resp, dict) and SIGNAL_TYPE in resp:
            image_hashes[img] = resp[SIGNAL_TYPE]
            _log_debug(f"Cached hash for {img}")
        else:
            print(f"[WARN] Failed to hash {img}")
            _log_warning(f"Failed to hash {img}: {resp}")

    results = []
    total_pairs = len(list(itertools.combinations(image_files, 2)))
    pair_count = 0
    print(f"[INFO] Comparing {total_pairs} image pairs...")
    for img1, img2 in itertools.combinations(image_files, 2):
        pair_count += 1
        print(f"[{pair_count}/{total_pairs}] Comparing {img1} <--> {img2}")
        _log_info(f"[{pair_count}/{total_pairs}] Comparing {img1} <--> {img2}")

        clip1 = image_hashes.get(img1)
        clip2 = image_hashes.get(img2)

        if not clip1 or not clip2:
            print(f"[WARN] Missing hashes for {img1} or {img2}. Skipping.")
            _log_warning(f"Missing hashes for {img1} or {img2}. Skipping.")
            continue

        compare_resp = evaluator.compare_hashes(clip1, clip2, signal_type=SIGNAL_TYPE)

        matched = False
        distance = None
        if compare_resp.get("status") == "success":
            try:
                distance = compare_resp["result"][1]
                matched = compare_resp["result"][0]
                print(f"✓ Comparison: matched={matched}, distance={distance}")
                _log_info(f"✓ Comparison: matched={matched}, distance={distance}")
            except (ValueError, TypeError):
                print(f"[WARN] Failed to parse compare result")
                _log_warning(f"Failed to parse compare result: {compare_resp}")
        else:
            error_msg = f"Compare API failed: {compare_resp.get('error', 'Unknown error')}"
            print(f"[WARN] {error_msg}")
            _log_warning(error_msg)

        result = {
            "image1": img1,
            "image2": img2,
            "matched": matched,
            "distance": distance,
        }
        results.append(result)
        _log_debug(json.dumps(result, indent=2))

    write_results(results, OUTPUT_FILE)
    print(f"[INFO] Pairwise CLIP comparison complete. Results saved to {OUTPUT_FILE}")
    _log_info(f"Pairwise CLIP comparison complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
