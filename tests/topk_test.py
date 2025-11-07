import os
import json
import sys
from tqdm import tqdm
from tests.test_utils import get_image_files, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "topk_test_results.json")
SIGNAL_TYPE = "clip"
MAX_K = int(os.getenv("MAX_K", 5))

def main():
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    k_values = range(1, MAX_K + 1)
    print(f"[INFO] Found {len(image_files)} images. Starting top-k match test with k values={list(k_values)}...")
    _log_info(f"Found {len(image_files)} images. Starting top-k match test with k values={list(k_values)}...")

    results = []
    total_tests = len(k_values) * len(image_files)
    
    # Create list of all test combinations for tqdm
    test_items = [(k, img) for k in k_values for img in image_files]
    
    # tqdm for terminal progress, _log_info for log file
    for k, img in tqdm(test_items, desc="Top-k test progress", file=sys.stderr, ncols=80, disable=False):
        _log_info(f"Matching {img} with k={k}")
        
        match_resp = evaluator.match_local_content_topk(img, k)

        if match_resp.get("status") == "success":
            matches_count = len(match_resp.get("matches", []))
            result = {
                "image": img,
                "k": k,
                "matches": match_resp.get("matches", [])
            }
            _log_info(f"✓ Success: Found {matches_count} matches for {img} with k={k}")
        else:
            _log_warning(f"Top-k match API failed for {img} with k={k}: {match_resp.get('error', 'Unknown error')}")
            result = {
                "image": img,
                "k": k,
                "error": match_resp.get("error"),
                "response": match_resp.get("response")
            }
        
        results.append(result)
        _log_debug(json.dumps(result, indent=2))

    output_path = write_results(results, OUTPUT_FILE)
    print(f"[INFO] Top-k match test complete. Results saved to {output_path}")
    _log_info(f"Top-k match test complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
