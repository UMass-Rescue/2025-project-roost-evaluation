import os
import json
import sys
from tqdm import tqdm
from tests.test_utils import get_image_files, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "threshold_test_results.json")
SIGNAL_TYPE = "clip_float"
THRESHOLD_MAX = int(os.getenv("THRESHOLD_MAX", 100))
THRESHOLD_STEP = int(os.getenv("THRESHOLD_STEP", 20))

def main():
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    thresholds = range(0, THRESHOLD_MAX, THRESHOLD_STEP)
    print(f"[INFO] Found {len(image_files)} images. Starting threshold match test with thresholds={list(thresholds)}...")
    _log_info(f"Found {len(image_files)} images. Starting threshold match test with thresholds={list(thresholds)}...")

    results = []
    total_tests = len(thresholds) * len(image_files)
    
    # Create list of all test combinations for tqdm
    test_items = [(threshold, img) for threshold in thresholds for img in image_files]
    
    # tqdm for terminal progress, _log_info for log file
    for threshold, img in tqdm(test_items, desc="Threshold test progress", file=sys.stderr, ncols=80, disable=False):
        _log_info(f"Matching {img} with threshold={threshold}")
        
        match_resp = evaluator.match_local_content_threshold(img, threshold)

        if match_resp.get("status") == "success":
            matches_count = len(match_resp.get("matches", []))
            result = {
                "image": img,
                "threshold": threshold,
                "matches": match_resp.get("matches", [])
            }
            _log_info(f"✓ Success: Found {matches_count} matches for {img} with threshold={threshold}")
        else:
            _log_warning(f"Threshold match API failed for {img} with threshold={threshold}: {match_resp.get('error', 'Unknown error')}")
            result = {
                "image": img,
                "threshold": threshold,
                "error": match_resp.get("error"),
                "response": match_resp.get("response")
            }
        
        results.append(result)
        _log_debug(json.dumps(result, indent=2))

    output_path = write_results(results, OUTPUT_FILE)
    print(f"[INFO] Threshold match test complete. Results saved to {output_path}")
    _log_info(f"Threshold match test complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
