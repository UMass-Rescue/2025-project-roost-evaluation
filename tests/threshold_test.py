import os
import json
import sys
from tqdm import tqdm
from tests.test_utils import get_image_files, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "threshold_test_results.json")
SIGNAL_TYPE = "clip_float"

# Different threshold configurations for different signal types
# PDQ and CLIP use integers, CLIP_FLOAT uses floats (0.0-1.0)
INT_CONFIG = {
    "max": int(os.getenv("THRESHOLD_MAX_INT", "100")),
    "step": int(os.getenv("THRESHOLD_STEP_INT", "20")),
    "type": "int"
}

THRESHOLD_CONFIG = {
    "pdq": INT_CONFIG,
    "clip": INT_CONFIG,
    "clip_float": {
        "max": float(os.getenv("THRESHOLD_MAX_FLOAT", "1.0")),
        "step": float(os.getenv("THRESHOLD_STEP_FLOAT", "0.2")),
        "start": float(os.getenv("THRESHOLD_START_FLOAT", "0.1")),  # Don't start at 0.0
        "type": "float"
    }
}

def main():
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    
    # Get threshold configuration for this signal type
    config = THRESHOLD_CONFIG.get(SIGNAL_TYPE, THRESHOLD_CONFIG["clip_float"])
    
    if config["type"] == "int":
        thresholds = list(range(0, config["max"] + 1, config["step"]))
    else:  # float
        import numpy as np
        start = config.get("start", 0.0)
        thresholds = np.arange(start, config["max"] + config["step"]/2, config["step"]).tolist()
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
