import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tests.test_utils import get_image_files, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

def main():
    # Read signal_type from env (fresh each call)
    SIGNAL_TYPE = os.getenv("SIGNAL_TYPE", "clip_float")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", f"topk_test_{SIGNAL_TYPE}_results.json")
    MAX_K = int(os.getenv("MAX_K", 5))
    
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    k_values = range(1, MAX_K + 1)
    print(f"[INFO] Found {len(image_files)} images. Starting top-k match test with k values={list(k_values)}...")
    _log_info(f"Found {len(image_files)} images. Starting top-k match test with k values={list(k_values)}...")

    def test_topk(item):
        """Worker function to test a single image-k combination."""
        k, img = item
        match_resp = evaluator.match_local_content_topk(img, k, SIGNAL_TYPE)

        if match_resp.get("status") == "success":
            return {
                "image": img,
                "k": k,
                "matches": match_resp.get("matches", [])
            }
        else:
            return {
                "image": img,
                "k": k,
                "error": match_resp.get("error"),
                "response": match_resp.get("response")
            }
    
    # Create list of all test combinations
    test_items = [(k, img) for k in k_values for img in image_files]
    total_tests = len(test_items)
    _log_info(f"Testing {total_tests} image-k combinations with parallelization...")
    
    # Parallel processing with ThreadPoolExecutor
    max_workers = int(os.getenv("MAX_WORKERS", "8"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_topk, item): item for item in test_items}
        
        for future in tqdm(as_completed(futures), total=total_tests, desc="Top-k test progress", file=sys.stderr, ncols=80):
            try:
                result = future.result()
                results.append(result)
                _log_debug(json.dumps(result, indent=2))
            except Exception as e:
                _log_warning(f"Top-k test failed: {e}")

    output_path = write_results(results, OUTPUT_FILE)
    print(f"[INFO] Top-k match test complete. Results saved to {output_path}")
    _log_info(f"Top-k match test complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
