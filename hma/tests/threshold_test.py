import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tests.test_utils import get_image_files, hash_images_batch, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

def main():
    # Read signal_type from env (fresh each call)
    SIGNAL_TYPE = os.getenv("SIGNAL_TYPE", "clip_float")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", f"threshold_test_{SIGNAL_TYPE}_results.csv")
    
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    
    # Determine threshold configuration based on signal type
    # clip uses int (0-100), clip_float and cliphnsw use float (0.0-1.0)
    if SIGNAL_TYPE == "clip":
        max_threshold = int(os.getenv("THRESHOLD_MAX", "100"))
        step = int(os.getenv("THRESHOLD_STEP", "20"))
        thresholds = list(range(0, max_threshold + 1, step))
    else:  # clip_float, cliphnsw
        import numpy as np
        max_threshold = float(os.getenv("THRESHOLD_MAX", "1.0"))
        step = float(os.getenv("THRESHOLD_STEP", "0.2"))
        num_steps = int(max_threshold / step) + 1
        thresholds = np.linspace(0.0, max_threshold, num_steps).tolist()
    print(f"[INFO] Found {len(image_files)} images. Starting threshold match test with thresholds={list(thresholds)}...")
    _log_info(f"Found {len(image_files)} images. Starting threshold match test with thresholds={list(thresholds)}...")

    # Batch hash all images upfront
    batch_size = int(os.getenv("HASH_BATCH_SIZE", "32"))
    _log_info(f"Batch hashing {len(image_files)} images (batch_size={batch_size})...")
    batch_results = hash_images_batch(evaluator, image_files, signal_type=SIGNAL_TYPE, batch_size=batch_size)
    image_hashes = {}
    for img, resp in zip(image_files, batch_results):
        if isinstance(resp, dict) and SIGNAL_TYPE in resp:
            image_hashes[img] = resp[SIGNAL_TYPE]
        else:
            _log_warning(f"Batch hash failed for {img}: {resp}")
    _log_info(f"Batch hashing complete: {len(image_hashes)}/{len(image_files)} successful")

    def test_threshold(item):
        """Worker function to test a single image-threshold combination using cached hash."""
        threshold, img = item
        signal = image_hashes.get(img)
        if not signal:
            return {
                "image": img,
                "threshold": threshold,
                "error": f"No cached hash for {img}",
                "response": ""
            }
        match_resp = evaluator.match_with_signal_threshold(signal, threshold, SIGNAL_TYPE)

        if match_resp.get("status") == "success":
            return {
                "image": img,
                "threshold": threshold,
                "matches": match_resp.get("matches", [])
            }
        else:
            return {
                "image": img,
                "threshold": threshold,
                "error": match_resp.get("error"),
                "response": match_resp.get("response")
            }
    
    # Create list of all test combinations
    test_items = [(threshold, img) for threshold in thresholds for img in image_files]
    total_tests = len(test_items)
    _log_info(f"Testing {total_tests} image-threshold combinations with parallelization...")
    
    # Parallel processing with ThreadPoolExecutor
    max_workers = int(os.getenv("MAX_WORKERS", "8"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_threshold, item): item for item in test_items}
        
        for future in tqdm(as_completed(futures), total=total_tests, desc="Threshold test progress", file=sys.stderr, ncols=80):
            try:
                result = future.result()
                results.append(result)
                num_matches = len(result.get('matches', []))
                _log_debug(f"Threshold {result.get('threshold')} for {result.get('image')}: {num_matches} matches")
            except Exception as e:
                _log_warning(f"Threshold test failed: {e}")

    output_path = write_results(results, OUTPUT_FILE)
    print(f"[INFO] Threshold match test complete. Results saved to {output_path}")
    _log_info(f"Threshold match test complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
