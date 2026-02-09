import os
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tests.test_utils import get_image_files, hash_image, write_results
from evaluate import Evaluator, get_logger, _log_info, _log_debug, _log_warning

def main():
    # Read signal_type from env (fresh each call)
    SIGNAL_TYPE = os.getenv("SIGNAL_TYPE", "clip_float")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", f"pairwise_{SIGNAL_TYPE}_compare.csv")
    
    # Use existing logger if available, otherwise create a simple one
    logger = get_logger()
    evaluator = Evaluator()
    image_files = get_image_files()
    # Print to terminal, log to file
    print(f"[INFO] Found {len(image_files)} images. Starting pairwise CLIP hash comparison...")
    _log_info(f"Found {len(image_files)} images. Starting pairwise CLIP hash comparison...")

    # Cache hashes
    image_hashes = {}
    _log_info("Caching image hashes...")
    for img in image_files:
        resp = hash_image(evaluator, img)
        if isinstance(resp, dict) and SIGNAL_TYPE in resp:
            image_hashes[img] = resp[SIGNAL_TYPE]
            _log_debug(f"Cached hash for {img}")
        else:
            # Hash succeeded but signal type not found - show available signal types
            available_types = list(resp.keys()) if isinstance(resp, dict) else "unknown"
            _log_warning(f"Hash response for {img} does not contain signal type '{SIGNAL_TYPE}'. Available types: {available_types}. Response: {resp}")

    def compare_pair(pair):
        """Worker function to compare a single pair of images."""
        img1, img2 = pair
        clip1 = image_hashes.get(img1)
        clip2 = image_hashes.get(img2)

        if not clip1 or not clip2:
            return {
                "image1": img1,
                "image2": img2,
                "matched": False,
                "distance": None,
            }

        compare_resp = evaluator.compare_hashes(clip1, clip2, signal_type=SIGNAL_TYPE)

        matched = False
        distance = None
        if compare_resp.get("status") == "success":
            try:
                distance = compare_resp["result"][1]
                matched = compare_resp["result"][0]
            except (ValueError, TypeError):
                pass

        return {
            "image1": img1,
            "image2": img2,
            "matched": matched,
            "distance": distance,
        }

    pairs = list(itertools.combinations(image_files, 2))
    total_pairs = len(pairs)
    _log_info(f"Comparing {total_pairs} image pairs with parallelization...")
    
    # Parallel processing with ThreadPoolExecutor
    max_workers = int(os.getenv("MAX_WORKERS", "8"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compare_pair, pair): pair for pair in pairs}
        
        for future in tqdm(as_completed(futures), total=total_pairs, desc="Pairwise test progress", file=sys.stderr, ncols=80):
            try:
                result = future.result()
                results.append(result)
                _log_debug(f"Compared: {result['image1']} <-> {result['image2']}, distance={result['distance']}, matched={result['matched']}")
            except Exception as e:
                _log_warning(f"Pair comparison failed: {e}")

    output_path = write_results(results, OUTPUT_FILE)
    print(f"[INFO] Pairwise CLIP comparison complete. Results saved to {output_path}")
    _log_info(f"Pairwise CLIP comparison complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
