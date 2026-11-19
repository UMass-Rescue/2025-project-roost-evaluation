import os
import sys
import requests
from requests import RequestException
import json
import time
from pathlib import Path
from hma.image.series_labels_utils import get_image_files, ensure_labels_file
from hma.logging_utils import setup_logging, get_logger, get_log_file, _log_debug, _log_info, _log_warning, _log_error  # noqa: F401 (get_logger is re-exported to the suites)
from hma.hma_client import HMAClient, match_url
from datetime import datetime

image_input_dir = Path("./hma/image/resources/images")

LABELS_PATH_ENV = "LABELS_PATH"

# Signal type to use - defaults to clip_float
# Can be overridden via SIGNAL_TYPE env var
SIGNAL_TYPE = os.getenv("SIGNAL_TYPE", "clip_float")

class Evaluator(HMAClient):
    def match_uploaded_files(self, files_to_send, signal_type: str):
        """Match each uploaded file using batch hashing + per-file lookup."""
        _log_info("Sleeping 35 seconds to allow in-memory index cache to refresh...")
        time.sleep(35)
        batch_size = int(os.getenv("HASH_BATCH_SIZE", "32"))
        _log_info(f"Batch hashing {len(files_to_send)} files for matching (batch_size={batch_size})...")
        batch_results = self.hash_local_content_batch(files_to_send, signal_type=signal_type, batch_size=batch_size)
        for match_file_path, hash_resp in zip(files_to_send, batch_results):
            _log_debug(match_file_path)
            if not isinstance(hash_resp, dict) or signal_type not in hash_resp:
                _log_warning(f"Hash failed for {match_file_path}: {hash_resp}")
                continue
            signal = hash_resp[signal_type]
            data = {
                'signal_type': signal_type,
                'signal': signal
            }
            try:
                response = requests.post(match_url, json=data)
                if response.ok:
                    match_resp = {
                        'status': 'success',
                        'matches': response.json(),
                        'signal_type': signal_type,
                        'signal': signal
                    }
                else:
                    match_resp = {'status': 'failure', 'error': response.text}
            except RequestException as e:
                match_resp = {'status': 'failure', 'error': str(e)}
            _log_debug(json.dumps(match_resp, indent=2))


def calculate_metrics(results_dir):
    """Calculate MAP (from retrieval), classification PR (from pairwise), and distance plots."""
    # Import here to avoid circular dependency
    from hma.metrics.map import compute_map_from_retrieval_csv
    from hma.metrics.precision_recall import compute_precision_recall_from_pairwise
    from hma.metrics.distance_distribution import compute_distance_distribution
    from hma.metrics.common import validate_series_metadata_exists
    
    labels_path = Path(os.getenv(LABELS_PATH_ENV, "hma/image/resources/labels/images_series_labels.json"))
    image_dir = Path(os.environ.get("IMAGE_INPUT_DIR", str(image_input_dir)))
    labels_path = ensure_labels_file(labels_path, image_dir)
    
    try:
        validate_series_metadata_exists(str(labels_path))
    except ValueError as e:
        print(f"✗ Cannot calculate metrics: {e}")
        return
    
    # Get anon_id_map path using same logic as PathIdStore.from_env()
    anon_map_override = os.getenv("ANON_ID_MAP_FILEPATH")
    if anon_map_override and anon_map_override.strip():
        anon_map_path = Path(anon_map_override.strip())
    else:
        output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
        anon_map_path = output_root / "file_to_id_map" / "anon_id_map.json"
    
    # Look for CSV file only
    pairwise_file = results_dir / f"pairwise_{SIGNAL_TYPE}_compare.csv"
    
    if not pairwise_file.exists():
        _log_warning(f"Pairwise results not found: {pairwise_file}")
        return
    
    map_output_csv = results_dir / f"map_by_series_{SIGNAL_TYPE}_results.csv"
    topk_csv = results_dir / f"topk_test_{SIGNAL_TYPE}_results.csv"
    threshold_csv = results_dir / f"threshold_test_{SIGNAL_TYPE}_results.csv"
    
    # Find retrieval CSV (prefer topk, fallback to threshold)
    retrieval_csv = topk_csv if topk_csv.exists() else (threshold_csv if threshold_csv.exists() else None)
    result_type = "topk" if topk_csv.exists() else ("threshold" if threshold_csv.exists() else None)
    
    if not retrieval_csv:
        _log_warning(f"No retrieval results found (topk or threshold) for {SIGNAL_TYPE}")
    else:
        # Load content_id -> image mapping from saved file (created during upload)
        from hma.results_utils import load_content_id_mapping
        content_id_to_image = load_content_id_mapping(results_dir, SIGNAL_TYPE)
        
        if content_id_to_image:
            _log_info(f"Loaded content_id mapping ({len(content_id_to_image)} entries)")
        else:
            _log_warning("Content ID mapping file not found or empty. MAP calculation may be inaccurate.")
            content_id_to_image = {}
        
        try:
            _log_info(f"Computing MAP@k from {result_type} retrieval results for {SIGNAL_TYPE}...")
            print(f"  MAP@k ({result_type}) for {SIGNAL_TYPE}...", end=" ", flush=True)
            compute_map_from_retrieval_csv(
                str(retrieval_csv),
                str(labels_path),
                str(map_output_csv),
                result_type=result_type,
                anon_map_path=str(anon_map_path) if anon_map_path.exists() else None,
                content_id_to_image=content_id_to_image
            )
            print("✓")
            _log_info(f"Saved to: {map_output_csv}")
            print(f"    → {map_output_csv.name}")
        except Exception as e:
            print(f"✗ {e}")
            _log_error(f"Failed to compute MAP for {SIGNAL_TYPE}: {e}")
    
    # Compute classification Precision-Recall from pairwise - measures distance quality
    pr_csv = results_dir / f"precision_recall_{SIGNAL_TYPE}_results.csv"
    pr_plot = results_dir / f"precision_recall_{SIGNAL_TYPE}_curve.png"
    try:
        _log_info(f"Computing classification Precision-Recall from pairwise for {SIGNAL_TYPE}...")
        print(f"  Classification PR (pairwise) for {SIGNAL_TYPE}...", end=" ", flush=True)
        compute_precision_recall_from_pairwise(
            str(labels_path),
            str(pairwise_file),
            str(pr_csv),
            str(pr_plot),
            str(anon_map_path) if anon_map_path.exists() else None,
            SIGNAL_TYPE,
        )
        print("✓")
        _log_info(f"Saved CSV: {pr_csv}")
        _log_info(f"Saved plot: {pr_plot}")
        print(f"    → {pr_csv.name}")
        print(f"    → {pr_plot.name}")
    except Exception as e:
        print(f"✗ {e}")
        _log_error(f"Failed to compute classification PR for {SIGNAL_TYPE}: {e}")
    
    # Generate distance distribution histograms
    dist_output_plot = results_dir / f"distance_distribution_{SIGNAL_TYPE}.png"
    try:
        _log_info(f"Generating distance distribution plot for {SIGNAL_TYPE}...")
        print(f"  Distance distribution for {SIGNAL_TYPE}...", end=" ", flush=True)
        compute_distance_distribution(
            str(labels_path),
            str(pairwise_file),
            str(dist_output_plot),
            SIGNAL_TYPE,
            str(anon_map_path) if anon_map_path.exists() else None
        )
        print("✓")
        _log_info(f"Saved plot: {dist_output_plot}")
        print(f"    → {dist_output_plot.name}")
    except Exception as e:
        print(f"✗ {e}")
        _log_error(f"Failed to generate distance distribution for {SIGNAL_TYPE}: {e}")

def run_all_tests():
    start_time = time.time()
    setup_logging("test")
    _log_info("[STARTUP] Creating fresh database for test run...")
    print("Running tests...")
    
    # Import here to avoid circular dependency
    from hma.metrics.common import validate_series_metadata_exists
    try:
        validate_series_metadata_exists()
    except ValueError as e:
        print(f"✗ {e}")
        raise
    
    # Generate timestamp for this test run and pass to subprocesses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["TEST_RUN_TIMESTAMP"] = timestamp
    _log_info(f"Test run timestamp: {timestamp}")
    
    evaluator = Evaluator()
    evaluator.cleanup_test_environment()
    
    # Setup bank and upload images for tests that need index (topk, threshold)
    BANK_NAME = os.getenv("BANK_NAME", "TEST_BANK_DATA")
    _log_info("Setting up bank and uploading images for test run...")
    if not evaluator.setup_bank(BANK_NAME):
        _log_error("Failed to setup bank. Exiting.")
        return
    
    image_dir = Path(os.environ.get("IMAGE_INPUT_DIR", str(image_input_dir)))
    files_to_send = get_image_files(image_dir)
    print(f"Uploading {len(files_to_send)} images to bank...")
    content_id_to_image = evaluator.upload_files_to_bank(files_to_send, BANK_NAME)
    
    # Save content_id -> image_path mapping to results directory for later use in metrics
    from hma.results_utils import save_content_id_mapping
    content_id_map_file = save_content_id_mapping(content_id_to_image, SIGNAL_TYPE)
    _log_info(f"Saved content_id mapping to {content_id_map_file} ({len(content_id_to_image)} entries)")
    
    # Wait for index to update (longer timeout for large datasets)
    index_size_before = evaluator.get_index_size(SIGNAL_TYPE)
    expected_size = index_size_before + len(files_to_send)
    max_wait = 180 if len(files_to_send) > 100 else 60  # 3 min for large datasets, 1 min for small
    evaluator.wait_for_index_update(expected_size, SIGNAL_TYPE, max_wait=max_wait)
    _log_info(f"{SIGNAL_TYPE} index updated. Current size: {evaluator.get_index_size(SIGNAL_TYPE)}")
    
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_files = [f for f in os.listdir(test_dir) if f.endswith("_test.py")]
    _log_info(f"Found {len(test_files)} test files: {test_files}")
    print(f"Found {len(test_files)} test files")
    # Flush to ensure output is visible
    sys.stdout.flush()
    
    _log_info(f"Running tests with signal_type={SIGNAL_TYPE}")
    os.environ["SIGNAL_TYPE"] = SIGNAL_TYPE
    
    for i, fname in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {fname} ({SIGNAL_TYPE})")
        _log_info(f"Running {fname} ...")
        
        # Import and run test directly (no subprocess - much simpler!)
        test_name = fname.replace("_test.py", "").replace("_", " ").title()
        print(f"  {test_name}: ", end="", flush=True)
        
        try:
            # Import the test module directly
            module_name = fname.replace(".py", "")
            test_module = __import__(f"tests.{module_name}", fromlist=[module_name])
            
            # Run the test's main function directly
            if hasattr(test_module, 'main'):
                test_module.main()
                print()  # Newline after test completes
                _log_info(f"Test {fname} completed successfully")
            else:
                _log_error(f"Test {fname} has no main() function")
                raise ValueError(f"Test {fname} has no main() function")
                
        except Exception as e:
            print()  # Newline on error
            _log_error(f"Test {fname} failed: {e}")
            import traceback
            _log_error(traceback.format_exc())
            raise
    
    print(f"✓ All tests completed. Logs: {get_log_file()}")
    
    # Calculate MAP metrics
    print("\nCalculating MAP metrics...")
    _log_info("Calculating MAP metrics...")
    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    results_dir = output_root / "evaluation_results" / timestamp
    calculate_metrics(results_dir)
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"\n⏱  Total time elapsed: {elapsed_minutes}m {elapsed_seconds}s")
    _log_info(f"Total time elapsed: {elapsed_minutes}m {elapsed_seconds}s ({elapsed_time:.2f}s)")

def main():
    eval_mode = os.environ.get("EVAL_MODE", "smoke")
    test_run_type = "smoke" if eval_mode == "smoke" else "test"
    setup_logging(test_run_type)
    
    if eval_mode == "smoke":
        _log_info("[STARTUP] Creating fresh database for smoke test...")
        print("Running smoke test...")
        evaluator = Evaluator()
        evaluator.cleanup_test_environment()
        
        BANK_NAME = os.getenv("BANK_NAME", "TEST_BANK_DATA")
        if not evaluator.setup_bank(BANK_NAME):
            return

        image_dir = Path(os.environ.get("IMAGE_INPUT_DIR", str(image_input_dir)))
        files_to_send = get_image_files(image_dir)
        print(f"Uploading {len(files_to_send)} files...")
        content_id_to_image = evaluator.upload_files_to_bank(files_to_send, BANK_NAME)
        
        # Save content_id -> image_path mapping
        from hma.results_utils import save_content_id_mapping
        content_id_map_file = save_content_id_mapping(content_id_to_image, SIGNAL_TYPE)
        _log_info(f"Saved content_id mapping to {content_id_map_file} ({len(content_id_to_image)} entries)")
        
        _log_info(f"Testing signal_type={SIGNAL_TYPE}")
        index_size_before = evaluator.get_index_size(SIGNAL_TYPE)
        expected_size = index_size_before + len(files_to_send)
        evaluator.wait_for_index_update(expected_size, SIGNAL_TYPE)
        evaluator.match_uploaded_files(files_to_send, SIGNAL_TYPE)
        
        print(f"✓ Smoke test completed. Logs: {get_log_file()}")
    else:
        run_all_tests()

if __name__ == '__main__':
    main()
