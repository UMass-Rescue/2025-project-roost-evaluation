import os
import sys
import requests
from requests import RequestException
import json
import time
from pathlib import Path
import logging
from series_labels_utils import get_image_files, ensure_labels_file
from datetime import datetime

# Try to import psycopg2 for database access (optional)
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

image_input_dir = Path("./resources/images")
hma_host = os.getenv("HMA_HOST", "host.docker.internal")
hma_port = os.getenv("HMA_PORT", "5005")
hma_app_url = f"http://{hma_host}:{hma_port}"
hash_url = hma_app_url +  "/h/hash"
hash_batch_url = hma_app_url + "/h/hash/batch"
match_url = hma_app_url + "/m/lookup"
match_url_topk = hma_app_url + "/m/lookup_topk"
match_url_threshold = hma_app_url + "/m/lookup_threshold"

LABELS_PATH_ENV = "LABELS_PATH"

# Signal type to use - defaults to clip_float
# Can be overridden via SIGNAL_TYPE env var
SIGNAL_TYPE = os.getenv("SIGNAL_TYPE", "clip_float")

# Setup logging
logger = None
log_file = None

def setup_logging(test_run_type="test"):
    """Setup file logging and minimal terminal output"""
    global logger, log_file
    
    # Create logs directory
    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    logs_dir = output_root / "test_run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename: TestRunType_Date_Timestarted
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{test_run_type}_{timestamp}.log"
    log_file = logs_dir / log_filename
    
    # Setup logger
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers
    
    # File handler - all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler - only WARNING and above, minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Direct logger access since we just initialized it
    logger.info(f"Logging to: {log_file}")
    # Flush to ensure the log file is created and accessible
    file_handler.flush()
    return logger

# Fallback logger if setup_logging hasn't been called
def get_logger():
    """Get logger - uses parent's logger if available, otherwise NullHandler"""
    global logger
    if logger is None:
        logger = logging.getLogger("evaluation")
        logger.setLevel(logging.DEBUG)
        # Check if logger already has handlers (from setup_logging in parent)
        if not logger.handlers:
            # No parent logger - use NullHandler (tests run directly, not as subprocess)
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
    return logger

# Use get_logger() wrapper to ensure logger is always available
def _log_debug(msg):
    get_logger().debug(msg)
def _log_info(msg):
    get_logger().info(msg)
def _log_warning(msg):
    get_logger().warning(msg)
def _log_error(msg):
    get_logger().error(msg)

class Evaluator:
    def bank_exists(self,bank_name: str) -> bool:
        """Check if a bank exists by making API call to HMA."""
        try:
            response = requests.get(f"{hma_app_url}/c/bank/{bank_name}")
            if response.ok:
                _log_debug(f"Bank {bank_name} exists")
                return True
            elif response.status_code == 404:
                _log_debug(f"Bank {bank_name} does not exist")
                return False
            else:
                _log_error(f"Failed to check bank existence: {response.status_code} - {response.text}")
                return False
                
        except RequestException as e:
            _log_error(f"Request exception while checking bank existence: {str(e)}")
            return False
        

    def create_bank(self, bank_name: str) :
        """
        Create a new bank.
        Returns True if bank was created successfully, False otherwise.
        """
        try:
            create_response = requests.post(
                f"{hma_app_url}/c/banks",
                json={
                    "name": bank_name,
                    "matching_enabled_ratio": 1.0
                }
            )
            if create_response.ok:
                _log_info(f"Successfully created bank {bank_name}")
                return True
            else:
                _log_error(f"Failed to create bank: {create_response.status_code} - {create_response.text}")
                return False
                
        except RequestException as e:
            _log_error(f"Request exception while creating bank: {str(e)}")
            return False


    def add_file_to_hma_bank(self, file_path: str, bank_name: str):
        """Add a file to the HMA bank and store its hash. Returns content_id if successful."""
        try:
            filename = os.path.basename(file_path)
            _log_debug(f"Adding {filename} to HMA bank and storing hash...")

            with open(file_path, 'rb') as f:
                files = {'photo': (filename, f)}
                response = requests.post(f"{hma_app_url}/c/bank/{bank_name}/content", files=files)
                if response.ok:
                    _log_debug(f"Successfully added {filename} to bank {bank_name}")
                    # Try to parse JSON response to extract content_id
                    content_id = None
                    try:
                        response_json = response.json()
                        # Content ID might be in different fields depending on API response format
                        content_id = response_json.get('id') or response_json.get('content_id') or response_json.get('bank_content_id')
                        if content_id is not None:
                            content_id = str(content_id)
                    except (json.JSONDecodeError, AttributeError):
                        # If response is not JSON or doesn't have expected fields, log it
                        _log_debug(f"Could not parse content_id from response: {response.text[:200]}")
                    
                    return {
                        'status': 'success', 
                        'response': response.text,
                        'content_id': content_id
                    }
                else:
                    _log_error(f"Failed to add {filename} to bank {bank_name}: {response.status_code} - {response.text}")
                    return {'status': 'failure', 'response': f"Failed for {filename}: {response.status_code} - {response.text}"}
        except RequestException as e:
            _log_error(f"Request exception while adding file to bank: {str(e)}")
            return {'status': 'failure', 'response': f"Request failed for {filename}: {e}"}
            

    def hash_local_content(self, file_path: str) -> dict:
        filename = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            files = {'photo': (filename, f)}
            try:
                response = requests.post(hash_url, files=files)
                if response.ok:
                    return response.json()
                else:
                    return {
                        'success': False,
                        'status_code': response.status_code,
                        'error': response.text
                    }
            except RequestException as e:
                return {
                    'success': False,
                    'status_code': 500,
                    'error': str(e)
                }

    def hash_local_content_batch(self, file_paths: list, signal_type: str = None, batch_size: int = 32) -> list:
        """Hash multiple files via /h/hash/batch. Returns list of hash dicts in input order.

        Falls back to individual hashing if the batch request fails.
        """
        all_results = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            file_handles = []
            try:
                files_payload = []
                for fp in batch:
                    fh = open(fp, 'rb')
                    file_handles.append(fh)
                    files_payload.append(('photo', (os.path.basename(fp), fh)))

                params = {}
                if signal_type:
                    params['signal_type'] = signal_type

                response = requests.post(hash_batch_url, files=files_payload, params=params)
                if response.ok:
                    batch_results = response.json()
                    if isinstance(batch_results, list) and len(batch_results) == len(batch):
                        all_results.extend(batch_results)
                    else:
                        _log_warning(f"Batch hash returned unexpected format, falling back to individual hashing")
                        for fp in batch:
                            all_results.append(self.hash_local_content(fp))
                else:
                    _log_warning(f"Batch hash failed ({response.status_code}), falling back to individual hashing")
                    for fp in batch:
                        all_results.append(self.hash_local_content(fp))
            except RequestException as e:
                _log_warning(f"Batch hash request exception: {e}, falling back to individual hashing")
                for fp in batch:
                    all_results.append(self.hash_local_content(fp))
            finally:
                for fh in file_handles:
                    fh.close()
        return all_results

    def match_with_signal_topk(self, signal: str, k: int, signal_type: str) -> dict:
        """Look up top-k matches using a pre-computed hash signal."""
        data = {
            'signal_type': signal_type,
            'signal': signal,
            'k': k
        }
        try:
            response = requests.post(match_url_topk, json=data)
            if response.ok:
                result = response.json()
                return {
                    'status': 'success',
                    'matches': result.get("matches", []),
                    'signal_type': signal_type,
                    'signal': signal
                }
            else:
                _log_debug(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }
        except RequestException as e:
            _log_debug(f"Request exception: {str(e)}")
            return {'status': 'failure', 'error': str(e)}

    def match_with_signal_threshold(self, signal: str, threshold, signal_type: str) -> dict:
        """Look up matches within a threshold using a pre-computed hash signal."""
        data = {
            'signal_type': signal_type,
            'signal': signal,
            'threshold': threshold
        }
        try:
            response = requests.post(match_url_threshold, json=data)
            if response.ok:
                result = response.json()
                return {
                    'status': 'success',
                    'matches': result.get("matches", []),
                    'signal_type': signal_type,
                    'signal': signal
                }
            else:
                _log_debug(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }
        except RequestException as e:
            _log_debug(f"Request exception: {str(e)}")
            return {'status': 'failure', 'error': str(e)}

    def match_local_content(self, file_path: str, signal_type: str) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        # signal_type defaults to 'clip_float'
        
        # Check if hash was successful and contains the signal type
        if not isinstance(hasher_resp, dict) or signal_type not in hasher_resp:
            return {
                'status': 'failure',
                'error': f'Failed to get {signal_type} hash',
                'response': str(hasher_resp)
            }
        
        signal = hasher_resp[signal_type]
        data = {
            'signal_type': signal_type,
            'signal': signal
        }
   
        try:
            response = requests.post(f"{match_url}", json=data)
            if response.ok:
                result = response.json()
                #print(json.dumps(result, indent=2))
                
                return {
                    'status': 'success',
                    'matches': result,  # List of matches with bank_content_id, distance, and bank_name
                    'signal_type': signal_type,
                    'signal': signal
                }
            else:
                _log_debug(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }
            
        except RequestException as e:
            _log_debug(f"Request exception: {str(e)}")
            return {'status': 'failure', 'error': str(e)}

    def match_local_content_topk(self, file_path: str, k: int, signal_type: str) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        # signal_type defaults to 'clip_float'
        
        # Check if hash was successful and contains the signal type
        if not isinstance(hasher_resp, dict) or signal_type not in hasher_resp:
            return {
                'status': 'failure',
                'error': f'Failed to get {signal_type} hash',
                'response': str(hasher_resp)
            }
        
        signal = hasher_resp[signal_type]
        data = {
            'signal_type': signal_type,
            'signal': signal,
            'k': k
        }

        try:
            response = requests.post(f"{match_url_topk}", json=data)
            if response.ok:
                result = response.json()
                return {
                    'status': 'success',
                    'matches': result.get("matches", []),
                    'signal_type': signal_type,
                    'signal': signal
                }
            else:
                _log_debug(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }

        except RequestException as e:
            _log_debug(f"Request exception: {str(e)}")
            return {'status': 'failure', 'error': str(e)}

    def match_local_content_threshold(self, file_path: str, threshold: int | float, signal_type: str) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        # signal_type defaults to 'clip_float' (float thresholds)
        
        # Check if hash was successful and contains the signal type
        if not isinstance(hasher_resp, dict) or signal_type not in hasher_resp:
            return {
                'status': 'failure',
                'error': f'Failed to get {signal_type} hash',
                'response': str(hasher_resp)
            }
        
        signal = hasher_resp[signal_type]
        data = {
            'signal_type': signal_type,
            'signal': signal,
            'threshold': threshold
        }

        try:
            response = requests.post(f"{match_url_threshold}", json=data)
            if response.ok:
                result = response.json()
                return {
                    'status': 'success',
                    'matches': result.get("matches", []),
                    'signal_type': signal_type,
                    'signal': signal
                }
            else:
                _log_debug(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }

        except RequestException as e:
            _log_debug(f"Request exception: {str(e)}")
            return {'status': 'failure', 'error': str(e)}

    def get_index_size(self, SIGNAL_TYPE:  str) -> int:
        resp = requests.get(f"{hma_app_url}/m/index/status", params={"signal_type": SIGNAL_TYPE})
        index_size = resp.json().get(SIGNAL_TYPE, {}).get("size", 0)
        return index_size

    def setup_bank(self, bank_name):
        """Ensure the bank exists, create if not."""
        if not self.bank_exists(bank_name):
            if not self.create_bank(bank_name):
                _log_error(f"Failed to create bank {bank_name}. Exiting.")
                return False
        return True

    def delete_bank(self, bank_name):
        """Delete a bank from HMA."""
        try:
            response = requests.delete(f"{hma_app_url}/c/bank/{bank_name}")
            if response.ok:
                _log_info(f"Deleted bank {bank_name}")
                return True
            else:
                _log_warning(f"Failed to delete bank {bank_name}: {response.status_code} - {response.text}")
                return False
        except RequestException as e:
            _log_error(f"Request exception while deleting bank: {str(e)}")
            return False

    def upload_files_to_bank(self, files_to_send, bank_name):
        """Upload files to the specified bank. Returns mapping of content_id -> image_path."""
        content_id_to_image = {}
        missing_content_ids = []
        for file_path in files_to_send:
            result = self.add_file_to_hma_bank(file_path, bank_name)
            _log_debug(result['response'])
            # Store content_id -> image_path mapping if content_id was captured
            if result.get('status') == 'success':
                if result.get('content_id'):
                    content_id_to_image[result['content_id']] = str(file_path)
                else:
                    missing_content_ids.append(str(file_path))
                    _log_warning(f"No content_id found in upload response for {file_path}. Response: {result.get('response', '')[:200]}")
        
        if missing_content_ids:
            _log_warning(f"Failed to capture content_id for {len(missing_content_ids)}/{len(files_to_send)} uploads. MAP calculation may be inaccurate.")
        else:
            _log_info(f"Successfully captured content_id for all {len(files_to_send)} uploads.")
        return content_id_to_image

    def wait_for_index_update(self, expected_size=None, signal_type="clip_float", max_wait=60):
        """Wait until index contains new signal or until timeout."""
        _log_info("Waiting for index to update...")
        for _ in range(max_wait // 5):
            size = self.get_index_size(signal_type)
            _log_debug(f"Index size: {size}")
            if expected_size is None or size >= expected_size:
                _log_info("Index likely updated.")
                return
            time.sleep(5)
        _log_warning("Timed out waiting for index update.")


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

    def compare_hashes(self, hash1, hash2, signal_type: str) -> dict:
        url = f"{hma_app_url}/m/compare"
        headers = {"Content-Type": "application/json"}
        data = {
            signal_type: [hash1, hash2]
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.ok:
                return {
                    "status": "success",
                    "result": response.json().get(signal_type)
                }
            else:
                return {
                    "status": "failure",
                    "error": response.text,
                    "code": response.status_code
                }
        except RequestException as e:
            return {"status": "failure", "error": str(e)}

    def create_fresh_database(self):
        """
        Clear all data from the database to ensure a fresh start.
        Clears all tables and PostgreSQL large objects (where HMA stores indexes).
        """
        if not PSYCOPG2_AVAILABLE:
            _log_warning("psycopg2 not available, cannot clear database")
            return False

        db_host = os.getenv("POSTGRES_HOST", "hma-postgresql")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "media_match")

        try:
            _log_info("Clearing all data from database...")
            
            def clear_all_data():
                """Helper to clear all data and large objects"""
                conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    password=db_password,
                    dbname=db_name,
                    connect_timeout=10
                )
                conn.autocommit = True
                cur = conn.cursor()
                
                # Delete all exchange data, banks, and related content
                # Order matters due to foreign key constraints
                cur.execute("DELETE FROM bank_content;")
                bank_content_count = cur.rowcount
                cur.execute("DELETE FROM content_signal;")
                content_signal_count = cur.rowcount
                cur.execute("DELETE FROM signal_index;")
                signal_index_count = cur.rowcount
                cur.execute("DELETE FROM exchange_data;")
                exchange_data_count = cur.rowcount
                cur.execute("DELETE FROM bank;")
                bank_count = cur.rowcount
                cur.execute("DELETE FROM exchange_fetch_status;")
                exchange_fetch_status_count = cur.rowcount
                
                # Clear PostgreSQL large objects (where HMA stores indexes)
                cur.execute("SELECT lo_unlink(oid) FROM pg_largeobject_metadata;")
                large_object_count = cur.rowcount
                
                # Reset all sequences so IDs start from 1
                sequences = [
                    'bank_content_id_seq',
                    'bank_id_seq',
                    'exchange_data_id_seq',
                    'exchange_id_seq',
                    'exchange_api_config_id_seq',
                    'signal_index_id_seq',
                    'signal_type_override_id_seq'
                ]
                for seq in sequences:
                    cur.execute(f"SELECT setval('{seq}', 1, false);")
                
                cur.close()
                conn.close()
                
                return {
                    'bank_content': bank_content_count,
                    'content_signal': content_signal_count,
                    'signal_index': signal_index_count,
                    'exchange_data': exchange_data_count,
                    'bank': bank_count,
                    'exchange_fetch_status': exchange_fetch_status_count,
                    'large_objects': large_object_count
                }
            
            # Clear data multiple times to catch any auto-fetched data
            # HMA has TASK_FETCHER enabled, which auto-fetches data from exchanges
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(2)  # Wait for fetcher to potentially add more data
                deleted_counts = clear_all_data()
                if attempt == 0:
                    _log_info(f"Cleared: {deleted_counts}")
                elif sum(deleted_counts.values()) > 0:
                    _log_info(f"Additional data cleared: {deleted_counts}")
                else:
                    break  # No more data to clear
            
            _log_info("✓ Database cleared successfully")
            return True

        except Exception as e:
            _log_error(f"Failed to clear database: {e}")
            import traceback
            _log_error(traceback.format_exc())
            return False

    def cleanup_test_environment(self):
        """
        Clean up test environment by clearing all database data.
        This ensures complete isolation - no leftover indexes or data from previous runs.
        Verifies the signal type index is empty.
        """
        _log_info("Cleaning up test environment...")
        
        if self.create_fresh_database():
            # Wait for HMA to process the changes
            time.sleep(2)
            
            # Verify index is empty
            index_size = self.get_index_size(SIGNAL_TYPE)
            if index_size == 0:
                _log_info(f"✓ {SIGNAL_TYPE} index is empty")
            else:
                _log_warning(f"{SIGNAL_TYPE} index size is {index_size}, expected 0")
            return True
        else:
            _log_error("Failed to clear database")
            return False

def calculate_metrics(results_dir):
    """Calculate MAP (from retrieval), classification PR (from pairwise), and distance plots."""
    # Import here to avoid circular dependency
    from metrics.map import compute_map_from_retrieval_csv
    from metrics.precision_recall import compute_precision_recall_from_pairwise
    from metrics.distance_distribution import compute_distance_distribution
    from metrics.common import validate_series_metadata_exists
    
    labels_path = Path(os.getenv(LABELS_PATH_ENV, "resources/labels/images_series_labels.json"))
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
        from tests.test_utils import load_content_id_mapping
        content_id_to_image = load_content_id_mapping(results_dir, SIGNAL_TYPE)
        
        if content_id_to_image:
            _log_info(f"Loaded content_id mapping ({len(content_id_to_image)} entries)")
        else:
            _log_warning(f"Content ID mapping file not found or empty. MAP calculation may be inaccurate.")
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
            print(f"✓")
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
        print(f"✓")
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
        print(f"✓")
        _log_info(f"Saved plot: {dist_output_plot}")
        print(f"    → {dist_output_plot.name}")
    except Exception as e:
        print(f"✗ {e}")
        _log_error(f"Failed to generate distance distribution for {SIGNAL_TYPE}: {e}")

def run_all_tests():
    start_time = time.time()
    setup_logging("test")
    global log_file  # Ensure we can access the log_file variable
    _log_info("[STARTUP] Creating fresh database for test run...")
    print("Running tests...")
    
    # Import here to avoid circular dependency
    from metrics.common import validate_series_metadata_exists
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
    from tests.test_utils import save_content_id_mapping
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
    
    print(f"✓ All tests completed. Logs: {log_file}")
    
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
        print(f"Running smoke test...")
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
        from tests.test_utils import save_content_id_mapping
        content_id_map_file = save_content_id_mapping(content_id_to_image, SIGNAL_TYPE)
        _log_info(f"Saved content_id mapping to {content_id_map_file} ({len(content_id_to_image)} entries)")
        
        _log_info(f"Testing signal_type={SIGNAL_TYPE}")
        index_size_before = evaluator.get_index_size(SIGNAL_TYPE)
        expected_size = index_size_before + len(files_to_send)
        evaluator.wait_for_index_update(expected_size, SIGNAL_TYPE)
        evaluator.match_uploaded_files(files_to_send, SIGNAL_TYPE)
        
        print(f"✓ Smoke test completed. Logs: {log_file}")
    else:
        run_all_tests()

if __name__ == '__main__':
    main()
