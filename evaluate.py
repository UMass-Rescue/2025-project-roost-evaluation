import os
import requests
from requests import RequestException
import json
import time
from pathlib import Path
import logging
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
match_url = hma_app_url + "/m/lookup"
match_url_topk = hma_app_url + "/m/lookup_topk"
match_url_threshold = hma_app_url + "/m/lookup_threshold"

# Setup logging
logger = None
log_file = None

def setup_logging(test_run_type="test"):
    """Setup file logging and minimal terminal output"""
    global logger, log_file
    
    # Create logs directory
    logs_dir = Path("test_run_logs")
    logs_dir.mkdir(exist_ok=True)
    
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
        """Add a file to the HMA bank and store its hash."""
        try:
            filename = os.path.basename(file_path)
            _log_debug(f"Adding {filename} to HMA bank and storing hash...")

            with open(file_path, 'rb') as f:
                files = {'photo': (filename, f)}
                response = requests.post(f"{hma_app_url}/c/bank/{bank_name}/content", files=files)
                if response.ok:
                    _log_debug(f"Successfully added {filename} to bank {bank_name}")
                    return {'status': 'success', 'response': response.text}
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

    def match_local_content(self, file_path: str) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        signal_type = 'clip_float'
        
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

    def match_local_content_topk(self, file_path: str, k: int) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        signal_type = 'clip_float'
        
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

    def match_local_content_threshold(self, file_path: str, threshold: float) -> dict:
        hasher_resp = self.hash_local_content(file_path)
        signal_type = 'clip_float'
        
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
        """Upload files to the specified bank."""
        for file_path in files_to_send:
            result = self.add_file_to_hma_bank(file_path, bank_name)
            _log_debug(result['response'])

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


    def match_uploaded_files(self, files_to_send):
        """Match each uploaded file and print the response."""
        _log_info("Sleeping 35 seconds to allow in-memory index cache to refresh...")
        time.sleep(35)
        for match_file_path in files_to_send:
            _log_debug(match_file_path)
            match_resp = self.match_local_content(match_file_path)
            _log_debug(json.dumps(match_resp, indent=2))

    def compare_hashes(self, hash1, hash2, signal_type="clip_float") -> dict:
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

    def cleanup_test_environment(self, signal_type="clip"):
        """
        Clean up test environment by clearing all database data.
        This ensures complete isolation - no leftover indexes or data from previous runs.
        """
        _log_info("Cleaning up test environment...")
        
        if self.create_fresh_database():
            # Wait for HMA to process the changes
            time.sleep(2)
            
            # Verify index is empty
            index_size = self.get_index_size(signal_type)
            if index_size == 0:
                _log_info(f"✓ {signal_type} index is empty")
            else:
                _log_warning(f"Index size is {index_size}, expected 0")
            return True
        else:
            _log_error("Failed to clear database")
            return False

def run_all_tests():
    setup_logging("test")
    global log_file  # Ensure we can access the log_file variable
    _log_info("[STARTUP] Creating fresh database for test run...")
    print("Running tests...")
    
    # Generate timestamp for this test run and pass to subprocesses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["TEST_RUN_TIMESTAMP"] = timestamp
    _log_info(f"Test run timestamp: {timestamp}")
    
    evaluator = Evaluator()
    evaluator.cleanup_test_environment(signal_type="clip")
    
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_files = [f for f in os.listdir(test_dir) if f.endswith("_test.py")]
    _log_info(f"Found {len(test_files)} test files: {test_files}")
    print(f"Found {len(test_files)} test files")
    # Flush to ensure output is visible
    import sys
    sys.stdout.flush()
    
    for i, fname in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {fname}")
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

def main():
    eval_mode = os.environ.get("EVAL_MODE", "smoke")
    test_run_type = "smoke" if eval_mode == "smoke" else "test"
    setup_logging(test_run_type)
    
    if eval_mode == "smoke":
        _log_info("[STARTUP] Creating fresh database for smoke test...")
        print("Running smoke test...")
        evaluator = Evaluator()
        evaluator.cleanup_test_environment(signal_type="clip")
        
        BANK_NAME = os.getenv("BANK_NAME", "TEST_BANK_DATA")
        if not evaluator.setup_bank(BANK_NAME):
            return

        files_to_send = [str(file) for file in image_input_dir.iterdir() if file.is_file()]
        index_size_before = evaluator.get_index_size("clip_float")
        _log_info(f"Starting index size: {index_size_before}")
        print(f"Uploading {len(files_to_send)} files...")
        evaluator.upload_files_to_bank(files_to_send, BANK_NAME)
        expected_size = index_size_before + len(files_to_send)
        evaluator.wait_for_index_update(expected_size, "clip_float")
        evaluator.match_uploaded_files(files_to_send)
        print(f"✓ Smoke test completed. Logs: {log_file}")
    else:
        run_all_tests()

if __name__ == '__main__':
    main()
