import os
import json
import time
import requests
from requests import RequestException
from hma.logging_utils import _log_debug, _log_info, _log_warning, _log_error

# Try to import psycopg2 for database access (optional)
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

hma_host = os.getenv("HMA_HOST", "host.docker.internal")
hma_port = os.getenv("HMA_PORT", "5005")
hma_app_url = f"http://{hma_host}:{hma_port}"
hash_url = hma_app_url +  "/h/hash"
hash_batch_url = hma_app_url + "/h/hash/batch"
match_url = hma_app_url + "/m/lookup"
match_url_topk = hma_app_url + "/m/lookup_topk"
match_url_threshold = hma_app_url + "/m/lookup_threshold"


class HMAClient:
    """HTTP calls to a running HMA service and test-environment reset, shared by the image and video evaluations."""

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

    def hash_image_bytes(self, images: list[bytes], signal_type: str) -> list[str]:
        """Hash in-memory images (JPEG or PNG bytes) in one /h/hash/batch request.

        Returns one signal string per image, in input order. Raises RuntimeError on any failure,
        because a caller that needs every hash cannot carry on without one.
        """
        files = [("photo", (f"{i}.jpg", image)) for i, image in enumerate(images)]
        try:
            response = requests.post(hash_batch_url, files=files, params={"signal_type": signal_type})
        except RequestException as e:
            raise RuntimeError(f"Hashing request failed: {e}") from e
        if not response.ok:
            raise RuntimeError(f"Hashing failed: {response.status_code} - {response.text[:200]}")
        results = response.json()
        if not isinstance(results, list) or len(results) != len(images):
            raise RuntimeError(f"Hashing returned an unexpected response for {len(images)} images")
        signals = []
        for result in results:
            if not isinstance(result, dict) or signal_type not in result:
                raise RuntimeError(f"Hashing returned no {signal_type} signal: {str(result)[:200]}")
            signals.append(result[signal_type])
        return signals

    def add_signal_to_bank(self, bank_name: str, signal_type: str, signal: str) -> str | None:
        """Add an already computed signal to a bank. Returns the new bank content id, or None on failure."""
        try:
            response = requests.post(f"{hma_app_url}/c/bank/{bank_name}/signal", json={signal_type: signal})
        except RequestException as e:
            _log_error(f"Request exception while adding a signal to bank {bank_name}: {e}")
            return None
        if not response.ok:
            _log_error(f"Failed to add a signal to bank {bank_name}: {response.status_code} - {response.text}")
            return None
        content_id = response.json().get("id")
        return None if content_id is None else str(content_id)

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

    def cleanup_test_environment(self, signal_type: str | None = None):
        """
        Clean up test environment by clearing all database data.
        This ensures complete isolation - no leftover indexes or data from previous runs.
        Verifies the signal type index is empty.
        """
        signal_type = signal_type or os.getenv("SIGNAL_TYPE", "clip_float")
        _log_info("Cleaning up test environment...")
        
        if self.create_fresh_database():
            # Wait for HMA to process the changes
            time.sleep(2)
            
            # Verify index is empty
            index_size = self.get_index_size(signal_type)
            if index_size == 0:
                _log_info(f"✓ {signal_type} index is empty")
            else:
                _log_warning(f"{signal_type} index size is {index_size}, expected 0")
            return True
        else:
            _log_error("Failed to clear database")
            return False
