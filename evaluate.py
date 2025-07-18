import os
import random
import requests
from requests import RequestException
import json
import time
from pathlib import Path
import subprocess

image_input_dir = Path("./resources/images")
hma_app_url = "http://host.docker.internal:5005"
hash_url = hma_app_url +  "/h/hash"  
match_url = hma_app_url + "/m/lookup"

# Test parameters (can be overridden by environment variables)
DEFAULT_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", 0.2))
DEFAULT_TOPK = int(os.environ.get("EVAL_TOPK", 5))

class Evaluator:
    def bank_exists(self,bank_name: str) -> bool:
        """Check if a bank exists by making API call to HMA."""
        try:
            response = requests.get(f"{hma_app_url}/c/bank/{bank_name}")
            if response.ok:
                print(f"Bank {bank_name} exists")
                return True
            elif response.status_code == 404:
                print(f"Bank {bank_name} does not exist")
                return False
            else:
                print(f"Failed to check bank existence: {response.status_code} - {response.text}")
                return False
                
        except RequestException as e:
            print(f"Request exception while checking bank existence: {str(e)}")
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
                print(f"Successfully created bank {bank_name}")
                return True
            else:
                print(f"Failed to create bank: {create_response.status_code} - {create_response.text}")
                return False
                
        except RequestException as e:
            print(f"Request exception while creating bank: {str(e)}")
            return False


    def add_file_to_hma_bank(self, file_path: str, bank_name: str):
        """Add a file to the HMA bank and store its hash."""
        try:
            filename = os.path.basename(file_path)
            print(f"Adding {filename} to HMA bank and storing hash...")

            with open(file_path, 'rb') as f:
                files = {'photo': (filename, f)}
                response = requests.post(f"{hma_app_url}/c/bank/{bank_name}/content", files=files)
                if response.ok:
                    print(f"Successfully added {filename} to bank {bank_name}")
                    return {'status': 'success', 'response': response.text}
                else:
                    print(f"Failed to add {filename} to bank {bank_name}: {response.status_code} - {response.text}")
                    return {'status': 'failure', 'response': f"Failed for {filename}: {response.status_code} - {response.text}"}
        except RequestException as e:
            print(f"Request exception while adding file to bank: {str(e)}")
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
        signal_type = 'clip'
        signal = hasher_resp[signal_type]
        params = {
            'signal_type': signal_type,
            'signal': signal
        }
   
        try:
            response = requests.get(f"{match_url}", params=params)
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
                print(f"API request failed: {response.status_code} - {response.text}")  # Debug log
                return {
                    'status': 'failure',
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }
            
        except RequestException as e:
            print(f"Request exception: {str(e)}")  # Debug log
            return {'status': 'failure', 'error': str(e)}

    def get_index_size(self, SIGNAL_TYPE:  str) -> int:
        resp = requests.get(f"{hma_app_url}/m/index/status", params={"signal_type": SIGNAL_TYPE})
        index_size = resp.json().get(SIGNAL_TYPE, {}).get("size", 0)
        return index_size

    def setup_bank(self, bank_name):
        """Ensure the bank exists, create if not."""
        if not self.bank_exists(bank_name):
            if not self.create_bank(bank_name):
                print(f"Failed to create bank {bank_name}. Exiting.")
                return False
        return True

    def upload_files_to_bank(self, files_to_send, bank_name):
        """Upload files to the specified bank."""
        for file_path in files_to_send:
            result = self.add_file_to_hma_bank(file_path, bank_name)
            print(result['response'])

    def wait_for_index_update(self, expected_size, signal_type="clip"):
        """Wait until the index size reaches the expected value."""
        while True:
            current_size = self.get_index_size(signal_type)
            print(f"Current index size: {current_size}")
            if current_size >= expected_size:
                print("Index is up-to-date!")
                break
            time.sleep(5)

    def match_uploaded_files(self, files_to_send):
        """Match each uploaded file and print the response."""
        print("[INFO]: Sleeping 35 seconds to allow in-memory index cache to refresh...")
        time.sleep(35)
        for match_file_path in files_to_send:
            print(match_file_path)
            match_resp = self.match_local_content(match_file_path)
            print(json.dumps(match_resp, indent=2))

def run_all_tests():
    evaluator = Evaluator()
    image_files = [str(file) for file in image_input_dir.iterdir() if file.is_file()]
    threshold = DEFAULT_THRESHOLD
    k = DEFAULT_TOPK
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    for fname in os.listdir(test_dir):
        if fname.endswith("_test.py"):
            print(f"Running {fname} ...")
            subprocess.run(["python", os.path.join(test_dir, fname)], check=True)

def main():
    eval_mode = os.environ.get("EVAL_MODE", "smoke")
    if eval_mode == "smoke":
        evaluator = Evaluator()
        BANK_NAME = "TEST_BANK_DATA"
        if not evaluator.setup_bank(BANK_NAME):
            return

        files_to_send = [str(file) for file in image_input_dir.iterdir() if file.is_file()]
        index_size_before = evaluator.get_index_size("clip")
        evaluator.upload_files_to_bank(files_to_send, BANK_NAME)
        expected_size = index_size_before + len(files_to_send)
        evaluator.wait_for_index_update(expected_size, "clip")
        evaluator.match_uploaded_files(files_to_send)
    else:
        run_all_tests()


if __name__ == '__main__':
    main()
