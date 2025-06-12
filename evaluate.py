import os
import random
import requests
from requests import RequestException
import json
from pathlib import Path

image_input_dir = Path("./resources/images")
hma_app_url = "http://host.docker.internal:5000"
hash_url = hma_app_url +  "/h/hash"  
match_url = hma_app_url + "/m/lookup"

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
        signal_type = 'pdq'
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



    
def main():
    evaluator = Evaluator()
    # Bank names should be upper case with underscore
    BANK_NAME = "TEST_BANK_DATA"

    # Check and create the bank once
    if not evaluator.bank_exists(BANK_NAME):
        if not evaluator.create_bank(BANK_NAME):
            print(f"Failed to create bank {BANK_NAME}. Exiting.")
            return

    files_to_send = [str(file) for file in image_input_dir.iterdir() if file.is_file()]

    for file_path in files_to_send:
        result = evaluator.add_file_to_hma_bank(file_path, BANK_NAME)
        print(result['response'])

    for match_file_path in files_to_send:
        print(match_file_path)
        match_resp = evaluator.match_local_content(match_file_path)
        print(json.dumps(match_resp, indent=2))


if __name__ == '__main__':
    main()
