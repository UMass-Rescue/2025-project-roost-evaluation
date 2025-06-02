import os
import random
import requests
from requests import RequestException
import json
from pathlib import Path

image_input_dir = Path("./resources/images")
hma_app_url = "http://hma-clip-demo-app-1:5000"
curate_bank_url = hma_app_url + "/c/banks"
hash_url = hma_app_url +  "/h/hash"  
match_url = hma_app_url + "/m/raw_lookup"

class Evaluator:
    def bank_exists(self,bank_name: str) -> bool:
        """Check if a bank exists by making API call to HMA."""
        try:
            response = requests.get(f"{hma_app_url}/h/bank/{bank_name}")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
        

    def create_bank(self, bank_name: str) :
        response = requests.post(f"{hma_app_url}/c/banks", json={"name": bank_name})
        if response.ok:
            return {'status': 'success', 'response': response.text}
        else:
            return {'status': 'failure', 'response': f"Failed to hma bank: {response.status_code} - {response.text}"}


    def add_file_to_hma_bank(self,file_path: str, bank_name:str):
        if self.bank_exists(bank_name) is False:
            self.create_bank(bank_name)
        filename = os.path.basename(file_path)
        print(f"adding {filename} to HMA bank and store hash...")

        with open(file_path, 'rb') as f:
            files = {'photo': (filename, f)}
            try:
                response = requests.post(f"{hma_app_url}/c/bank/{bank_name}/content", files=files)
                if response.ok:
                    return {'status': 'success', 'response': response.text}
                else:
                    return {'status': 'failure', 'response': f"Failed for {filename}: {response.status_code} - {response.text}"}
            except requests.RequestException as e:
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
            'signal_type': 'pdq',
            'signal': signal,
            'include_distance' : True
        }
        try:
            response = requests.get(match_url, params=params)
            if response.ok:
                result = response.json()
                return result
            
        except RequestException as e:
            return {'status': 'failure', 'response': str(e)}
    
def main():

    evaluator = Evaluator()

    BANK_NAME = "LABELLED_FACES_BANK"

    files_to_send = [str(file) for file in image_input_dir.iterdir() if file.is_file()]

    for file_path in files_to_send:
        result = evaluator.add_file_to_hma_bank(file_path, BANK_NAME)
        print(result['response'])

    match_file_path = files_to_send[0]
    print(match_file_path)

    match_resp = evaluator.match_local_content(match_file_path)
    print(json.dumps(match_resp, indent=2))



if __name__ == '__main__':
    main()
