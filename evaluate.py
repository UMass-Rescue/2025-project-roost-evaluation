import os
import random
import requests
from pathlib import Path

image_input_dir = Path("./resources/images")
hma_app_url = "http://hma-clip-demo-app-1:5000"
curate_bank_url = hma_app_url + "/c/banks"
hash_url = hma_app_url +  "/h/hash"  # Adjust as needed

class Evaluator:
    def bank_exists(self,bank_name: str) -> bool:
        """Check if a bank exists by making API call."""
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


    def send_file_to_hma_bank(self,file_path: str, bank_name:str):
        if self.bank_exists(bank_name) is False:
            self.create_bank(bank_name)
        filename = os.path.basename(file_path)
        print(f"Sending {filename} to HMA bank and store hash...")

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

def main():
    # Collect and shuffle input file list
    evaluator = Evaluator()
    # bank names should be upper case with underscores
    bank_name = "LABELLED_FACES_BANK"
    files_to_send = [str(file) for file in image_input_dir.iterdir() if file.is_file()]
    random.shuffle(files_to_send)

    for file_path in files_to_send:
        result = evaluator.send_file_to_hma_bank(file_path, bank_name)
        print(result['response'])

if __name__ == '__main__':
    main()
