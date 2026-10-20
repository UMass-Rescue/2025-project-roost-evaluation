import requests
import os

hma_host = os.getenv("HMA_HOST", "host.docker.internal")
hma_port = os.getenv("HMA_PORT", "5005")
HMA_API_URL = f"http://{hma_host}:{hma_port}"
BANKS_ENDPOINT = f"{HMA_API_URL}/c/banks"
BANK_DELETE_ENDPOINT = f"{HMA_API_URL}/c/bank"

# Banks to preserve
PROTECTED_BANKS = {"TEST_BANK_DATA"}

def list_banks():
    """List all bank names."""
    try:
        resp = requests.get(BANKS_ENDPOINT)
        if resp.ok:
            banks = resp.json()
            return [bank["name"] for bank in banks]
        else:
            print(f"[ERROR] Failed to list banks: {resp.status_code} - {resp.text}")
            return []
    except requests.RequestException as e:
        print(f"[ERROR] Exception while listing banks: {e}")
        return []

def delete_bank(bank_name):
    """Delete a specific bank by name."""
    try:
        resp = requests.delete(f"{BANK_DELETE_ENDPOINT}/{bank_name}")
        if resp.ok:
            print(f"[INFO] Deleted bank: {bank_name}")
            return True
        else:
            print(f"[WARN] Failed to delete {bank_name}: {resp.status_code} - {resp.text}")
            return False
    except requests.RequestException as e:
        print(f"[ERROR] Exception while deleting bank {bank_name}: {e}")
        return False

def main():
    banks = list_banks()
    print(f"[INFO] Found {len(banks)} banks")

    to_delete = [b for b in banks if b not in PROTECTED_BANKS]
    print(f"[INFO] Will delete {len(to_delete)} banks (excluding protected)")

    for bank in to_delete:
        delete_bank(bank)

if __name__ == "__main__":
    main()
