import os
import json

from test_utils import get_image_files, write_results
from evaluate import Evaluator

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "topk_test_results.json")
SIGNAL_TYPE = "clip"
MAX_K = int(os.getenv("MAX_K", 5))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    k_values = range(1, MAX_K + 1)
    print(f"[INFO] Found {len(image_files)} images. Starting top-k match test with k values={list(k_values)}...")

    results = []
    for k in k_values:
        for img in image_files:
            print(f"\n[TOP-K] Matching {img} with k={k}")
            
            match_resp = evaluator.match_local_content_topk(img, k)

            if match_resp.get("status") == "success":
                result = {
                    "image": img,
                    "k": k,
                    "matches": match_resp.get("matches", [])
                }
            else:
                print(f"[WARN] Top-k match API failed for {img}: {match_resp}")
                result = {
                    "image": img,
                    "k": k,
                    "error": match_resp.get("error"),
                    "response": match_resp.get("response")
                }
            
            results.append(result)
            print(json.dumps(result, indent=2))

    write_results(results, OUTPUT_FILE)
    print(f"\n[INFO] Top-k match test complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
