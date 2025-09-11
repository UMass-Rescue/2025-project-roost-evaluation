import os
import json

from test_utils import get_image_files, write_results
from evaluate import Evaluator

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "threshold_test_results.json")
SIGNAL_TYPE = "clip"
THRESHOLD = int(os.getenv("THRESHOLD", 30))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    print(f"[INFO] Found {len(image_files)} images. Starting threshold match test with threshold={THRESHOLD}...")

    results = []
    for img in image_files:
        print(f"\n[THRESHOLD] Matching {img} with threshold={THRESHOLD}")
        
        match_resp = evaluator.match_local_content_threshold(img, THRESHOLD)

        if match_resp.get("status") == "success":
            result = {
                "image": img,
                "threshold": THRESHOLD,
                "matches": match_resp.get("matches", [])
            }
        else:
            print(f"[WARN] Threshold match API failed for {img}: {match_resp}")
            result = {
                "image": img,
                "threshold": THRESHOLD,
                "error": match_resp.get("error"),
                "response": match_resp.get("response")
            }
        
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, OUTPUT_FILE)
    print(f"\n[INFO] Threshold match test complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
