import os
import json

from test_utils import get_image_files, write_results
from evaluate import Evaluator

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "threshold_test_results.json")
SIGNAL_TYPE = "clip"
THRESHOLD_MAX = int(os.getenv("THRESHOLD_MAX", 100))
THRESHOLD_STEP = int(os.getenv("THRESHOLD_STEP", 20))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    thresholds = range(0, THRESHOLD_MAX, THRESHOLD_STEP)
    print(f"[INFO] Found {len(image_files)} images. Starting threshold match test with thresholds={list(thresholds)}...")

    results = []
    for threshold in thresholds:
        for img in image_files:
            print(f"\n[THRESHOLD] Matching {img} with threshold={threshold}")
            
            match_resp = evaluator.match_local_content_threshold(img, threshold)

            if match_resp.get("status") == "success":
                result = {
                    "image": img,
                    "threshold": threshold,
                    "matches": match_resp.get("matches", [])
                }
            else:
                print(f"[WARN] Threshold match API failed for {img}: {match_resp}")
                result = {
                    "image": img,
                    "threshold": threshold,
                    "error": match_resp.get("error"),
                    "response": match_resp.get("response")
                }
            
            results.append(result)
            print(json.dumps(result, indent=2))

    write_results(results, OUTPUT_FILE)
    print(f"\n[INFO] Threshold match test complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
