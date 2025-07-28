import os
import json
from test_utils import get_image_files, write_results, create_result, match_image, process_image, extract_matches
from evaluate import Evaluator

# Test parameter (can be overridden by environment variable)
DEFAULT_TOPK = int(os.environ.get("EVAL_TOPK", 5))

def process_matches_topk(matches, k):
    """Process matches and apply top-k filtering."""
    processed_matches = extract_matches(matches)
    
    # Sort by distance and take top k
    sorted_matches = sorted(processed_matches, key=lambda m: m.get('distance', float('inf')))
    return sorted_matches[:k]

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    k = DEFAULT_TOPK

    for img in image_files:
        result = process_image(evaluator, img, process_matches_topk, k=k)
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, f'topk_results_{k}.json')
    print(f"\n[INFO] Top-k test complete. Results saved to topk_results_{k}.json")

if __name__ == "__main__":
    main()