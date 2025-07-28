import os
import json
from test_utils import get_image_files, write_results, create_result, match_image, process_image, extract_matches
from evaluate import Evaluator

# Test parameter (can be overridden by environment variable)
DEFAULT_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", 0.2))

def process_matches_threshold(matches, threshold):
    """Process matches and apply threshold filtering."""
    processed_matches = extract_matches(matches)
    
    # Filter by threshold and sort by distance
    filtered_matches = [match for match in processed_matches if match.get('distance', float('inf')) <= threshold]
    return sorted(filtered_matches, key=lambda m: m.get('distance', float('inf')))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    threshold = DEFAULT_THRESHOLD

    for img in image_files:
        result = process_image(evaluator, img, process_matches_threshold, threshold=threshold)
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, f'threshold_results_{threshold}.json')
    print(f"\n[INFO] Threshold test complete. Results saved to threshold_results_{threshold}.json")

if __name__ == "__main__":
    main()