import os
import json
from test_utils import get_image_files, hash_image, write_results
from evaluate import Evaluator

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    threshold = float(os.environ.get("EVAL_THRESHOLD", 0.2))

    print(f"Found {len(image_files)} images. Running threshold test with threshold={threshold}...")

    for img in image_files:
        hash_result = hash_image(evaluator, img)
        try:
            match_result = evaluator.match_local_content_with_threshold(img, threshold)
        except Exception as e:
            match_result = {'status': 'failure', 'error': str(e)}
        result = {
            'image': os.path.basename(img),
            'hash': hash_result,
            'match_result': match_result
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, f'threshold_results_{threshold}.json')
    print(f"Threshold testing complete. Results written to threshold_results_{threshold}.json.")

if __name__ == "__main__":
    main() 