import sys
import os
import itertools
import json
from test_utils import get_image_files, hash_image, match_image, write_results
from evaluate import Evaluator

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []

    print(f"Found {len(image_files)} images. Running pairwise tests...")

    for img1, img2 in itertools.combinations(image_files, 2):
        hash1 = hash_image(evaluator, img1)
        hash2 = hash_image(evaluator, img2)
        try:
            match_result = match_image(evaluator, img1)
        except Exception as e:
            match_result = {'status': 'failure', 'error': str(e)}
        result = {
            'image1': os.path.basename(img1),
            'image2': os.path.basename(img2),
            'hash1': hash1,
            'hash2': hash2,
            'match_result': match_result
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, 'pairwise_results.json')
    print(f"Pairwise testing complete. Results written to pairwise_results.json.")

if __name__ == "__main__":
    main()
