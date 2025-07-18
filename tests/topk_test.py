import os
import json
from test_utils import get_image_files, hash_image, write_results
from evaluate import Evaluator

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    k = int(os.environ.get("EVAL_TOPK", 5))

    print(f"Found {len(image_files)} images. Running top-{k} tests...")

    for img in image_files:
        hash_result = hash_image(evaluator, img)
        try:
            match_result = evaluator.match_local_content_with_topk(img, k)
        except Exception as e:
            match_result = {'status': 'failure', 'error': str(e)}
        result = {
            'image': os.path.basename(img),
            'hash': hash_result,
            'match_result': match_result
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, f'topk_results_{k}.json')
    print(f"Top-{k} testing complete. Results written to topk_results_{k}.json.")

if __name__ == "__main__":
    main() 