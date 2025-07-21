import os
import json
from test_utils import get_image_files, write_results, process_matches_topk, create_result, match_image
from evaluate import Evaluator

# Test parameter (can be overridden by environment variable)
DEFAULT_TOPK = int(os.environ.get("EVAL_TOPK", 5))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    k = DEFAULT_TOPK

    for img in image_files:
        try:
            match_result = match_image(evaluator, img)
            
            # Extract clip hash from match result
            clip_hash = match_result.get('signal', '') if match_result.get('status') == 'success' else ''
            
            if match_result['status'] == 'success' and 'matches' in match_result:
                matches = match_result['matches']
                processed_matches = process_matches_topk(matches, k)
                
                result = create_result(img, clip_hash, processed_matches, top_k=k)
            else:
                result = create_result(
                    img, clip_hash, [], 
                    top_k=k, 
                    error=match_result.get('error', 'No matches found')
                )
                
        except Exception as e:
            result = create_result(
                img, '', [], 
                top_k=k, 
                error=str(e)
            )
            
        results.append(result)
        print(json.dumps(result, indent=2))

    # TODO : write results to file, mount docker volume if needed
    write_results(results, f'topk_results_{k}.json')

if __name__ == "__main__":
    main() 