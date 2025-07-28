import os
import json
from test_utils import get_image_files, write_results, process_matches_threshold, create_result, match_image
from evaluate import Evaluator

# Test parameter (can be overridden by environment variable)
DEFAULT_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", 0.2))

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    threshold = DEFAULT_THRESHOLD

    for img in image_files:
        try:
            match_result = match_image(evaluator, img)
            
            # Extract clip hash from match result
            clip_hash = match_result.get('signal', '') if match_result.get('status') == 'success' else ''
            
            if match_result.get('status') == 'success' and 'matches' in match_result:
                matches = match_result['matches']
                processed_matches = process_matches_threshold(matches, threshold)
                
                # Get clip hashes for each match
                for match in processed_matches:
                    if match.get('content_id'):
                        content_id = match['content_id']
                        content_result = evaluator.get_signal_from_contentid(content_id, "TEST_BANK_DATA")
                        if content_result.get('status') == 'success':
                            match['clip_hash'] = content_result.get('data', {}).get('signals', {}).get('clip', '')
                        else:
                            match['clip_hash'] = ''
                
                result = create_result(img, clip_hash, processed_matches, threshold=threshold)
            else:
                result = create_result(
                    img, clip_hash, [], 
                    threshold=threshold, 
                    error=match_result.get('error', 'No matches found')
                )
                
        except Exception as e:
            print(f"[WARN] Failed to process {img}: {e}")
            result = create_result(
                img, '', [], 
                threshold=threshold, 
                error=str(e)
            )
            
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, f'threshold_results_{threshold}.json')
    print(f"\n[INFO] Threshold test complete. Results saved to threshold_results_{threshold}.json")

if __name__ == "__main__":
    main() 