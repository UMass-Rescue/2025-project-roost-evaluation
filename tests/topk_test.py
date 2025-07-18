import os
import json
from test_utils import get_image_files, hash_image, write_results
from evaluate import Evaluator

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    k = int(os.environ.get("EVAL_TOPK", 5))

    for img in image_files:
        hash_result = hash_image(evaluator, img)
        try:
            match_result = evaluator.match_local_content(img)
            # Apply top-k filtering
            if match_result['status'] == 'success' and 'matches' in match_result:
                matches = match_result['matches']
                # Check if matches is a dictionary (bank_name -> list of matches)
                if isinstance(matches, dict):
                    all_matches = []
                    for bank_name, bank_matches in matches.items():
                        if isinstance(bank_matches, list):
                            for match in bank_matches:
                                if isinstance(match, dict):
                                    # Convert distance string to float for sorting
                                    distance_str = match.get('distance', 'inf')
                                    try:
                                        distance = float(distance_str)
                                        # Add bank_name to match for reference
                                        match_with_bank = match.copy()
                                        match_with_bank['bank_name'] = bank_name
                                        all_matches.append(match_with_bank)
                                    except (ValueError, TypeError):
                                        pass
                    
                    # Sort by distance and take top-k
                    sorted_matches = sorted(all_matches, key=lambda m: m.get('distance', float('inf')))
                    # Keep only essential matching data, remove redundant signal info
                    match_result = {
                        'status': 'success',
                        'matches': sorted_matches[:k],
                        'top_k': k
                    }
                else:
                    match_result = {
                        'status': 'success',
                        'matches': []
                    }
                
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