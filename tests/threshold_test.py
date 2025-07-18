import os
import json
from test_utils import get_image_files, hash_image, write_results
from evaluate import Evaluator

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    results = []
    threshold = float(os.environ.get("EVAL_THRESHOLD", 0.2))

    for img in image_files:
        hash_result = hash_image(evaluator, img)
        try:
            match_result = evaluator.match_local_content(img)
            # Apply threshold filtering
            if match_result['status'] == 'success' and 'matches' in match_result:
                matches = match_result['matches']
                # Check if matches is a dictionary (bank_name -> list of matches)
                if isinstance(matches, dict):
                    filtered_matches = []
                    for bank_name, bank_matches in matches.items():
                        if isinstance(bank_matches, list):
                            for match in bank_matches:
                                if isinstance(match, dict):
                                    # Convert distance string to float for comparison
                                    distance_str = match.get('distance', 'inf')
                                    try:
                                        distance = float(distance_str)
                                        if distance <= threshold:
                                            # Add bank_name to match for reference
                                            match_with_bank = match.copy()
                                            match_with_bank['bank_name'] = bank_name
                                            filtered_matches.append(match_with_bank)
                                    except (ValueError, TypeError):
                                        pass
                    
                    # Keep only essential matching data, remove redundant signal info
                    match_result = {
                        'status': 'success',
                        'matches': filtered_matches,
                        'threshold': threshold
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

    write_results(results, f'threshold_results_{threshold}.json')
    print(f"Threshold testing complete. Results written to threshold_results_{threshold}.json.")

if __name__ == "__main__":
    main() 