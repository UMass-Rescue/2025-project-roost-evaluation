import itertools
import json
from test_utils import get_image_files, hash_image, write_results, decode_clip_hex_to_floats
from evaluate import Evaluator

OUTPUT_FILE = "pairwise_clip_compare.json"
SIGNAL_TYPE = "clip"

def main():
    evaluator = Evaluator()
    image_files = get_image_files()
    print(f"[INFO] Found {len(image_files)} images. Starting pairwise CLIP hash comparison...")

    # Cache hashes
    image_hashes = {}
    for img in image_files:
        resp = hash_image(evaluator, img)
        if isinstance(resp, dict) and SIGNAL_TYPE in resp:
            image_hashes[img] = resp[SIGNAL_TYPE]
        else:
            print(f"[WARN] Failed to hash {img}: {resp}")

    results = []
    for img1, img2 in itertools.combinations(image_files, 2):
        print(f"\n[PAIRWISE] Comparing {img1} <--> {img2}")

        clip1 = image_hashes.get(img1)
        clip2 = image_hashes.get(img2)

        if not clip1 or not clip2:
            print(f"[ERROR] Missing hashes for {img1} or {img2}. Skipping.")
            continue

        compare_resp = evaluator.compare_hashes(clip1, clip2, signal_type=SIGNAL_TYPE)

        matched = False
        distance = None
        if compare_resp.get("status") == "success":
            try:
                distance = float(compare_resp["result"].get("distance", None))
                matched = distance is not None
            except (ValueError, TypeError):
                pass
        else:
            print(f"[WARN] Compare API failed: {compare_resp}")

        result = {
            "image1": img1,
            "image2": img2,
            "matched": matched,
            "distance": distance,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    write_results(results, OUTPUT_FILE)
    print(f"\n[INFO] Pairwise CLIP comparison complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
