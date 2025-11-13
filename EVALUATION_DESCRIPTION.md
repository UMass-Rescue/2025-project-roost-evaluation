## Evaluation tests overview

### Top‑k Match Test (`tests/topk_test.py`)
- **Goal**: For each image, fetch the top‑K most similar matches from the local content bank.
- **How it works**: Iterates K from 1 to `MAX_K` (inclusive) and calls `Evaluator.match_local_content_topk(image_path, k)`.
- **Inputs**:
  - Images discovered by `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir` (commonly `resources/images/`).
  - Env vars: `MAX_K` (default `5`), `OUTPUT_FILE` (default `topk_test_results.json`).
- **Output**: JSON list of results. On success: `{ "image", "k", "matches": [...] }`. On failure: `{ "image", "k", "error", "response" }`.
- **Progress/Logging**: Shows a `tqdm` progress bar in the terminal and logs via the shared logger (`_log_info/_log_debug/_log_warning`).

### Threshold Match Test (`tests/threshold_test.py`)
- **Goal**: Test how match results change as a score threshold varies.
- **How it works**: Iterates thresholds in `range(0, THRESHOLD_MAX, THRESHOLD_STEP)` and calls `Evaluator.match_local_content_threshold(image_path, threshold)`.
- **Inputs**:
  - Images from `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir`.
  - Env vars: `THRESHOLD_MAX` (default `100`), `THRESHOLD_STEP` (default `20`), `OUTPUT_FILE` (default `threshold_test_results.json`).
- **Output**: JSON list of results. On success: `{ "image", "threshold", "matches": [...] }`. On failure: `{ "image", "threshold", "error", "response" }`.
- **Progress/Logging**: Uses `tqdm` for terminal progress and the shared logger for detailed logs.

### Pairwise CLIP Comparison (`tests/pairwise_test.py`)
- **Goal**: Compare CLIP hashes for every unique pair of images to measure similarity.
- **How it works**:
  - First caches each image’s CLIP hash via `tests.test_utils.hash_image(evaluator, image)` (expects `resp["clip"]`).
  - Compares all unique image pairs with `Evaluator.compare_hashes(clip1, clip2, signal_type="clip")`.
  - Parses the compare result into a boolean `matched` and numeric `distance`.
- **Inputs**:
  - Images from `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir`.
  - Fixed `SIGNAL_TYPE = "clip"`. Env var: `OUTPUT_FILE` (default `pairwise_clip_compare.json`).
- **Output**: JSON list of pairwise results: `{ "image1", "image2", "matched", "distance" }`.
- **Progress/Logging**: `tqdm` shows progress; detailed messages go to the shared logger.

Notes:
- All three tests discover images automatically and write results to a JSON file (override with `OUTPUT_FILE`).
- Set `IMAGE_INPUT_DIR` to point the tests at a different directory of images.

Anonymization (optional):
- Set `ANONYMIZE_IMAGE_PATHS=1` to replace image paths in results with simple index IDs ("1", "2", ...).
- Optionally set `ANON_ID_MAP_FILEPATH` to persist IDs across runs: the file is loaded if it exists and updated after tests. It contains `{ "<full_path>": "<id>" }`. **DON'T SHARE THIS FILE** if you don't want to leak the original image paths.
- If `ANON_ID_MAP_FILEPATH` is not set, anonymized IDs are assigned in-memory for the current run only (not stable across runs).

