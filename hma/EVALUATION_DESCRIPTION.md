## Evaluation tests overview

### Top‑k Match Test (`tests/topk_test.py`)
- **Goal**: For each image, fetch the top‑K most similar matches from the local content bank.
- **How it works**: Iterates K from 1 to `MAX_K` (inclusive) and calls `Evaluator.match_local_content_topk(image_path, k, SIGNAL_TYPE)`, where `SIGNAL_TYPE` is read from the environment.
- **Inputs**:
  - Images discovered by `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir` (commonly `resources/images/`).
  - Env vars:
    - `SIGNAL_TYPE` (default `clip_float` when running the test module directly; when run via `evaluate.py`, the parent sets this per signal type)
    - `MAX_K` (default `5`)
    - `OUTPUT_FILE` (default `topk_test_<signal_type>_results.json`)
- **Output**: JSON list of results. On success: `{ "image", "k", "matches": [...] }`. On failure: `{ "image", "k", "error", "response" }`.
- **Progress/Logging**: Shows a `tqdm` progress bar in the terminal and logs via the shared logger (`_log_info/_log_debug/_log_warning`).

### Threshold Match Test (`tests/threshold_test.py`)
- **Goal**: Test how match results change as a score threshold varies.
- **How it works**: Builds a threshold sweep based on `SIGNAL_TYPE`, then calls `Evaluator.match_local_content_threshold(image_path, threshold, SIGNAL_TYPE)`.
  - For `SIGNAL_TYPE=clip`: thresholds are integers from 0..`THRESHOLD_MAX` (inclusive) stepping by `THRESHOLD_STEP`.
  - For `SIGNAL_TYPE=clip_float`: thresholds are floats from 0.0..`THRESHOLD_MAX` using `numpy.linspace`.
- **Inputs**:
  - Images from `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir`.
  - Env vars:
    - `SIGNAL_TYPE` (default `clip_float` when running the test module directly; when run via `evaluate.py`, the parent sets this per signal type)
    - `THRESHOLD_MAX` (default `100` for `clip`, `1.0` for `clip_float`)
    - `THRESHOLD_STEP` (default `20` for `clip`, `0.2` for `clip_float`)
    - `OUTPUT_FILE` (default `threshold_test_<signal_type>_results.json`)
- **Output**: JSON list of results. On success: `{ "image", "threshold", "matches": [...] }`. On failure: `{ "image", "threshold", "error", "response" }`.
- **Progress/Logging**: Uses `tqdm` for terminal progress and the shared logger for detailed logs.

### Pairwise Signal Comparison (`tests/pairwise_test.py`)
- **Goal**: Compare hashes for every unique pair of images (for the chosen `SIGNAL_TYPE`) to measure similarity.
- **How it works**:
  - First caches each image’s hash via `tests.test_utils.hash_image(evaluator, image)` and stores `resp[SIGNAL_TYPE]`.
  - Compares all unique image pairs with `Evaluator.compare_hashes(hash1, hash2, signal_type=SIGNAL_TYPE)`.
  - Parses the compare result into a boolean `matched` and numeric `distance`.
- **Inputs**:
  - Images from `tests.test_utils.get_image_files()`; the directory is controlled by `IMAGE_INPUT_DIR` (env var). If unset, it falls back to `evaluate.image_input_dir`.
  - Env vars:
    - `SIGNAL_TYPE` (default `clip_float` when running the test module directly; when run via `evaluate.py`, the parent sets this per signal type)
    - `OUTPUT_FILE` (default `pairwise_<signal_type>_compare.json`)
- **Output**: JSON list of pairwise results: `{ "image1", "image2", "matched", "distance" }`.
- **Progress/Logging**: `tqdm` shows progress; detailed messages go to the shared logger.

Notes:
- All three tests discover images automatically and write results under `OUTPUT_DIR/evaluation_results/<timestamp>/` (override filename with `OUTPUT_FILE`).
- Set `IMAGE_INPUT_DIR` to point the tests at a different directory of images.
 - When running via `evaluate.py`, tests are executed once per signal type in `evaluate.SIGNAL_TYPES` (unless `SIGNAL_TYPE` is set to force a single signal type). The parent process also sets `TEST_RUN_TIMESTAMP` so all outputs land in the same timestamped results directory.

Anonymization and outputs:
- Anonymization is ON by default. Set `DEANONYMIZE_IMAGE_PATHS=1` to disable.
- `OUTPUT_DIR` controls all outputs (default `./results`):
  - Results JSONs: `OUTPUT_DIR/evaluation_results/<timestamp>/`
  - Logs: `OUTPUT_DIR/test_run_logs/`
  - Mapping file: `OUTPUT_DIR/file_to_id_map/anon_id_map.json`
- `ANON_ID_MAP_FILEPATH` optionally overrides only the mapping file path. The mapping contains `{ "<full_path>": "<id>" }`. **DON'T SHARE THIS FILE** if you don't want to leak original image paths.

