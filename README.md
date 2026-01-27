# 2025 Project Roost Evaluation

## Overview

This project evaluates content matching capabilities using HMA (Hasher-Matcher-Actioner) from Facebook's ThreatExchange repository. Includes docker-compose setup that builds HMA from a locked commit and an evaluation service for running tests.

## Prerequisites

- Docker and Docker Compose
- Git

## Quick Start

### 1. Create Docker Network

```bash
docker network create shared-hma-network
```

### 2. Start Services

```bash
docker compose up --build -d
```

This builds HMA from ThreatExchange at a configured commit and starts:
- PostgreSQL database for HMA
- Database migrations
- HMA application (port 5005 on host, 5100 internally)
- Evaluation service

To change the ThreatExchange commit: set build-time `THREATEXCHANGE_COMMIT` (in `docker-compose.yaml` or via `THREATEXCHANGE_COMMIT=... docker compose up --build -d`).

### 3. Run Tests

**Smoke test (iterates through all available signals by default):**
```bash
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```
The smoke test cleans the database, creates a bank, uploads all images from `resources/images/`, and tests matching.

**All tests (iterates through all available signals by default):**
```bash
docker compose run --rm -e EVAL_MODE=test evaluation
```
Tests clean the database, create a bank, upload all images from `resources/images/`, and then run all test suites (pairwise, topk, threshold).

**Test with clip (integer thresholds):**
```bash
docker compose run --rm -e EVAL_MODE=test -e SIGNAL_TYPE=clip evaluation
```

**With custom parameters:**
```bash
docker compose run --rm \
  -e EVAL_MODE=test \
  -e MAX_K=10 \
  -e THRESHOLD_MAX=100 \
  -e THRESHOLD_STEP=20 \
  evaluation
```

### Local development (without docker run)
- Ensure services are up (step 2). The compose file exposes:
  - HMA on host port `5005`
  - PostgreSQL on host port `55432`
- Then you can run locally via the Makefile:
```bash
make smoke-test       # Uses localhost:5005 and localhost:55432
make run-evaluation   # Runs all tests locally
```

### Performance Metrics & Visualizations

After running tests with `EVAL_MODE=test`, metrics and graphs are **automatically generated** and saved to:
```
OUTPUT_DIR/evaluation_results/<timestamp>/
├── pairwise_<signal_type>_compare.json
├── map_by_series_<signal_type>_results.csv
├── precision_recall_<signal_type>_results.csv
├── precision_recall_<signal_type>_curve.png
└── distance_distribution_<signal_type>.png
```

Example: `./results/evaluation_results/20251215_193616/`

You can also compute metrics manually:

**mAP (Mean Average Precision):**
```bash
python metrics/map.py \
  --labels resources/labels/images_series_labels.json \
  --pairwise results/evaluation_results/<timestamp>/pairwise_clip_compare.json \
  --output_csv results/evaluation_results/<timestamp>/map_by_series.csv
```

**Classification Precision-Recall (threshold sweep):**
```bash
python metrics/precision_recall.py \
  --labels resources/labels/images_series_labels.json \
  --pairwise results/evaluation_results/<timestamp>/pairwise_clip_compare.json \
  --output_csv precision_recall_results.csv \
  --output_plot precision_recall_curve.png
```

**Distance Distribution Histograms:**
```bash
python metrics/distance_distribution.py \
  --labels resources/labels/images_series_labels.json \
  --pairwise results/evaluation_results/<timestamp>/pairwise_clip_compare.json \
  --output_plot distance_distribution.png \
  --signal_type clip
```

**What gets generated automatically:**
- **MAP CSV**: Mean Average Precision for each series at different k values
- **Precision-Recall CSV**: Threshold sweep results with TP/FP/FN counts
- **Precision-Recall Plot**: Visualization of the precision-recall curve
- **Distance Distribution Plot**: Side-by-side histograms (same-series vs different-series)

The terminal will show which files were created, e.g.:
```
✓ All tests completed.
Calculating MAP metrics...
  MAP@k for clip... ✓
    → map_by_series_clip_results.csv
  Classification PR for clip... ✓
    → precision_recall_clip_results.csv
    → precision_recall_clip_curve.png
  Distance distribution for clip... ✓
    → distance_distribution_clip.png
```

Notes:
- Paths in pairwise JSON generated inside Docker may start with `/build/`; the scripts normalize these automatically.
- All metrics support `--anon_map` parameter if using anonymized paths.

## Series Metadata

### What is a Series?

A **series** is a group of related images that should be recognized as similar by the matching system. Examples:
- Multiple photos of the same person (e.g., "Barbara_Walters")
- The same scene with transformations (rotations, flips)
- The same image with different filters applied

Series are defined in `resources/labels/images_series_labels.json`:

```json
{
  "series_name": [
    "./resources/images/image1.jpg",
    "./resources/images/image2.jpg"
  ]
}
```

### Series Requirements

- Each series must contain **at least 2 images**
- Series metadata is **required** for running metrics
- The evaluation automatically validates series metadata before running
- If metadata is missing or invalid, tests will fail with a clear error message

The series metadata is used to:
- Calculate MAP (Mean Average Precision) for each series
- Determine ground truth for precision/recall calculations
- Separate distance distributions into same-series vs different-series pairs

### Labels Creation (Optional)

**Automatic generation:** If `resources/labels/images_series_labels.json` doesn't exist and your image directory has series subfolders (like `resources/images/image-series-dataset/series`), labels will be auto-generated once on the first test run.

**Manual creation:** For flat image directories, you must manually create `resources/labels/images_series_labels.json` before running tests (or set `LABELS_PATH` to point to an existing labels file).

The included `image-series-dataset` is located at `resources/images/image-series-dataset/series`.

## Test Logs

All test runs create detailed logs in `OUTPUT_DIR/test_run_logs/` (default `./results/test_run_logs`):
- Format: `{test_run_type}_{date}_{time}.log` (e.g., `smoke_20251107_115430.log`)
- Terminal shows minimal progress output
- Full logs with timestamps saved to files

## Features

- **Automatic database cleanup**: Both smoke test and full test suite clean the database and populate it with test images
- **Test isolation**: Each test run starts with a fresh database and empty index
- **CLIP extension support**: HMA configured with CLIP signal type for semantic image matching
- **Custom endpoints**: Includes `lookup_topk` and `lookup_threshold` endpoints via patch
- **File logging**: All test output logged to files with minimal terminal noise

## HMA Configuration

- **Repository**: https://github.com/facebook/ThreatExchange
- **Commit**: build-time `THREATEXCHANGE_COMMIT` (default in `Dockerfile.hma`, override in `docker-compose.yaml`).
- **Port**: 5005 (host) → 5100 (container)
- **Network**: `shared-hma-network`
- **Config**: `omm_config.py` (includes CLIP extension)

## Environment Variables

### Test Configuration
- `EVAL_MODE`: `smoke` (default) or `test`
- `BANK_NAME`: Bank name for testing (default: `TEST_BANK_DATA`)
- `SIGNAL_TYPE`: Signal type to test (default: tests `clip`, `clip_float`, `cliphnsw`). Set to a single value like `clip_float` to test only that signal type.
- `MAX_K`: Maximum k for top-k test (default: `5`)
- `THRESHOLD_MAX`: Maximum threshold value (default: `1.0` for clip_float)
- `THRESHOLD_STEP`: Threshold step size (default: `0.2` for clip_float)
- `MAX_WORKERS`: Number of parallel workers for test API calls (default: `8`). Controls parallelization of pairwise/threshold/topk tests. Higher values = faster tests but more load on HMA server.

### HMA Connection (auto-configured in docker-compose)
- `HMA_HOST`: `hma-app` (internal container name)
- `HMA_PORT`: `5100` (internal container port)

### Output and Anonymization
- `OUTPUT_DIR`: Root directory for all outputs (default: `./results`).
  - Results: `OUTPUT_DIR/evaluation_results/<timestamp>/...`
  - Logs: `OUTPUT_DIR/test_run_logs/`
  - Mapping: `OUTPUT_DIR/file_to_id_map/anon_id_map.json` (default if not overridden)
- `DEANONYMIZE_IMAGE_PATHS`: Set to `1` to disable anonymization (filename anonymization is ON by default).
- `ANON_ID_MAP_FILEPATH`: Optional override for the mapping file path. If set, uses this as a persistent path→ID store (loads existing mappings and updates after runs). The file contains `{ "<full_path>": "<id>" }`. **DON'T SHARE THIS FILE** if you don't want to leak original image paths.

## Results

Test results are saved under `OUTPUT_DIR/evaluation_results/<timestamp>/` (default `./results/evaluation_results/<timestamp>/`):
- `pairwise_clip_compare.json`
- `topk_test_results.json`
- `threshold_test_results.json`

## Troubleshooting

```bash
# Check service status
docker compose ps

# View logs
docker compose logs hma-app
docker compose logs evaluation

# Check HMA API
curl http://localhost:5005/c/banks

# Verify database
docker compose exec hma-postgresql psql -U postgres -d media_match -c "\dt"
```

## Description of the tests

Refer to [EVALUATION_DESCRIPTION.md](EVALUATION_DESCRIPTION.md) for an overview of the tests.