# HMA Test Commands Reference

Run `docker compose up --build -d` first (see root [README.md](../README.md)'s Quick Start section).

## Run Tests

**Note**: By default, all tests run for both `clip` and `clip_float` signal types. Results are saved with signal type in filename (e.g., `pairwise_clip_compare.json`, `pairwise_clip_float_compare.json`).

**To test only one signal type**, set the `SIGNAL_TYPE` environment variable (e.g., `-e SIGNAL_TYPE=clip`).

### Smoke Test
```bash
# Tests both clip and clip_float
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```

### All Tests
```bash
# Runs all tests for both clip and clip_float signal types (default)
docker compose run --rm -e EVAL_MODE=test evaluation

# Run all tests for ONLY clip signal type
docker compose run --rm -e EVAL_MODE=test -e SIGNAL_TYPE=clip evaluation

# Run all tests for ONLY clip_float signal type
docker compose run --rm -e EVAL_MODE=test -e SIGNAL_TYPE=clip_float evaluation
```

### Individual Tests
```bash
# Pairwise test (specify signal type)
docker compose run --rm -e SIGNAL_TYPE=clip evaluation python hma/tests/pairwise_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float evaluation python hma/tests/pairwise_test.py

# Top-k test
docker compose run --rm -e SIGNAL_TYPE=clip -e MAX_K=5 evaluation python hma/tests/topk_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float -e MAX_K=5 evaluation python hma/tests/topk_test.py

# Threshold test
docker compose run --rm -e SIGNAL_TYPE=clip -e THRESHOLD_MAX=50 -e THRESHOLD_STEP=10 evaluation python hma/tests/threshold_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float -e THRESHOLD_MAX=1.0 -e THRESHOLD_STEP=0.2 evaluation python hma/tests/threshold_test.py
```

### Custom Parameters
```bash
docker compose run --rm \
  -e EVAL_MODE=test \
  -e MAX_K=10 \
  -e THRESHOLD_MAX=100 \
  -e THRESHOLD_STEP=20 \
  evaluation
```

## Test Logs

Logs saved to `test_run_logs/` folder:
- Format: `{type}_{date}_{time}.log` (e.g., `smoke_20251107_115430.log`)
- View latest: `ls -t test_run_logs/ | head -1 | xargs cat`

## Verify Services

```bash
# Check HMA API
curl http://localhost:5005/c/banks

# Check database
docker compose exec hma-postgresql psql -U postgres -d media_match -c "\dt"

# Check HMA health
docker compose exec hma-postgresql pg_isready -U postgres
```

## View Logs

```bash
# Service logs
docker compose logs hma-app
docker compose logs evaluation

# Follow logs
docker compose logs -f hma-app
```

## Results

Results are saved as JSON files with signal type in filename:
```bash
# List results for a test run
ls -lh evaluations_results/<timestamp>/

# View pairwise results
cat evaluations_results/<timestamp>/pairwise_clip_compare.json | head -50
cat evaluations_results/<timestamp>/pairwise_clip_float_compare.json | head -50
```

## Performance Metrics & Graphs

After tests complete, `hma/evaluate.py` automatically generates performance metrics and visualizations:

### Outputs Generated

For each signal type (clip, clip_float):

**MAP (Mean Average Precision):**
- CSV: `map_by_series_{signal_type}_results.csv`

**Classification Precision-Recall (threshold sweep):**
- CSV: `precision_recall_{signal_type}_results.csv`
- Plot: `precision_recall_{signal_type}_curve.png`

**Distance Distributions:**
- Plot: `distance_distribution_{signal_type}.png` (histograms comparing same-series vs different-series pairs)

### View Results

```bash
# View CSV results
cat evaluations_results/<timestamp>/map_by_series_clip_results.csv
cat evaluations_results/<timestamp>/precision_recall_clip_results.csv

# View graphs (open in browser/viewer)
open evaluations_results/<timestamp>/precision_recall_clip_curve.png
open evaluations_results/<timestamp>/distance_distribution_clip.png
```

### Manual Metric Computation

**MAP:**
```bash
PYTHONPATH=. python hma/metrics/map.py \
  --labels hma/resources/labels/images_series_labels.json \
  --pairwise evaluations_results/<timestamp>/pairwise_clip_compare.json \
  --output_csv custom_map_results.csv
```

**Classification Precision-Recall (threshold sweep):**
```bash
PYTHONPATH=. python hma/metrics/precision_recall.py \
  --labels hma/resources/labels/images_series_labels.json \
  --pairwise evaluations_results/<timestamp>/pairwise_clip_compare.json \
  --output_csv precision_recall_results.csv \
  --output_plot precision_recall_curve.png
```

**Distance distribution:**
```bash
PYTHONPATH=. python hma/metrics/distance_distribution.py \
  --labels hma/resources/labels/images_series_labels.json \
  --pairwise evaluations_results/<timestamp>/pairwise_clip_compare.json \
  --output_plot distance_distribution.png \
  --signal_type clip
```

## Troubleshooting

```bash
# Test connectivity
docker compose run --rm evaluation curl -s http://hma-app:5100/c/banks

# Check database connection
docker compose run --rm evaluation python -c "import psycopg2; psycopg2.connect('postgresql://postgres:postgres@hma-postgresql/media_match')"
```
