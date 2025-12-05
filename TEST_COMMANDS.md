# Test Commands Reference

## Quick Reference

- **`docker compose exec`**: Run commands in running services (hma-app, hma-postgresql)
- **`docker compose run --rm`**: Run one-off commands in evaluation service (exits after completion)

## Setup

```bash
# Create network
docker network create shared-hma-network

# Start services
docker compose up --build -d

# Check status
docker compose ps
```

## Run Tests

**Note**: By default, all tests run for both `clip` and `clip_float` signal types. Results are saved with signal type in filename (e.g., `pairwise_clip_compare.json`, `pairwise_clip_float_compare.json`).

### Smoke Test
```bash
# Tests both clip and clip_float
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```

### All Tests
```bash
# Runs all tests for both clip and clip_float signal types
docker compose run --rm -e EVAL_MODE=test evaluation
```

### Individual Tests
```bash
# Pairwise test (specify signal type)
docker compose run --rm -e SIGNAL_TYPE=clip evaluation python tests/pairwise_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float evaluation python tests/pairwise_test.py

# Top-k test
docker compose run --rm -e SIGNAL_TYPE=clip -e MAX_K=5 evaluation python tests/topk_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float -e MAX_K=5 evaluation python tests/topk_test.py

# Threshold test
docker compose run --rm -e SIGNAL_TYPE=clip -e THRESHOLD_MAX=50 -e THRESHOLD_STEP=10 evaluation python tests/threshold_test.py
docker compose run --rm -e SIGNAL_TYPE=clip_float -e THRESHOLD_MAX=1.0 -e THRESHOLD_STEP=0.2 evaluation python tests/threshold_test.py
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

## MAP Metrics

MAP (Mean Average Precision) metrics are automatically computed by `evaluate.py` after tests complete. View the results:

```bash
# View MAP results for CLIP
cat evaluations_results/<timestamp>/map_by_series_clip_results.csv

# View MAP results for CLIP_float
cat evaluations_results/<timestamp>/map_by_series_clip_float_results.csv
```

The CSV files contain mAP@k values for each image series (k=1 to max series size) with a mean row at the bottom.

You can also manually compute MAP from pairwise results:
```bash
python metrics/map.py \
  --labels resources/labels/images_series_labels.json \
  --pairwise evaluations_results/<timestamp>/pairwise_clip_compare.json \
  --output_csv custom_map_results.csv
```

## Cleanup

```bash
# Stop services
docker compose down

# Remove volumes (fresh start)
docker compose down -v

# Rebuild
docker compose build --no-cache
docker compose up --build -d
```

## Troubleshooting

```bash
# Check for errors
docker compose logs | grep -i error

# Test connectivity
docker compose run --rm evaluation curl -s http://hma-app:5100/c/banks

# Check database connection
docker compose run --rm evaluation python -c "import psycopg2; psycopg2.connect('postgresql://postgres:postgres@hma-postgresql/media_match')"
```
