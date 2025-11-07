# Test Commands Reference

## Quick Reference

- **`docker compose exec`**: For running services (hma-app, hma-postgresql)
- **`docker compose run --rm`**: For evaluation service (runs and exits)

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

### Smoke Test
```bash
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```

### All Tests
```bash
docker compose run --rm -e EVAL_MODE=test evaluation
```

### Individual Tests
```bash
# Pairwise test
docker compose run --rm -e EVAL_MODE=test evaluation python tests/pairwise_test.py

# Top-k test
docker compose run --rm -e MAX_K=5 evaluation python tests/topk_test.py

# Threshold test
docker compose run --rm -e THRESHOLD_MAX=50 -e THRESHOLD_STEP=10 evaluation python tests/threshold_test.py
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

Logs are saved to `test_run_logs/` folder:
- Format: `{type}_{date}_{time}.log` (e.g., `smoke_20251107_115430.log`)
- View latest log: `ls -t test_run_logs/ | head -1 | xargs cat`

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

Results are saved as JSON files:
```bash
ls -lh *.json
cat pairwise_clip_compare.json | head -50
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
