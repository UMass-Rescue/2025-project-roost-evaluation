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

This builds HMA from ThreatExchange at commit `aff3f3b8` and starts:
- PostgreSQL database for HMA
- Database migrations
- HMA application (port 5005 on host, 5100 internally)
- Evaluation service

### 3. Run Tests

**Smoke test (default):**
```bash
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```

**All tests:**
```bash
docker compose run --rm -e EVAL_MODE=test evaluation
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

## Test Logs

All test runs create detailed logs in `test_run_logs/` folder:
- Format: `{test_run_type}_{date}_{time}.log` (e.g., `smoke_20251107_115430.log`)
- Terminal shows minimal progress output
- Full logs with timestamps saved to files

## Features

- **Automatic database cleanup**: Tests start with a clean database and empty index
- **CLIP extension support**: HMA configured with CLIP signal type for semantic image matching
- **Custom endpoints**: Includes `lookup_topk` and `lookup_threshold` endpoints via patch
- **File logging**: All test output logged to files with minimal terminal noise

## HMA Configuration

- **Repository**: https://github.com/facebook/ThreatExchange
- **Commit**: `aff3f3b8` (locked for reproducibility)
- **Port**: 5005 (host) → 5100 (container)
- **Network**: `shared-hma-network`
- **Config**: `omm_config.py` (includes CLIP extension)

## Environment Variables

### Test Configuration
- `EVAL_MODE`: `smoke` (default) or `test`
- `BANK_NAME`: Bank name for testing (default: `TEST_BANK_DATA`)
- `MAX_K`: Maximum k for top-k test (default: `5`)
- `THRESHOLD_MAX`: Maximum threshold value (default: `100`)
- `THRESHOLD_STEP`: Threshold step size (default: `20`)

### HMA Connection (auto-configured in docker-compose)
- `HMA_HOST`: `hma-app` (internal container name)
- `HMA_PORT`: `5100` (internal container port)

## Results

Test results are saved as JSON files in the project root:
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
