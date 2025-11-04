# 2025 Project Roost Evaluation

## Overview

This project evaluates content matching capabilities using HMA (Hasher-Matcher-Actioner) from Facebook's ThreatExchange repository. The project includes a docker-compose setup that automatically builds and runs HMA from a locked commit of the ThreatExchange repository, along with an evaluation service that runs tests against the local HMA container.

## Prerequisites

- Docker and Docker Compose must be installed on your machine
- Git (for cloning the repository)

## Quick Start

### 1. Create the Docker Network

First, create the shared Docker network that all services will use:

```bash
docker network create shared-hma-network
```

### 2. Start All Services

Use Docker Compose to build and start all services (HMA PostgreSQL, migrations, HMA app, and evaluation):

```bash
docker compose up --build
```

This will:
- Build HMA from the ThreatExchange repository at commit `2e5f23f6526e5c08793c946fa38f02baf3e56747` (locked to current main branch)
- Start PostgreSQL database for HMA
- Run database migrations
- Start the HMA application on port 5005 (accessible from host)
- Start the evaluation service configured to connect to the local HMA container

### 3. Running the Evaluation

The evaluation service automatically connects to the local HMA container via the Docker network. By default, it runs in "smoke" mode which:
- Creates a test bank
- Uploads sample images
- Performs matching operations

To run in smoke mode (default):
```bash
docker compose up evaluation
```

To run all tests:
```bash
docker compose run --rm -e EVAL_MODE=test evaluation
```

---

## HMA Configuration

The HMA container is built from the ThreatExchange repository and locked to a specific commit:
- **Repository**: https://github.com/facebook/ThreatExchange
- **Commit**: `2e5f23f6526e5c08793c946fa38f02baf3e56747` (current main branch)
- **Port**: HMA API is exposed on port 5005 (mapped from internal port 5100)
- **Network**: All services run on `shared-hma-network` for inter-container communication


---

## Running Tests and Retrieving Results

### Using Docker Compose (Recommended)

With the docker-compose setup, tests automatically connect to the local HMA container. The project directory is automatically mounted to `/build` in the container, so results are saved directly to your project directory.

1. **Run all tests:**

   ```bash
   docker compose run --rm -e EVAL_MODE=test -e MAX_K=10 -e THRESHOLD_MAX=100 -e THRESHOLD_STEP=20 evaluation
   ```

   - The `-e EVAL_MODE=test` environment variable tells the container to run all tests.
   - Results are automatically saved to your project directory (e.g., `pairwise_clip_compare.json`, `topk_test_results.json`).

2. **Run a single test:**

   For example, to run only the top-k test:

   ```bash
   docker compose run --rm \
     -e MAX_K=10 \
     -e OUTPUT_FILE=topk_results.json \
     evaluation python tests/topk_test.py
   ```

3. **Find the results:**

   After the container finishes, you will find the result JSON files (e.g., `pairwise_clip_compare.json`, `topk_test_results.json`) in your project root directory.

### Using Standalone Docker (Alternative)

If you prefer to run tests separately without docker-compose, ensure HMA is running and accessible:

1. **Build the Docker image:**

   ```bash
   docker build -t roost-eval .
   ```

2. **Run tests pointing to local HMA:**

   ```bash
   docker run --rm \
     --network shared-hma-network \
     -e EVAL_MODE=test \
     -e HMA_HOST=hma-app \
     -e HMA_PORT=5100 \
     -e MAX_K=10 \
     -e THRESHOLD_MAX=100 \
     -e THRESHOLD_STEP=20 \
     -v "$PWD:/build" \
     roost-eval
   ```

   Note: When using standalone Docker, you must be on the same network (`shared-hma-network`) and use `HMA_HOST=hma-app` and `HMA_PORT=5100` to connect to the HMA container.

---

## Runtime Configuration Variables

The evaluation pipeline supports several environment variables for customization:

### HMA API Configuration

- **`HMA_HOST`**: Hostname for the HMA API server
  - Default: `host.docker.internal` (when running standalone)
  - Default in docker-compose: `hma-app` (configured automatically)
  - Used in: `evaluate.py`, `tests/cleanup_banks.py`
  - When using docker-compose, the evaluation service automatically uses `hma-app` to connect to the local HMA container
  - Example (standalone): `docker run --rm -e HMA_HOST=localhost -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

- **`HMA_PORT`**: Port number for the HMA API server
  - Default: `5005` (when accessing from host)
  - Default in docker-compose: `5100` (internal port, configured automatically)
  - Used in: `evaluate.py`, `tests/cleanup_banks.py`
  - When using docker-compose, the evaluation service automatically uses port `5100` (internal container port)
  - External access: Port `5005` on host maps to port `5100` in the HMA container
  - Example (standalone): `docker run --rm -e HMA_PORT=5000 -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

### Test Configuration

- **`EVAL_MODE`**: Controls the evaluation mode
  - Default: `smoke` (runs smoke test)
  - Options: `smoke`, `test` (runs all tests)
  - Used in: `evaluate.py`

- **`OUTPUT_FILE`**: Specifies the output filename for test results
  - Default: Varies by test (e.g., `pairwise_clip_compare.json`, `topk_test_results.json`)
  - Used in: `tests/pairwise_test.py`, `tests/topk_test.py`, `tests/threshold_test.py`
  - Example: `docker run --rm -e OUTPUT_FILE=my_results.json -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

- **`MAX_K`**: In top-k test, specifies the maximum k value to test, testing the range [1, MAX_K].
  - Default: `5`
  - Used in: `tests/topk_test.py`

- **`THRESHOLD_MAX`**: In the threshold test, this specifies the maximum threshold to test. The test will run from 0 to `THRESHOLD_MAX`.
  - Default: `100`
  - Used in: `tests/threshold_test.py`

- **`THRESHOLD_STEP`**: In the threshold test, this specifies the step size for the threshold range.
  - Default: `20`
  - Used in: `tests/threshold_test.py`


### Bank Management

- **`BANK_NAME`**: Name of the HMA bank to use for testing
  - Default: `TEST_BANK_DATA`
  - Used in: `evaluate.py`
  - Example: `docker run --rm -e BANK_NAME=MY_TEST_BANK -e EVAL_MODE=smoke -v "$PWD:/build" roost-eval`

### Input Configuration

- **`IMAGE_INPUT_DIR`**: Directory containing images for processing
  - Default: `./resources/images` (relative to project root)
  - Used in: `tests/test_utils.py`
  - Example: `docker run --rm -e IMAGE_INPUT_DIR=/custom/images -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

### Example with Multiple Custom Configuration

**Using Docker Compose (recommended):**

```bash
docker compose run --rm \
  -e EVAL_MODE=test \
  -e MAX_K=5 \
  -e THRESHOLD_MAX=80 \
  -e THRESHOLD_STEP=10 \
  -e OUTPUT_FILE=custom_results.json \
  -e BANK_NAME=CUSTOM_BANK \
  -e IMAGE_INPUT_DIR=/custom/images \
  evaluation
```

**Using Standalone Docker:**

```bash
docker run --rm \
  --network shared-hma-network \
  -e EVAL_MODE=test \
  -e MAX_K=5 \
  -e THRESHOLD_MAX=80 \
  -e THRESHOLD_STEP=10 \
  -e OUTPUT_FILE=custom_results.json \
  -e BANK_NAME=CUSTOM_BANK \
  -e HMA_HOST=hma-app \
  -e HMA_PORT=5100 \
  -e IMAGE_INPUT_DIR=/custom/images \
  -v "$PWD:/build" \
  roost-eval
```