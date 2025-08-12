# 2025 Project Roost Evaluation

## Steps for Running HMA with CLIP Extensions

The HMA CLIP demo repository is available at [2025-project-hma-clip-demo](https://github.com/UMass-Rescue/2025-project-hma-clip-demo).

To get HMA CLIP up and running, follow these steps:

### Prerequisites

- Docker and Docker Compose must be installed on your machine.

### Setup

1. **Clone the Repository**

   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. **Launch the Services**

   Use Docker Compose to build and start the services defined in the `docker-compose.yml` file:

   ```bash
   docker compose up --build
   ```

   This command builds the Docker image and starts the services, including the application and the database.

   You can configure the PostgreSQL database and other environment variables for HMA CLIP in `docker-compose.yml` and `omm_config.py`.

---

## Instructions to Run the Evaluation Pipeline

1. **Ensure the Latest HMA Version**

   In the Dockerfile of HMA CLIP, ensure the latest version of HMA is pulled. If not, update line 1 in the Dockerfile to:

   ```dockerfile
   FROM ghcr.io/facebook/threatexchange/hma:1.0.17
   ```

   Alternatively, use the `hma:latest` tag.

2. **Use an External Network**

   Before running HMA CLIP, configure it to use an external network:

   - Create a Docker network called `shared-hma-network` by running the following command:

     ```bash
     docker network create shared-hma-network
     ```

   - Update the `networks` section in `docker-compose.yml` to:

     ```yaml
     networks:
       shared-hma-network:
         external: true
     ```

   - Under the `services` section, ensure the network is set to `shared-hma-network`.

3. **Check the port for HMA-CLIP**
   
   In macOS, 5000 port could be a reserved port. So, update the ports column under  `services`: `app` in `docker-compose.yml` of HMA CLIP to:

   ```yaml
   ports:
      - 5005:5000  
   ```

   - This forwards the requests on port 5005 on your machine to port 5000 on docker. You also need to change the hma_app_url variable to use port 5005 in evaluate.py of roost-evaluation project.


---

## Running Tests and Retrieving Results

To run the test suite (including pairwise_test.py) and retrieve the results file (pairwise_clip_compare.json) to your host machine, follow these steps:

1. **Build the Docker image (if not already built):**

   ```bash
   docker build -t roost-eval .
   ```

2. **Run the tests and save results to your host:**

   ```bash
   docker run --rm -e EVAL_MODE=test -v "$PWD:/build" roost-eval
   ```

   - The `-e EVAL_MODE=test` environment variable tells the container to run all tests.
   - The `-v "$PWD:/build"` flag mounts your current directory to `/build` in the container, so any files written to `/build` (such as `pairwise_results.json`) will appear in your project directory on your host.

3. **Find the results:**

   After the container finishes, you will find `pairwise_results.json` in your project root directory.

---

## Runtime Configuration Variables

The evaluation pipeline supports several environment variables for customization:

### HMA API Configuration

- **`HMA_APP_URL`**: Base URL for the HMA API endpoints
  - Default: `http://host.docker.internal:5005`
  - Used in: `evaluate.py`, `tests/cleanup_banks.py`
  - Example: `docker run --rm -e HMA_APP_URL=http://localhost:5005 -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

### Test Configuration

- **`EVAL_MODE`**: Controls the evaluation mode
  - Default: `smoke` (runs smoke test)
  - Options: `smoke`, `test` (runs all tests)
  - Used in: `evaluate.py`

- **`OUTPUT_FILE`**: Specifies the output filename for test results
  - Default: `pairwise_clip_compare.json`
  - Used in: `tests/pairwise_test.py`
  - Example: `docker run --rm -e OUTPUT_FILE=my_results.json -e EVAL_MODE=test -v "$PWD:/build" roost-eval`

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

```bash
docker run --rm \
  -e EVAL_MODE=test \
  -e OUTPUT_FILE=custom_results.json \
  -e BANK_NAME=CUSTOM_BANK \
  -e HMA_APP_URL=http://localhost:5005 \
  -e IMAGE_INPUT_DIR=/custom/images \
  -v "$PWD:/build" \
  roost-eval
```

---
