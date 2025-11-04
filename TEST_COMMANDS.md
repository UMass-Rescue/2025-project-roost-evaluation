# Test Commands for HMA Docker Compose Setup

## Note on Command Usage

- **`docker compose exec <service>`**: Used for services that are **running continuously** (like `hma-app`, `hma-postgresql`)
- **`docker compose run --rm <service>`**: Used for services that **run and exit** (like `evaluation`), or to run one-off commands

The `evaluation` service exits after completing its task, so always use `run --rm` for it, not `exec`.

## Prerequisites Check

```bash
# Check if Docker and Docker Compose are installed
docker --version
docker compose version

# Check if the network exists (create if it doesn't)
docker network ls | grep shared-hma-network || docker network create shared-hma-network
```

## 1. Build and Start Services

```bash
# Build and start all services in detached mode
docker compose up --build -d

# Check status of all services
docker compose ps

# View logs from all services
docker compose logs

# View logs from specific services
docker compose logs hma-app
docker compose logs hma-postgresql
docker compose logs evaluation
```

## 2. Verify HMA Services are Running

```bash
# Check if HMA PostgreSQL is healthy
docker compose exec hma-postgresql pg_isready -U postgres

# Check if HMA app is responding (from host)
curl http://localhost:5005/health || curl http://localhost:5005/

# Check HMA app from within the Docker network
docker compose run --rm evaluation curl http://hma-app:5100/health || docker compose run --rm evaluation curl http://hma-app:5100/

# List running containers
docker ps --filter "name=hma" --filter "name=evaluation"
```

## 3. Test Database Connection

```bash
# Connect to PostgreSQL and check databases
docker compose exec hma-postgresql psql -U postgres -c "\l"

# Check if migrations ran successfully (should see media_match database and tables)
docker compose exec hma-postgresql psql -U postgres -d media_match -c "\dt"
```

## 4. Test HMA API Endpoints

```bash
# Test HMA API from host (should work if port mapping is correct)
curl http://localhost:5005/c/banks

# Test from within evaluation container (using Docker network)
docker compose run --rm evaluation curl http://hma-app:5100/c/banks

# Test hash endpoint (will need an image file)
docker compose run --rm evaluation curl -X POST http://hma-app:5100/h/hash -F "photo=@/build/resources/images/square-256x256.jpg"
```

## 5. Test Evaluation Service Connection

```bash
# Check environment variables in evaluation container
docker compose run --rm evaluation env | grep HMA

# Test Python import and basic connectivity
docker compose run --rm evaluation python -c "from evaluate import Evaluator, hma_app_url; print(f'✓ HMA URL: {hma_app_url}')"

# Test bank creation (smoke test)
docker compose run --rm evaluation python -c "from evaluate import Evaluator; e = Evaluator(); print('Bank exists:', e.bank_exists('TEST_BANK'))"
```

## 6. Run Smoke Test

```bash
# Run the default smoke test
docker compose run --rm evaluation

# Run smoke test with custom bank name
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation
```

## 7. Run Individual Tests

```bash
# Run pairwise test
docker compose run --rm -e EVAL_MODE=test evaluation python tests/pairwise_test.py

# Run top-k test
docker compose run --rm -e MAX_K=5 evaluation python tests/topk_test.py

# Run threshold test
docker compose run --rm -e THRESHOLD_MAX=50 -e THRESHOLD_STEP=10 evaluation python tests/threshold_test.py

# Run cleanup banks script
docker compose run --rm evaluation python tests/cleanup_banks.py
```

## 8. Run Full Test Suite

```bash
# Run all tests with custom parameters
docker compose run --rm \
  -e EVAL_MODE=test \
  -e MAX_K=10 \
  -e THRESHOLD_MAX=100 \
  -e THRESHOLD_STEP=20 \
  evaluation

# Check if results files were created
ls -lh *.json
```

## 9. Verify Results Files

```bash
# Check if result files exist
ls -lh pairwise_clip_compare.json topk_test_results.json threshold_test_results.json 2>/dev/null || echo "Results files not found"

# View a results file
cat pairwise_clip_compare.json | head -50
```

## 10. Test Service Dependencies

```bash
# Stop HMA app and verify evaluation waits or fails gracefully
docker compose stop hma-app
docker compose run --rm evaluation python -c "from evaluate import Evaluator; e = Evaluator(); print('Connecting...')" || echo "Expected failure - HMA is down"

# Restart HMA app
docker compose start hma-app

# Wait a few seconds for HMA to be ready, then test again
sleep 5
docker compose run --rm evaluation python -c "from evaluate import Evaluator; e = Evaluator(); print('Connecting...')"
```

## 11. Test Port Mapping

```bash
# Test external port access (from host)
curl -v http://localhost:5005/c/banks

# Test internal port access (from container)
docker compose run --rm evaluation curl -v http://hma-app:5100/c/banks
```

## 12. Verify Commit Lock

```bash
# Check the commit hash in Dockerfile.hma
grep THREATEXCHANGE_COMMIT Dockerfile.hma

# Verify commit hash in docker-compose.yaml
grep THREATEXCHANGE_COMMIT docker-compose.yaml

# Build logs should show the commit being checked out
docker compose build hma-app 2>&1 | grep -i commit
```

## 13. Cleanup and Restart Tests

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Remove all containers and rebuild
docker compose down --rmi local
docker compose build --no-cache

# Start fresh
docker compose up --build -d
```

## 14. Monitor Logs During Test Run

```bash
# In one terminal, watch HMA logs
docker compose logs -f hma-app

# In another terminal, run tests
docker compose run --rm -e EVAL_MODE=test evaluation
```

## 15. Quick Health Check Script

```bash
# Run all health checks at once
echo "=== Checking Services ==="
docker compose ps

echo -e "\n=== Checking Network ==="
docker network inspect shared-hma-network | grep -A 5 "Containers"

echo -e "\n=== Checking HMA API ==="
curl -s http://localhost:5005/c/banks | head -20 || echo "HMA API not responding"

echo -e "\n=== Checking Environment Variables ==="
docker compose run --rm evaluation env | grep HMA

echo -e "\n=== Testing Connection ==="
docker compose run --rm evaluation python -c "from evaluate import Evaluator, hma_app_url; print(f'✓ Connected to: {hma_app_url}')"
```

## Expected Results

When everything is working correctly, you should see:

1. ✅ All services (hma-postgresql, hma-migrations, hma-app, evaluation) are running
2. ✅ HMA API responds at http://localhost:5005
3. ✅ Evaluation container can connect to hma-app:5100
4. ✅ Smoke test creates a bank and uploads images
5. ✅ Test suite generates JSON result files
6. ✅ Results files appear in project root directory
7. ✅ No connection errors in logs

## Troubleshooting

If tests fail, check:

```bash
# Check for errors in logs
docker compose logs | grep -i error

# Check container status
docker compose ps -a

# Check network connectivity (note: ping may not be installed, curl is more reliable)
docker compose run --rm evaluation curl -s http://hma-app:5100/c/banks > /dev/null && echo "✓ Network connectivity OK" || echo "✗ Network connectivity failed"

# Check HMA app is listening
docker compose exec hma-app netstat -tlnp 2>/dev/null | grep 5100 || docker compose exec hma-app ss -tlnp | grep 5100

# Check PostgreSQL is accessible
docker compose run --rm evaluation python -c "import psycopg2; psycopg2.connect('postgresql://postgres:postgres@hma-postgresql/media_match')" && echo "✓ DB connected" || echo "✗ DB connection failed"
```

