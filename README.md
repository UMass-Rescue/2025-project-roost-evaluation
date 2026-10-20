# 2025 Project Roost Evaluation

## Overview

This repo evaluates two independent systems against ground-truth datasets:

- **[HMA](hma/README.md)** (Hasher-Matcher-Actioner) — image-hash matching
  (PDQ, CLIP, CLIP-float signals) from Facebook's ThreatExchange repository,
  via a dockerized OpenMediaMatch service.
- **[Sentinel](sentinel/README.md)** — a rare-class affinity detector for text,
  evaluated here on a grooming-conversation dataset, via its own dockerized
  API service.

They're kept separate by design — different data types and different metrics
(retrieval/distance metrics for HMA vs. classification metrics for Sentinel),
so their code, docs, and tests live in their own `hma/` and `sentinel/`
folders rather than shared modules. This doc only covers what's shared
between them; follow the linked doc for either system's actual setup and
test commands.

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

This builds and starts, for both systems:
- HMA: PostgreSQL database, database migrations, and the HMA application
  (port 5005 on host, 5100 internally)
- Sentinel: the `sentinel-app` API service (port 8001 on host, 8000 internally)
- The shared `evaluation` service both systems' test scripts run in

### 3. Run an Evaluation

- **HMA**: see [hma/README.md](hma/README.md) and [hma/TEST_COMMANDS.md](hma/TEST_COMMANDS.md)
  for test commands, metrics, and configuration.
- **Sentinel**: see [sentinel/README.md](sentinel/README.md) for test commands,
  dataset setup, and configuration.

## Environment Variables

These two are read by both systems' evaluation scripts; everything else is
system-specific (see the linked docs above):

- `EVAL_MODE`: `smoke` (default) or `test`
- `OUTPUT_DIR`: root directory for all outputs (default: `./results`)

## Troubleshooting

```bash
# Check service status
docker compose ps

# View evaluation service logs
docker compose logs evaluation

# Check for errors across all services
docker compose logs | grep -i error
```

For system-specific troubleshooting (service health checks, database
connectivity, etc.), see [hma/README.md](hma/README.md#troubleshooting) or
[sentinel/README.md](sentinel/README.md).

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
