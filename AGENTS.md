# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

This repo evaluates two independent systems against ground-truth datasets:

- **HMA** (Hasher-Matcher-Actioner) — image-hash matching (PDQ, CLIP, CLIP-float
  signals) via a dockerized OpenMediaMatch service.
- **Sentinel** — a rare-class affinity detector for text, evaluated here on a
  grooming-conversation dataset, via its own dockerized API service.

## Repo layout

- `hma/` holds everything HMA-specific: `evaluate.py`, `metrics/`,
  `resources/`, `tests/`, `series_labels_utils.py`, `omm_config.py`,
  `Dockerfile.hma`, plus its own `GPU_SETUP.md` and
  `EVALUATION_DESCRIPTION.md`.
- `sentinel/` holds the equivalent shape for the **Sentinel** domain.
- Root-level files (`README.md`, `docker-compose.yaml`, `Dockerfile`,
  `requirements.txt`, `Makefile`) are shared across both domains —
  `Dockerfile`/`requirements.txt` build the single `evaluation` image both
  `hma/evaluate.py` and `sentinel/evaluate.py` run in.
- `hma/TEST_COMMANDS.md` and `sentinel/README.md` hold each domain's own
  exhaustive command reference; the root `README.md` only covers shared
  setup/cleanup.

These stay separate by design — different data types, different metrics
(retrieval/distance metrics vs. classification metrics). Don't merge them
into shared modules.

## Setup & common commands

```bash
docker network create shared-hma-network
docker compose up --build -d
docker compose run --rm -e BANK_NAME=SMOKE_TEST evaluation   # smoke test
docker compose run --rm -e EVAL_MODE=test evaluation         # full test suite
make smoke-test       # local equivalent, against localhost
make run-evaluation    # local equivalent, full suite
```

See `hma/TEST_COMMANDS.md`, `sentinel/README.md`, `hma/EVALUATION_DESCRIPTION.md`,
and `hma/GPU_SETUP.md` for full detail — don't duplicate that detail here.

## Code conventions

- stdlib only for data handling (`csv`, `pathlib.Path`, `dataclasses`) — no
  `pandas`.
- Configuration via env vars (`SIGNAL_TYPE`, `EVAL_MODE`, `OUTPUT_DIR`,
  per-service host/port vars), not hardcoded values.
- External services (HMA/ThreatExchange, Sentinel API) are built via
  pinned-commit Docker builds, never vendored/copied into this repo's source
  tree.

## Sensitive-data handling

- Any real dataset under `sentinel/dataset/` (or any future sensitive eval
  data) must never be committed — gitignored, access-controlled, obtained
  separately.
- Dataset-specific preprocessing/cleanup logic belongs in that dataset's own
  repo, not here — this repo only ever consumes an already-clean,
  pre-labeled export.
- Don't print/log raw sensitive content (e.g. full conversation text) into
  shared outputs, logs, or commit messages.

## Git/PR hygiene

- Prefer clean, squashed, single-purpose commits.
- PR descriptions should reflect only what's actually in the diff/commits,
  not leftover context from earlier iterations.
- Scope each PR to one concern.

## Further reading

- `README.md` — shared setup/cleanup; links out to the two systems below
- `hma/README.md` / `hma/TEST_COMMANDS.md` — HMA setup, configuration, and
  exhaustive command reference
- `hma/EVALUATION_DESCRIPTION.md` — what each HMA test suite actually measures
- `hma/GPU_SETUP.md` — GPU acceleration setup
- `sentinel/README.md` — Sentinel service wiring, evaluation code layout, and
  command reference
