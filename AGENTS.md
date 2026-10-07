# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

This repo evaluates two independent systems against ground-truth datasets:

- **HMA** (Hasher-Matcher-Actioner) — image-hash matching (PDQ, CLIP, CLIP-float
  signals) via a dockerized OpenMediaMatch service.
- **Sentinel** — a text-based grooming classifier, via its own dockerized API
  service.

## Repo layout

- Root-level `evaluate.py`, `metrics/`, `resources/`, `tests/` are **HMA-specific**.
- `sentinel/` holds the equivalent shape for the **Sentinel** domain.

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

See `TEST_COMMANDS.md`, `EVALUATION_DESCRIPTION.md`, and `GPU_SETUP.md` for
full detail — don't duplicate that detail here.

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
- Never push directly to a repo's main/default branch — this applies to this
  repo and any other repo touched while working here (e.g. a dataset repo
  cloned under `sentinel/dataset/`). Always push to a feature branch and open
  a PR, unless the user explicitly asks for a direct push to main.

## Further reading

- `README.md` — full setup, configuration, and HMA/Sentinel service details
- `EVALUATION_DESCRIPTION.md` — what each test suite actually measures
- `TEST_COMMANDS.md` — exhaustive command reference
- `GPU_SETUP.md` — GPU acceleration setup
- `sentinel/README.md` — Sentinel service wiring and evaluation code layout
