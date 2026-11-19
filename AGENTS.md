# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

This repo evaluates two independent systems against ground-truth datasets:

- **HMA** (Hasher-Matcher-Actioner) — image-hash matching (PDQ, CLIP, CLIP-float
  signals) via a dockerized OpenMediaMatch service.
- **Sentinel** — a rare-class affinity detector for text, evaluated here on a
  grooming-conversation dataset, via its own dockerized API service.

## Repo layout

- `hma/` holds everything HMA-specific. Code shared by the HMA evaluations
  sits at its root (`hma_client.py`, `logging_utils.py`, `results_utils.py`,
  `path_id_store.py`, `series_labels_utils.py`, `metrics/`, `unit_tests/`), next to the HMA build
  (`Dockerfile.hma`, `omm_config.py`, `resources/patches/`) and `GPU_SETUP.md`.
  Each content type has its own folder: `hma/image/` holds `evaluate.py`,
  `tests/`, `resources/`, `series_labels_utils.py` (the image file types over the shared logic)
  and `docs/` (`TEST_COMMANDS.md`, `EVALUATION_DESCRIPTION.md`). `hma/video/` holds the
  video evaluation (`evaluate.py`, `smoke_test.py`, the frame-selection pipeline, a `README.md`
  and a `docs/` folder with `EVALUATION_DESCRIPTION.md` and `FRAME_SELECTION.md`); its dataset is
  fetched into `hma/video/resources/`, never committed. Each content type keeps its detailed docs
  in its own `docs/` folder.
- `sentinel/` holds the equivalent shape for the **Sentinel** domain.
- Root-level files (`README.md`, `docker-compose.yaml`, `Dockerfile`,
  `requirements.txt`, `Makefile`) are shared across both domains —
  `Dockerfile`/`requirements.txt` build the single `evaluation` image both
  `hma/image/evaluate.py` and `sentinel/evaluate.py` run in.
- `hma/image/docs/TEST_COMMANDS.md` and `sentinel/README.md` hold each domain's own
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
make video-smoke-test  # video evaluation smoke test (generated sample clips)
make fetch-video-dataset && make run-video-evaluation   # full video evaluation
```

See `hma/image/docs/TEST_COMMANDS.md`, `hma/video/README.md`, `sentinel/README.md`,
`hma/image/docs/EVALUATION_DESCRIPTION.md`, and `hma/GPU_SETUP.md` for full detail — don't duplicate that detail here.

## CI

`.github/workflows/ci.yml` runs lint, unit tests and the image build on every PR; keep it
green. PRs use `.github/pull_request_template.md`.

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
- `hma/README.md` / `hma/image/docs/TEST_COMMANDS.md` — HMA setup, configuration, and
  exhaustive command reference
- `hma/image/docs/EVALUATION_DESCRIPTION.md` — what each HMA test suite actually measures
- `hma/video/README.md` — video evaluation quick start; `hma/video/docs/EVALUATION_DESCRIPTION.md`
  and `hma/video/docs/FRAME_SELECTION.md` hold what it measures and how frames are chosen
- `hma/GPU_SETUP.md` — GPU acceleration setup
- `sentinel/README.md` — Sentinel service wiring, evaluation code layout, and
  command reference
