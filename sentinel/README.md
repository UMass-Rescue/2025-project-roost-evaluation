# Sentinel (grooming classifier) evaluation

Sentinel is a grooming classifier, served as its own HTTP API from a separate,
standalone repo: [UMass-Rescue/sentinel-api](https://github.com/UMass-Rescue/sentinel-api).
Its code is **not** vendored in this repo — `docker-compose.yaml` builds the
`sentinel-app` service directly from that repo via a pinned git-context build,
the same way `hma-app` is built from a pinned `THREATEXCHANGE_COMMIT`.

Make sure you've completed the root [README.md](../README.md)'s "Quick Start"
(network + `docker compose up --build -d`) before following this doc.

## Service wiring

- **Repository**: https://github.com/UMass-Rescue/sentinel-api
- **Commit**: pinned in `docker-compose.yaml`'s `sentinel-app.build.context`
  (currently `29c84fb9651f48d4e347fdc8e99cfcb7e7914ec7`)
- **Port**: 8001 (host) → 8000 (container)
- **Network**: `shared-hma-network`
- **Env vars for the `evaluation` service**: `SENTINEL_HOST=sentinel-app`,
  `SENTINEL_PORT=8000`, `SENTINEL_DATA_DIR=/data`
- **Shared volume**: `sentinel-data` is mounted at `/data` in both `sentinel-app`
  and `evaluation`. Sentinel's `POST /banks/create` takes filesystem folder
  paths rather than uploaded files, so bank-creation inputs/outputs have to live
  somewhere both containers can read/write at the same path.

To bump the pinned commit, update the ref after the `#` in `sentinel-app.build.context`
in `docker-compose.yaml` and rebuild with `docker compose build --no-cache sentinel-app`.

**Verify sentinel-app is reachable:**
```bash
docker compose run --rm evaluation curl -s http://sentinel-app:8000/health
```

## Code layout

- `dataset.py` — loads the grooming-conversation dataset's `labeled_conversations.csv`
  manifest, stages the 1-line split into bank-creation input folders, and
  derives ground-truth labels for the 10-line human-reviewed test set.
- `metrics.py` — precision/recall threshold sweep over Sentinel's
  `rare_class_affinity_score`, writing a CSV and a plot.
- `evaluate.py` — the evaluation script that ties the above together and
  calls the live Sentinel API.
- `tests/` — fixture-based unit tests against a small fake dataset; no live
  service required.

## The dataset

This evaluation needs the grooming-conversations dataset placed under
`sentinel/dataset/` (e.g. a clone of that dataset's own repo), containing a
`labeled_conversations.csv` manifest. The dataset is private and never
committed here — `sentinel/dataset/` is gitignored. If it lives somewhere
else, point `SENTINEL_DATASET_DIR` at it directly.

## Running the evaluation

```bash
# Smoke test: tiny subset, fast end-to-end sanity check
docker compose run --rm evaluation python sentinel/evaluate.py

# Full evaluation: whole dataset, results saved under $OUTPUT_DIR
docker compose run --rm -e EVAL_MODE=test evaluation python sentinel/evaluate.py
```

Results land under `$OUTPUT_DIR/sentinel_evaluation_results/<timestamp>/`
(defaults to `./results/...`) as `precision_recall_results.csv` and
`precision_recall_curve.png`.

## Running tests

`pytest` is dev-only and isn't in the `evaluation` image by default:

```bash
docker compose run --rm evaluation bash -c \
  "pip install -q -r sentinel/requirements-dev.txt && python -m pytest sentinel/tests/ -v"
```

## Env vars

- `SENTINEL_DATASET_DIR` — where the dataset lives (default: auto-discovered
  under `sentinel/dataset/`)
- `SENTINEL_DATA_DIR` — the shared bank-creation staging volume (see root
  README's "Sentinel Configuration"; defaults to `/data` in the `evaluation`
  container)
- `SENTINEL_CERTAINTY_THRESHOLD` — `lenient` (default) or `strict`; how the
  10-line test set's 5-point human-review scale collapses to a binary
  ground truth
- `SENTINEL_BANK_MODEL` — sentence-transformer model used for bank creation
  (default: `all-MiniLM-L6-v2`)
- `SENTINEL_CREATE_BANKS_TIMEOUT` / `SENTINEL_SCORE_TIMEOUT` — HTTP timeouts
  in seconds (defaults: `1800` / `60`)
- `SENTINEL_SCORE_TOP_K` / `SENTINEL_MIN_SCORE_TO_CONSIDER` — passed straight
  through to `POST /score`; unset by default, so the Sentinel API's own
  defaults apply (`top_k=5`, `min_score_to_consider=0.1`). See "Score
  parameter experiment" below before changing these
- `SENTINEL_DEFAULT_MIN_SIZE_OF_SCORES` — minimum number of per-message
  scores a conversation needs before the skewness aggregator returns
  anything other than `0.0`. Defaults to `5`, matching the Sentinel API's
  own default. Set on the `sentinel-app` service, not `evaluation` — unlike
  the two vars above, this isn't a `/score` request parameter
- `EVAL_MODE` — `smoke` (default) or `test`

## Score parameter experiment

A full-dataset run showed 46 of 89 grooming conversations scoring exactly
`0.0`, tied with 82 of 87 innocuous conversations also at `0.0` — no
threshold can separate those, capping recall at ~48% regardless of where you
set the decision boundary. We ran the full evaluation a few more times,
varying one scoring param at a time against the same banks, to see what's
actually causing it:

| `SENTINEL_SCORE_TOP_K` | `SENTINEL_MIN_SCORE_TO_CONSIDER` | max achievable recall | precision there |
| --- | --- | --- | --- |
| 5 (default) | 0.1 (default) | 48.3% | 89.6% |
| 5 (default) | 0.0 | 48.3% (no change at all) | 89.6% |
| 1 | 0.1 (default) | 82.0% | 85.9% |
| 2 | 0.1 (default) | 77.5% | 87.3% |
| 3 | 0.1 (default) | 65.2% | 89.2% |
| 20 | 0.1 (default) | 12.4% | 91.7% |

Two things fell out of this:

- `min_score_to_consider` isn't the cause — dropping it to `0.0` changed
  nothing. It filters individual bank-match scores, not the aggregated
  conversation score, so it can't be what's zeroing out whole conversations.
- `top_k` is the real lever, and lower is better here: averaging in more
  "top" bank matches dilutes the signal from the single closest match,
  which matters a lot when the bank entries are single lines and the test
  conversations are ten. `top_k=1` nearly doubles the recall ceiling
  (48% → 82%) for a small precision cost (90% → 86%).
