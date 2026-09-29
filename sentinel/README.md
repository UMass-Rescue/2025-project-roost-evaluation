# Sentinel (grooming classifier) evaluation

This directory holds **evaluation-side** files for Sentinel, mirroring the root-level
convention already used for HMA (`evaluate.py`, `metrics/`, `resources/`, `tests/`).

## Where the actual Sentinel service lives

The Sentinel API service itself is **not** vendored in this repo. It's a standalone
repo, [UMass-Rescue/sentinel-api](https://github.com/UMass-Rescue/sentinel-api), and is
built directly from there via `docker-compose.yaml`'s `sentinel-app` service, which uses
a git-context build pinned to a specific commit — the same philosophy as the `hma-app`
service pinning `THREATEXCHANGE_COMMIT` from `Dockerfile.hma`.

```yaml
  sentinel-app:
    build:
      context: https://github.com/UMass-Rescue/sentinel-api.git#<pinned-commit>
    ports:
      - "8001:8000"
```

See the root `README.md`'s "Sentinel Configuration" section for the currently pinned
commit and how to bump it.

`sentinel-api` exposes `/score`, `/health`, and `/banks/*` over HTTP — this repo's
evaluation code (once added) will call it the same way `evaluate.py`/`tests/` call HMA
over HTTP, via `SENTINEL_HOST`/`SENTINEL_PORT` (wired into the `evaluation` service in
`docker-compose.yaml`).
