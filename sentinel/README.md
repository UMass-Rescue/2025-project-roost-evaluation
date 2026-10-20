# Sentinel (grooming classifier) evaluation

Sentinel's service code lives in its own repo, [UMass-Rescue/sentinel-api](https://github.com/UMass-Rescue/sentinel-api),
and isn't vendored here. `docker-compose.yaml` builds the `sentinel-app` service
directly from a pinned commit on that repo — see the root `README.md`'s
"Sentinel Configuration" section for details.

This directory is for Sentinel-specific evaluation code (analogous to the
root `evaluate.py` / `metrics/` / `resources/` / `tests/` used for HMA).
