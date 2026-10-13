# Patches

Patches applied to upstream ThreatExchange to add custom functionality.

## Creating a Patch

The commit hash should be from your fork's `origin/main` branch synced with upstream ThreatExchange.

1. Generate patch from your fork:
```bash
git diff origin/main origin/development > hma.patch
```

2. Copy to `resources/patches/hma.patch`

3. Update the commit hash at THREATEXCHANGE_COMMIT in `Dockerfile.hma`

4. Update the commit hash at THREATEXCHANGE_COMMIT in `docker-compose.yaml` 

## Usage

Patches are applied automatically during `docker-compose build`.

If you update the patch or commit hash, rebuild the container with `docker-compose build --no-cache`.
