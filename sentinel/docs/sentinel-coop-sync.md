# Syncing `sentinel/api` from `coop-public`

`sentinel/api` in this repo is not hand-copied ‚Äî it's pulled in via `git subtree`
from the `server/sentinel-api/` directory of the
[`coop-public`](https://github.com/UMass-Rescue/coop-public) repo (branch
`sentinel/api-docker-service`). This keeps full commit history and lets changes
flow in either direction later without patch files or manual copy-pasting.

Do not hand-edit history here to "fix" this ‚Äî just re-run the sync commands
below. Do not rename or move `sentinel/api` without updating the `--prefix`
in every command below to match.

## One-time setup (already done if this doc exists in the repo)

```bash
git remote add sentinel-source /Users/sahilsharma/projects/forks/coop-public
git fetch sentinel-source sentinel-api-export
git subtree add --prefix=sentinel/api sentinel-source sentinel-api-export
```

Notes:

- `sentinel-api-export` is a synthetic branch that only exists in `coop-public`.
  It's produced by `git subtree split --prefix=server/sentinel-api` there ‚Äî it
  is not a "real" branch you'll see in the coop-public PR list.
- If `coop-public` is not available as a local filesystem path (e.g. you're
  syncing from a teammate's machine or CI), replace the remote URL with the
  repo's Git URL instead, e.g.:
  `git remote add sentinel-source https://github.com/UMass-Rescue/coop-public.git`
- `git subtree add` requires that `sentinel/api` **does not already exist** in
  this repo at the time you run it ‚Äî it creates that directory as part of the
  merge commit. `sentinel/` itself can already have other files in it.

## Pulling new changes from `coop-public` ‚Üí this repo

Whenever `server/sentinel-api/` changes upstream (in `coop-public`, on the
`sentinel/api-docker-service` branch or wherever it lives after merging), pull
the update over:

```bash
# 1. In coop-public: re-cut the export branch (--rejoin makes this incremental)
cd /path/to/coop-public
git checkout sentinel/api-docker-service   # or main, once merged
git pull
git subtree split --prefix=server/sentinel-api -b sentinel-api-export --rejoin

# 2. Back in this repo: fetch and merge the update
cd /path/to/this-repo
git fetch sentinel-source sentinel-api-export
git subtree pull --prefix=sentinel/api sentinel-source sentinel-api-export
```

`git subtree pull` will open a merge commit; resolve any conflicts under
`sentinel/api/` normally and commit.

## Pushing changes made here ‚Üí back to `coop-public`

If you make fixes to `sentinel/api` in this repo and want them to flow back
upstream:

```bash
cd /path/to/this-repo
git subtree push --prefix=sentinel/api sentinel-source my-fix-branch-name
```

That creates/updates `my-fix-branch-name` in `coop-public` with just the
`sentinel/api` history rewritten back onto `server/sentinel-api/`. Open a PR
against `coop-public` from that branch as usual.

## Quick reference

| Direction | Command |
|---|---|
| coop-public ‚Üí here (first time) | `git subtree add --prefix=sentinel/api sentinel-source sentinel-api-export` |
| coop-public ‚Üí here (updates) | `git subtree pull --prefix=sentinel/api sentinel-source sentinel-api-export` |
| here ‚Üí coop-public | `git subtree push --prefix=sentinel/api sentinel-source <branch-name>` |

## Troubleshooting

- **"prefix already exists"** on `subtree add` ‚Äî remove/rename whatever is
  currently at `sentinel/api` first; subtree needs to create that path itself.
- **Merge conflicts on `subtree pull`** ‚Äî resolve like any normal merge
  conflict; the conflicting files will be under `sentinel/api/`.
- **Split is slow** ‚Äî only the first `git subtree split` walks full history;
  subsequent ones with `--rejoin` are incremental and much faster.
- **Lost track of the remote** ‚Äî re-run
  `git remote add sentinel-source <path-or-url>` (or `git remote set-url` if
  it already exists but points somewhere stale).
