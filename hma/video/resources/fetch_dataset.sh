#!/usr/bin/env bash
# Fetch the video series dataset at a pinned commit, for the video evaluation.
#
#   hma/video/resources/fetch_dataset.sh [destination]
#
# The default destination is hma/video/resources/video-series-dataset (ignored by git), and the
# evaluation reads its series/ folder. Running it again is harmless: a destination already at the
# pinned commit is left alone. Anything else already at the destination is never touched, so move it
# aside first. The download is about 550 MB.
set -euo pipefail

REPO="https://github.com/UMass-Rescue/video-series-dataset.git"
COMMIT="b5e12e253c863d85c0891eb8d87a9476dbf675ff"
DEST="${1:-$(cd "$(dirname "$0")" && pwd)/video-series-dataset}"

if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" ]; then
  echo "Dataset already at $COMMIT in $DEST"
  exit 0
fi
if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST")" ]; then
  echo "$DEST exists and is not the pinned dataset checkout; move it aside and run again." >&2
  exit 1
fi

set -x
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin "$REPO"
git -C "$DEST" fetch -q --depth 1 origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
set +x
echo "Dataset ready: $DEST/series"
