"""
Loader for Sentinel's grooming-conversation dataset.

The dataset itself is never vendored into this repo (see .gitignore) -- it's
private, access-controlled data that must be cloned/placed under
sentinel/dataset/ separately. See sentinel/README.md.

This module only ever reads a dataset's labeled_conversations.csv (produced
by that dataset's own preprocessing script) plus the files it references.
It never parses raw annotation CSVs directly -- that reconciliation logic
belongs in the dataset's own repo, not here.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MANIFEST_FILENAME = "labeled_conversations.csv"


class DatasetNotFoundError(RuntimeError):
    """Raised when the dataset directory or its labeled_conversations.csv can't be found."""


def _dataset_root() -> Path:
    """Resolve the dataset root directory.

    Defaults to the single subdirectory of sentinel/dataset/ that contains a
    labeled_conversations.csv, overridable via the SENTINEL_DATASET_DIR env
    var (pointed directly at the dataset root, not its parent).
    """
    override = os.environ.get("SENTINEL_DATASET_DIR")
    if override:
        root = Path(override)
        if not root.exists():
            raise DatasetNotFoundError(
                f"SENTINEL_DATASET_DIR is set to {root}, but it doesn't exist."
            )
        return root

    base = Path(__file__).resolve().parent / "dataset"
    if not base.exists():
        raise DatasetNotFoundError(
            f"No dataset found at {base}. This dataset is private and not shipped "
            f"in this repo -- clone it there yourself, or set SENTINEL_DATASET_DIR "
            f"to point at it directly. See sentinel/README.md."
        )

    candidates = [
        p for p in sorted(base.iterdir())
        if p.is_dir() and (p / MANIFEST_FILENAME).exists()
    ]
    if not candidates:
        raise DatasetNotFoundError(
            f"No subdirectory of {base} contains a {MANIFEST_FILENAME}. "
            f"Run that dataset's preprocessing script first, or set "
            f"SENTINEL_DATASET_DIR to point at the dataset root directly. "
            f"See sentinel/README.md."
        )
    if len(candidates) > 1:
        raise DatasetNotFoundError(
            f"Multiple dataset candidates found under {base}: "
            f"{[str(c) for c in candidates]}. Set SENTINEL_DATASET_DIR to disambiguate."
        )
    return candidates[0]


@dataclass(frozen=True)
class Conversation:
    """A single labeled conversation from the 10-line (eval/test) split."""

    path: Path
    split: str  # always "10l" for Conversation instances
    generation_label: str  # "grooming" or "innocuous"
    human_classification: str  # raw 5-point text, e.g. "probably grooming"

    def lines(self) -> List[str]:
        """Read the conversation file and return its non-empty lines."""
        if not self.path.exists():
            raise DatasetNotFoundError(f"Conversation file not found: {self.path}")
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        return [line for line in text.splitlines() if line.strip()]


def _read_manifest(dataset_root: Path) -> List[dict]:
    manifest_path = dataset_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise DatasetNotFoundError(f"{MANIFEST_FILENAME} not found at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_1l_paths(dataset_root: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:
    """Load the 1-line split as (grooming_paths, innocuous_paths).

    These are file paths only, intended to be handed to Sentinel's
    POST /banks/create endpoint (positive_folder/negative_folder) as a
    folder of files -- not read into memory here. grooming_paths is the
    small, human-confirmed subset; innocuous_paths is the full
    generation-trusted set. See sentinel/README.md and the dataset's own
    README for why these are asymmetric.
    """
    root = dataset_root or _dataset_root()
    rows = [r for r in _read_manifest(root) if r["split"] == "1l"]

    grooming = [root / r["relative_path"] for r in rows if r["generation_label"] == "grooming"]
    innocuous = [root / r["relative_path"] for r in rows if r["generation_label"] == "innocuous"]
    return grooming, innocuous


def load_10l_test_set(dataset_root: Optional[Path] = None) -> List[Conversation]:
    """Load the 10-line split as a list of human-reviewed test Conversations."""
    root = dataset_root or _dataset_root()
    rows = [r for r in _read_manifest(root) if r["split"] == "10l"]

    return [
        Conversation(
            path=root / r["relative_path"],
            split="10l",
            generation_label=r["generation_label"],
            human_classification=r["human_classification"],
        )
        for r in rows
    ]
