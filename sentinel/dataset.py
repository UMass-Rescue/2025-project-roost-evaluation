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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MANIFEST_FILENAME = "labeled_conversations.csv"
SENTINEL_DATA_DIR_ENV = "SENTINEL_DATA_DIR"
SENTINEL_CERTAINTY_THRESHOLD_ENV = "SENTINEL_CERTAINTY_THRESHOLD"
DEFAULT_CERTAINTY_THRESHOLD = "lenient"

# Collapses the 5-point human_classification scale into a binary ground
# truth label. "too difficult to tell" is always excluded, regardless of
# threshold.
_CERTAINTY_LEVELS = {
    "lenient": {
        "grooming": {"definitely grooming", "probably grooming"},
        "innocuous": {"definitely innocuous", "probably innocuous"},
    },
    "strict": {
        "grooming": {"definitely grooming"},
        "innocuous": {"definitely innocuous"},
    },
}


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


def _data_dir(data_dir: Optional[Path] = None) -> Path:
    """Resolve the shared staging directory used for bank creation.

    This is a Docker volume mounted at the same path into both this
    container and sentinel-app -- see the root README.md's 'Sentinel
    Configuration' section -- so paths written here resolve identically on
    both sides. Required because Sentinel's POST /banks/create takes
    filesystem folder paths, not uploaded content.
    """
    if data_dir is not None:
        path = data_dir
    else:
        override = os.environ.get(SENTINEL_DATA_DIR_ENV)
        if not override:
            raise DatasetNotFoundError(
                f"{SENTINEL_DATA_DIR_ENV} is not set. This must point at the "
                f"shared volume mounted into both this container and "
                f"sentinel-app -- see the root README.md's 'Sentinel "
                f"Configuration' section."
            )
        path = Path(override)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _restage_folder(folder: Path, source_paths: List[Path]) -> None:
    """Replace folder's contents with copies of source_paths."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for source_path in source_paths:
        shutil.copy2(source_path, folder / source_path.name)


def stage_1l_banks(
    data_dir: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Copy the 1-line split into positive/negative folders for bank creation.

    Materializes <data_dir>/texts/positive (confirmed-grooming files) and
    <data_dir>/texts/negative (all innocuous files) by copying from the
    dataset, clearing any previous contents first. data_dir defaults to the
    shared staging volume (SENTINEL_DATA_DIR).

    limit, if given, caps how many files are copied into each folder (e.g.
    for a fast smoke test against a tiny subset instead of the full split).

    Returns (positive_dir, negative_dir). Pass these same paths straight
    into a CreateBanksRequest's positive_folder/negative_folder -- they
    resolve identically inside sentinel-app, which shares the same volume.
    """
    root = _data_dir(data_dir)
    positive_dir = root / "texts" / "positive"
    negative_dir = root / "texts" / "negative"

    grooming_paths, innocuous_paths = load_1l_paths(dataset_root=dataset_root)
    if limit is not None:
        grooming_paths = grooming_paths[:limit]
        innocuous_paths = innocuous_paths[:limit]
    _restage_folder(positive_dir, grooming_paths)
    _restage_folder(negative_dir, innocuous_paths)

    return positive_dir, negative_dir


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


def ground_truth_label(
    conversation: Conversation, certainty_threshold: Optional[str] = None
) -> Optional[str]:
    """Collapse a Conversation's human_classification into a ground-truth label.

    Returns "grooming", "innocuous", or None if this conversation should be
    excluded from evaluation at the given certainty threshold (always true
    for "too difficult to tell"; also true for "probably X" rows under
    "strict").

    Based on human_classification -- the actual human review -- not
    generation_label, since the two disagree often enough in this dataset
    that generation_label can't be treated as ground truth here.

    certainty_threshold defaults to the SENTINEL_CERTAINTY_THRESHOLD env
    var, falling back to "lenient" ("definitely X"/"probably X" both
    count). Pass "strict" to keep only "definitely X" rows.
    """
    threshold = certainty_threshold or os.environ.get(
        SENTINEL_CERTAINTY_THRESHOLD_ENV, DEFAULT_CERTAINTY_THRESHOLD
    )
    try:
        levels = _CERTAINTY_LEVELS[threshold]
    except KeyError:
        raise ValueError(
            f"Unknown certainty threshold {threshold!r}. Expected one of: "
            f"{sorted(_CERTAINTY_LEVELS)}."
        )

    classification = conversation.human_classification.strip().lower()
    if classification in levels["grooming"]:
        return "grooming"
    if classification in levels["innocuous"]:
        return "innocuous"
    return None


def load_10l_eval_set(
    dataset_root: Optional[Path] = None, certainty_threshold: Optional[str] = None
) -> List[Tuple[Conversation, str]]:
    """Load the 10-line test set paired with ground-truth labels.

    Convenience wrapper around load_10l_test_set() + ground_truth_label():
    excludes conversations with no clear ground truth at the given
    certainty threshold.
    """
    conversations = load_10l_test_set(dataset_root=dataset_root)
    paired = []
    for conversation in conversations:
        label = ground_truth_label(conversation, certainty_threshold=certainty_threshold)
        if label is not None:
            paired.append((conversation, label))
    return paired
