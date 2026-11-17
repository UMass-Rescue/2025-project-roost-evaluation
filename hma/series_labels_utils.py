"""Find media files and series labels in a directory of series folders.

Shared by the image and video evaluations, which differ only in the file types and where
the labels file lives.
"""

import json
from pathlib import Path
from typing import Sequence

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _is_media(path: Path, extensions: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def _has_media_subdirs(media_dir: Path, extensions: Sequence[str]) -> bool:
    for child in media_dir.iterdir():
        if not child.is_dir():
            continue
        if any(_is_media(p, extensions) for p in child.rglob("*")):
            return True
    return False


def get_media_files(media_dir: Path, extensions: Sequence[str]) -> list[str]:
    """Return sorted list of file paths with one of the extensions (auto-recursive if subdirs exist)."""
    recursive = _has_media_subdirs(media_dir, extensions)
    if recursive:
        files = [p for p in media_dir.rglob("*") if _is_media(p, extensions)]
    else:
        files = [p for p in media_dir.iterdir() if _is_media(p, extensions)]
    return sorted({str(p) for p in files})


def _format_label_path(path: Path, label_root: Path | None) -> str:
    if label_root is None:
        return path.as_posix()
    try:
        rel = path.relative_to(label_root).as_posix()
        return f"./{rel}"
    except ValueError:
        return path.as_posix()


def find_series(
    series_root: Path,
    extensions: Sequence[str],
    label_root: Path | None = None,
    min_size: int = 2,
) -> dict[str, list[str]]:
    """Series name -> file paths, one series per subfolder of series_root with at least min_size files.

    Paths are written relative to label_root when they are inside it, so a labels file made on one
    machine matches the paths another machine finds. Without a label_root they are written as found.
    """
    labels: dict[str, list[str]] = {}
    for series_dir in sorted(p for p in series_root.iterdir() if p.is_dir()):
        files = sorted(p for p in series_dir.rglob("*") if _is_media(p, extensions))
        if len(files) < min_size:
            continue
        labels[series_dir.name] = [_format_label_path(p, label_root) for p in files]
    if not labels:
        raise ValueError(f"No series with ≥{min_size} files found under {series_root}")
    return labels


def ensure_labels_file(
    labels_path: Path,
    media_dir: Path,
    extensions: Sequence[str],
    default_labels_path: Path,
    label_root: Path,
) -> Path:
    """Return an existing labels file, building default_labels_path from the series folders if none exists."""
    if default_labels_path.exists() or labels_path.exists():
        return labels_path if labels_path.exists() else default_labels_path

    # Generate labels only if missing and series subfolders exist
    if _has_media_subdirs(media_dir, extensions):
        labels = find_series(media_dir, extensions, label_root)
        default_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(default_labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        return default_labels_path

    raise ValueError(
        "Series labels file missing and input files appear flat. "
        "Provide LABELS_PATH or use a directory of series subfolders."
    )
