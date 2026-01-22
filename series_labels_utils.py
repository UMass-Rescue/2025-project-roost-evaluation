import json
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_LABELS_FILENAME = "images_series_labels.json"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _has_image_subdirs(image_dir: Path) -> bool:
    for child in image_dir.iterdir():
        if not child.is_dir():
            continue
        if any(_is_image(p) for p in child.rglob("*")):
            return True
    return False


def get_image_files(image_dir: Path) -> list[str]:
    """Return sorted list of image file paths (auto-recursive if subdirs exist)."""
    recursive = _has_image_subdirs(image_dir)
    if recursive:
        images = [p for p in image_dir.rglob("*") if _is_image(p)]
    else:
        images = [p for p in image_dir.iterdir() if _is_image(p)]
    return sorted({str(p) for p in images})


def _format_label_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.relative_to(repo_root).as_posix()
        return f"./{rel}"
    except ValueError:
        return path.as_posix()


def _build_labels_from_series_dirs(
    series_root: Path,
    labels_path: Path,
    min_size: int = 2,
) -> None:
    repo_root = Path(__file__).resolve().parent
    series_dirs = sorted([p for p in series_root.iterdir() if p.is_dir()])
    labels: dict[str, list[str]] = {}
    for series_dir in series_dirs:
        images = sorted([p for p in series_dir.rglob("*") if _is_image(p)])
        if len(images) < min_size:
            continue
        labels[series_dir.name] = [_format_label_path(p, repo_root) for p in images]
    if not labels:
        raise ValueError(f"No series with ≥{min_size} images found under {series_root}")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)


def ensure_labels_file(labels_path: Path, image_dir: Path) -> Path:
    if _has_image_subdirs(image_dir):
        repo_root = Path(__file__).resolve().parent
        default_labels_path = repo_root / "resources/labels" / DEFAULT_LABELS_FILENAME
        _build_labels_from_series_dirs(image_dir, default_labels_path)
        return default_labels_path
    if labels_path.exists():
        return labels_path
    raise ValueError(
        "Series labels file missing and input images appear flat. "
        "Provide LABELS_PATH or use a directory of series subfolders."
    )
