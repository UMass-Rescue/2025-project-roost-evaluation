"""Video series labels: the shared logic in hma/series_labels_utils.py with the video file types."""

from pathlib import Path

from hma import series_labels_utils as shared


def get_video_files(video_dir: Path) -> list[str]:
    """Return sorted list of video file paths (auto-recursive if subdirs exist)."""
    return shared.get_media_files(video_dir, shared.VIDEO_EXTENSIONS)


def find_video_series(video_dir: Path) -> dict[str, list[str]]:
    """Series name -> video paths as get_video_files finds them, one series per subfolder with at least two videos."""
    return shared.find_series(video_dir, shared.VIDEO_EXTENSIONS)
