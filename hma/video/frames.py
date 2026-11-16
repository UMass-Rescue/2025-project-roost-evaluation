"""Decode a video and choose candidate frames to hash.

Scene cuts are found with PySceneDetect (OpenCV backend). Each scene is then
sampled at most every `max_frame_gap_seconds`, taking the midpoint of each
slice, so scene-aware sampling and uniform sampling are the same code path:
with the detector set to "none" the whole video is a single scene.

This module is self-contained (no evaluation imports) so it can move into a
package of its own later.
"""
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

SCENE_DETECTORS = ("content", "adaptive", "none")

# Detector sensitivity used when none is given. The adaptive value is lower than
# PySceneDetect's own 3.0: on real clips spliced together at known cut positions
# (6 fps, hard cuts) 2.0 found 95% of the cuts at 95% precision, while 3.0 found 73%
# and "content" at its default 27.0 found 83% at 80% precision. Revisit with footage
# that has dissolves or a different frame rate.
DEFAULT_SCENE_THRESHOLDS = {"content": 27.0, "adaptive": 2.0}


@dataclass(frozen=True)
class SamplingConfig:
    """
    How candidate frames are chosen.

    scene_detector: "adaptive" compares the colour/brightness change between frames
        with its neighbours (fewer false cuts on fast motion), "content" flags a cut
        when that change is large on its own, "none" skips detection (uniform sampling).
    scene_threshold: detector sensitivity (lower finds more cuts); None uses
        DEFAULT_SCENE_THRESHOLDS for the detector (2.0 for "adaptive", 27.0 for "content").
    min_scene_seconds: shortest allowed scene. PySceneDetect counts this in frames,
        so it is given in seconds here to behave the same at any frame rate.
    max_frame_gap_seconds: longest gap between sampled frames inside a scene.
    """

    scene_detector: str = "adaptive"
    scene_threshold: Optional[float] = None
    min_scene_seconds: float = 1.0
    max_frame_gap_seconds: float = 1.0

    def __post_init__(self):
        if self.scene_detector not in SCENE_DETECTORS:
            raise ValueError(
                f"scene_detector must be one of {SCENE_DETECTORS}, got {self.scene_detector!r}"
            )
        if self.min_scene_seconds <= 0:
            raise ValueError("min_scene_seconds must be positive")
        if self.max_frame_gap_seconds <= 0:
            raise ValueError("max_frame_gap_seconds must be positive")

    @classmethod
    def from_env(cls) -> "SamplingConfig":
        """Read VIDEO_SCENE_DETECTOR, VIDEO_SCENE_THRESHOLD, VIDEO_MIN_SCENE_SECONDS
        and VIDEO_MAX_FRAME_GAP_SECONDS; unset or empty values keep the defaults."""
        defaults = cls()

        def env(name):
            value = os.getenv(name, "").strip()
            return value or None

        threshold = env("VIDEO_SCENE_THRESHOLD")
        return cls(
            scene_detector=(env("VIDEO_SCENE_DETECTOR") or defaults.scene_detector).lower(),
            scene_threshold=float(threshold) if threshold is not None else None,
            min_scene_seconds=float(env("VIDEO_MIN_SCENE_SECONDS") or defaults.min_scene_seconds),
            max_frame_gap_seconds=float(
                env("VIDEO_MAX_FRAME_GAP_SECONDS") or defaults.max_frame_gap_seconds
            ),
        )


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class SampledFrame:
    index: int  # frame number in the video
    seconds: float  # position in the video
    image: np.ndarray  # decoded frame, BGR as returned by OpenCV


def probe_video(path) -> VideoInfo:
    """Read the frame rate, frame count and size; raises ValueError if the file cannot be used."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not fps or math.isnan(fps) or fps <= 0:
        raise ValueError(f"Cannot determine the frame rate of {path}")
    if frame_count <= 0:
        raise ValueError(f"Cannot determine the frame count of {path}")
    return VideoInfo(fps=fps, frame_count=frame_count, width=width, height=height)


def detect_scenes(path, info: VideoInfo, config: SamplingConfig) -> List[Tuple[int, int]]:
    """Scenes as (start_frame, end_frame) ranges, end exclusive, covering the video.

    A video with no detected cut (or with detection off) is one scene."""
    whole_video = [(0, info.frame_count)]
    if config.scene_detector == "none":
        return whole_video

    # imported here so that detector "none" works without PySceneDetect installed
    from scenedetect import AdaptiveDetector, ContentDetector, detect

    threshold = config.scene_threshold
    if threshold is None:
        threshold = DEFAULT_SCENE_THRESHOLDS[config.scene_detector]
    min_scene_len = max(1, math.ceil(config.min_scene_seconds * info.fps))
    if config.scene_detector == "content":
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    else:
        detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)

    scenes = detect(str(path), detector, backend="opencv")
    if not scenes:
        return whole_video
    return [
        (start.frame_num, min(end.frame_num, info.frame_count)) for start, end in scenes
    ]


def candidate_frame_indices(
    scenes: Iterable[Tuple[int, int]], fps: float, max_frame_gap_seconds: float
) -> List[int]:
    """
    Frame numbers to consider, in increasing order.

    Each scene is cut into the fewest equal slices no longer than max_frame_gap_seconds
    and the middle frame of every slice is taken, so a scene shorter than the gap gets
    one frame at its midpoint. Midpoints also keep clear of the blur and fades that tend
    to sit right at a cut.
    """
    indices = set()
    for start, end in scenes:
        length = end - start
        if length <= 0:
            continue
        # round() absorbs float noise so an exact multiple (3.0 s at a 1.0 s gap) is not bumped up
        slices = max(1, math.ceil(round(length / fps / max_frame_gap_seconds, 9)))
        for i in range(slices):
            indices.add(min(start + int((i + 0.5) * length / slices), end - 1))
    return sorted(indices)


def iter_frames(path, indices: Iterable[int], fps: float) -> Iterator[SampledFrame]:
    """
    Decode the frames at `indices`, one at a time, in increasing order.

    Frames are yielded lazily so a long video is never held in memory. If the file
    ends earlier than its reported frame count, decoding stops and the missing
    frames are simply not yielded.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        position = 0
        for target in sorted(set(indices)):
            while position < target:
                if not cap.grab():
                    return
                position += 1
            ok, image = cap.read()
            if not ok:
                return
            position += 1
            yield SampledFrame(index=target, seconds=target / fps, image=image)
    finally:
        cap.release()


def sample_frames(path, config: Optional[SamplingConfig] = None) -> Iterator[SampledFrame]:
    """Candidate frames of a video: probe, detect scenes, choose indices, decode them."""
    config = config or SamplingConfig.from_env()
    path = Path(path)
    info = probe_video(path)
    scenes = detect_scenes(path, info, config)
    indices = candidate_frame_indices(scenes, info.fps, config.max_frame_gap_seconds)
    return iter_frames(path, indices, info.fps)
