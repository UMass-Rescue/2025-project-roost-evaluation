"""Decode a video and choose candidate frames to hash.

Scene cuts are found with PySceneDetect (OpenCV backend). Each scene is then
sampled at most every `max_frame_gap_seconds`, taking the midpoint of each
slice, so scene-aware sampling and uniform sampling are the same code path:
with the detector set to "none" the whole video is a single scene.

This module imports nothing from the evaluation beyond hma.video.env, so it can move
into a package of its own later.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hma.video.env import env_value

SCENE_DETECTORS = ("content", "adaptive", "none")


class UnreadableVideoError(ValueError):
    """The file cannot be opened or has no usable frame rate or frame count."""

# Detector sensitivity used when none is given. The adaptive value is lower than
# PySceneDetect's own 3.0 because it found more of the cuts in real clips spliced at
# known positions (see docs/FRAME_SELECTION.md). Revisit with dissolves or another frame rate.
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
    candidate_window_seconds: how far either side of a slice's midpoint to also offer frames,
        so that a quality filter can pick the sharpest; 0 offers the midpoint only.
    """

    scene_detector: str = "adaptive"
    scene_threshold: Optional[float] = None
    min_scene_seconds: float = 1.0
    max_frame_gap_seconds: float = 1.0
    candidate_window_seconds: float = 0.25

    def __post_init__(self):
        if self.scene_detector not in SCENE_DETECTORS:
            raise ValueError(
                f"scene_detector must be one of {SCENE_DETECTORS}, got {self.scene_detector!r}"
            )
        if self.min_scene_seconds <= 0:
            raise ValueError("min_scene_seconds must be positive")
        if self.max_frame_gap_seconds <= 0:
            raise ValueError("max_frame_gap_seconds must be positive")
        if self.candidate_window_seconds < 0:
            raise ValueError("candidate_window_seconds must not be negative")

    @classmethod
    def from_env(cls) -> "SamplingConfig":
        """Read VIDEO_SCENE_DETECTOR, VIDEO_SCENE_THRESHOLD, VIDEO_MIN_SCENE_SECONDS,
        VIDEO_MAX_FRAME_GAP_SECONDS and VIDEO_CANDIDATE_WINDOW_SECONDS; unset or empty
        values keep the defaults."""
        defaults = cls()
        return cls(
            scene_detector=env_value("VIDEO_SCENE_DETECTOR", defaults.scene_detector, str).lower(),
            scene_threshold=env_value("VIDEO_SCENE_THRESHOLD", None),
            min_scene_seconds=env_value("VIDEO_MIN_SCENE_SECONDS", defaults.min_scene_seconds),
            max_frame_gap_seconds=env_value("VIDEO_MAX_FRAME_GAP_SECONDS", defaults.max_frame_gap_seconds),
            candidate_window_seconds=env_value("VIDEO_CANDIDATE_WINDOW_SECONDS", defaults.candidate_window_seconds),
        )


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int


@dataclass(frozen=True)
class SampledFrame:
    index: int  # frame number in the video
    seconds: float  # position in the video
    image: np.ndarray  # decoded frame, BGR as returned by OpenCV


def probe_video(path) -> VideoInfo:
    """Read the frame rate and frame count; raises UnreadableVideoError if the file cannot be used."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise UnreadableVideoError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps or math.isnan(fps) or fps <= 0:
        raise UnreadableVideoError(f"Cannot determine the frame rate of {path}")
    if frame_count <= 0:
        raise UnreadableVideoError(f"Cannot determine the frame count of {path}")
    return VideoInfo(fps=fps, frame_count=frame_count)


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


def _slices(
    scenes: Iterable[Tuple[int, int]], fps: float, max_frame_gap_seconds: float
) -> Iterator[Tuple[int, int, int]]:
    """(first_frame, end_frame, midpoint) of every slice: each scene is cut into the fewest
    equal slices no longer than max_frame_gap_seconds (never more than one per frame)."""
    for start, end in scenes:
        length = end - start
        if length <= 0:
            continue
        # round() absorbs float noise so an exact multiple (3.0 s at a 1.0 s gap) is not bumped up
        slices = max(1, math.ceil(round(length / fps / max_frame_gap_seconds, 9)))
        slices = min(slices, length)
        for i in range(slices):
            first = start + int(i * length / slices)
            last = start + int((i + 1) * length / slices)
            midpoint = min(start + int((i + 0.5) * length / slices), end - 1)
            yield first, max(last, first + 1), midpoint


def candidate_frame_groups(
    scenes: Iterable[Tuple[int, int]],
    fps: float,
    max_frame_gap_seconds: float,
    window_seconds: float,
) -> List[List[int]]:
    """
    The candidate frames of each slice: its midpoint and the frames within window_seconds
    either side, never leaving the slice. Groups are disjoint and in increasing order, so a
    quality filter can choose one frame per slice. With a window of 0 each group is the
    midpoint alone.
    """
    window = max(0, round(window_seconds * fps))
    groups = []
    for first, end, midpoint in _slices(scenes, fps, max_frame_gap_seconds):
        groups.append(list(range(max(first, midpoint - window), min(end - 1, midpoint + window) + 1)))
    return groups


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
            raise UnreadableVideoError(f"Cannot open video: {path}")
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


def iter_frame_groups(path, groups: Sequence[Sequence[int]], fps: float) -> Iterator[List[SampledFrame]]:
    """
    Decode the frames of each group, one group at a time. `groups` must be disjoint and in
    increasing order, as candidate_frame_groups returns them. A group none of whose frames
    could be decoded (the file ended early) is skipped.
    """
    frames = iter_frames(path, [index for group in groups for index in group], fps)
    pending = next(frames, None)
    for group in groups:
        wanted = set(group)
        decoded = []
        while pending is not None and pending.index in wanted:
            decoded.append(pending)
            pending = next(frames, None)
        if decoded:
            yield decoded


def sample_frame_groups(path, config: Optional[SamplingConfig] = None) -> Iterator[List[SampledFrame]]:
    """Candidate frames grouped by slice: probe, detect scenes, choose groups, decode them."""
    config = config or SamplingConfig.from_env()
    path = Path(path)
    info = probe_video(path)
    scenes = detect_scenes(path, info, config)
    groups = candidate_frame_groups(
        scenes, info.fps, config.max_frame_gap_seconds, config.candidate_window_seconds
    )
    return iter_frame_groups(path, groups, info.fps)
