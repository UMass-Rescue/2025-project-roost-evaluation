"""Score frames for low information and pick the best candidate of each slice.

A frame is "junk" when it is flat (black, white or a solid colour), a set of vertical
bars (colour bars, test patterns) or blurry. Everything is measured on a small grayscale
copy of the frame, so the checks are cheap and do not depend on the video's resolution.

Blur is handled by choosing, rather than rejecting: for every slice of a video the
sampler offers a few neighbouring frames, and the sharpest one that is not junk is kept.
A slice with no usable frame is dropped. A video in which every slice is dropped yields no
frames at all, and the report says so, so that a caller can tell the user.

This module is self-contained (apart from the frame type) so it can move into a package
of its own later.
"""
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hma.video.frames import SampledFrame

FLAT = "flat"
BARS = "bars"
BLURRY = "blurry"


@dataclass(frozen=True)
class QualityConfig:
    """
    Thresholds, all measured on a grayscale copy `analysis_width` pixels wide (0-255 scale).
    256 is close to the 224 pixels CLIP looks at, so blur too fine to survive that
    resizing is not penalised. How the defaults were chosen is in hma/video/README.md.

    flat_std_max: a frame whose brightness varies less than this is flat.
    bars_row_diff_max: a frame whose rows are nearly identical (mean absolute difference
        between adjacent rows below this) is vertical bars. Real footage has some
        vertical change almost everywhere.
    blur_min: a frame whose Laplacian variance (edge sharpness) is below this is blurry.
    """

    analysis_width: int = 256
    flat_std_max: float = 5.0
    bars_row_diff_max: float = 0.7
    blur_min: float = 20.0

    def __post_init__(self):
        if self.analysis_width < 8:
            raise ValueError("analysis_width must be at least 8")
        for name in ("flat_std_max", "bars_row_diff_max", "blur_min"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    @classmethod
    def from_env(cls) -> "QualityConfig":
        """Read VIDEO_QUALITY_ANALYSIS_WIDTH, VIDEO_FLAT_STD_MAX, VIDEO_BARS_ROW_DIFF_MAX and
        VIDEO_BLUR_MIN; unset or empty values keep the defaults."""
        defaults = cls()

        def env(name, default, kind=float):
            value = os.getenv(name, "").strip()
            return kind(value) if value else default

        return cls(
            analysis_width=env("VIDEO_QUALITY_ANALYSIS_WIDTH", defaults.analysis_width, int),
            flat_std_max=env("VIDEO_FLAT_STD_MAX", defaults.flat_std_max),
            bars_row_diff_max=env("VIDEO_BARS_ROW_DIFF_MAX", defaults.bars_row_diff_max),
            blur_min=env("VIDEO_BLUR_MIN", defaults.blur_min),
        )


@dataclass(frozen=True)
class FrameQuality:
    luma_std: float  # spread of brightness; ~0 for black, white or a solid colour
    row_diff: float  # mean absolute difference between adjacent rows; ~0 for vertical bars
    sharpness: float  # variance of the Laplacian; low for blur and for flat frames
    reasons: Tuple[str, ...]  # why the frame is junk, empty when it is usable

    @property
    def junk(self) -> bool:
        return bool(self.reasons)


def measure_frame(image: np.ndarray, analysis_width: int = 256) -> Tuple[float, float, float]:
    """(luma_std, row_diff, sharpness) of a BGR frame, measured on a small grayscale copy."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width != analysis_width:
        new_height = max(2, round(height * analysis_width / width))
        gray = cv2.resize(gray, (analysis_width, new_height), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)
    luma_std = float(gray.std())
    row_diff = float(np.abs(np.diff(gray, axis=0)).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return luma_std, row_diff, sharpness


def assess_frame(image: np.ndarray, config: Optional[QualityConfig] = None) -> FrameQuality:
    """Measure a frame and say why it is junk, if it is."""
    config = config or QualityConfig()
    luma_std, row_diff, sharpness = measure_frame(image, config.analysis_width)
    reasons = []
    if luma_std < config.flat_std_max:
        reasons.append(FLAT)
    elif row_diff < config.bars_row_diff_max:
        reasons.append(BARS)
    if sharpness < config.blur_min and FLAT not in reasons:
        reasons.append(BLURRY)
    return FrameQuality(luma_std, row_diff, sharpness, tuple(reasons))


@dataclass(frozen=True)
class SelectedFrame:
    frame: SampledFrame
    quality: FrameQuality


@dataclass
class QualityReport:
    """What the selection did, so a run can say how many frames it threw away and why."""

    slices: int = 0
    frames_assessed: int = 0
    junk_frames: Counter = field(default_factory=Counter)  # reason -> frames with that reason
    slices_dropped: int = 0  # slices where every candidate was junk

    @property
    def no_usable_frames(self) -> bool:
        """Every slice was junk, so the video has nothing to index."""
        return self.slices > 0 and self.slices_dropped == self.slices

    def summary(self) -> str:
        reasons = ", ".join(f"{n} {r}" for r, n in sorted(self.junk_frames.items())) or "none"
        text = (
            f"{self.slices} slices, {self.frames_assessed} frames assessed, junk: {reasons}; "
            f"{self.slices_dropped} slices dropped"
        )
        return text + "; NO USABLE FRAMES" if self.no_usable_frames else text


def iter_selected_frames(
    groups: Iterable[Sequence[SampledFrame]],
    config: Optional[QualityConfig] = None,
    report: Optional[QualityReport] = None,
) -> Iterator[SelectedFrame]:
    """
    Yield the sharpest usable frame of each group (the neighbouring candidates of one slice).

    A group with only junk frames is dropped and counted in `report`, which is filled in as
    the groups are consumed. Frames are handled one group at a time, so a long video can be
    streamed through without holding its frames in memory.
    """
    config = config or QualityConfig()
    report = report if report is not None else QualityReport()

    for group in groups:
        report.slices += 1
        best: Optional[SelectedFrame] = None
        for frame in group:
            quality = assess_frame(frame.image, config)
            report.frames_assessed += 1
            report.junk_frames.update(quality.reasons)
            if not quality.junk and (best is None or quality.sharpness > best.quality.sharpness):
                best = SelectedFrame(frame, quality)
        if best is None:
            report.slices_dropped += 1
        else:
            yield best


def select_frames(
    groups: Iterable[Sequence[SampledFrame]], config: Optional[QualityConfig] = None
) -> Tuple[List[SelectedFrame], QualityReport]:
    """iter_selected_frames, collected into a list together with its report."""
    report = QualityReport()
    return list(iter_selected_frames(groups, config, report)), report
