"""From a video file to the frames that represent it.

Runs the steps in order and streams through the video, so a long one is never held in
memory: sample candidate frames per slice (frames.py), keep the sharpest usable frame of each
slice (quality.py), drop frames that repeat an earlier one by thumbnail and then by embedding
(dedupe.py), and finally thin the result evenly if a cap is set.

Embeddings are computed by a function the caller passes in, so this module does not depend on
how they are made (the evaluation gets them from HMA's CLIP signal). The function receives
JPEG bytes, in the same encoding the frames would be uploaded in, and returns one vector per
frame.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from hma.video.dedupe import DedupeConfig, FrameDeduper, thin_evenly, thumbnail
from hma.video.frames import SamplingConfig, sample_frame_groups
from hma.video.quality import FrameQuality, QualityConfig, QualityReport, iter_selected_frames

EmbedFn = Callable[[Sequence[bytes]], Sequence[np.ndarray]]


@dataclass(frozen=True)
class PipelineConfig:
    """
    sampling, quality, dedupe: the settings of those steps.
    max_frames: cap on the frames kept per video; None (the default) means no cap, because
        a fixed cap would silently drop frames of long videos. When set, the frames are
        thinned evenly after de-duplication. Embedding cost grows with the video's length.
    embed_batch_size: frames sent to the embedding function at once.
    jpeg_quality: quality (1-100) of the JPEG each kept frame is encoded as.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)
    max_frames: Optional[int] = None
    embed_batch_size: int = 32
    jpeg_quality: int = 90

    def __post_init__(self):
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1, or None for no cap")
        if self.embed_batch_size < 1:
            raise ValueError("embed_batch_size must be at least 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Every step's settings from the environment, plus VIDEO_MAX_FRAMES (unset, empty or
        0 means no cap), VIDEO_EMBED_BATCH_SIZE and VIDEO_JPEG_QUALITY."""
        defaults = cls()

        def env(name, default):
            value = os.getenv(name, "").strip()
            return int(value) if value else default

        return cls(
            sampling=SamplingConfig.from_env(),
            quality=QualityConfig.from_env(),
            dedupe=DedupeConfig.from_env(),
            max_frames=env("VIDEO_MAX_FRAMES", 0) or None,
            embed_batch_size=env("VIDEO_EMBED_BATCH_SIZE", defaults.embed_batch_size),
            jpeg_quality=env("VIDEO_JPEG_QUALITY", defaults.jpeg_quality),
        )


@dataclass(frozen=True)
class KeptFrame:
    index: int  # frame number in the video
    seconds: float  # position in the video
    jpeg: bytes  # the frame, JPEG encoded
    embedding: np.ndarray  # float32 vector from the embedding function
    quality: FrameQuality


@dataclass
class PipelineReport:
    """What happened to the candidate frames of one video."""

    quality: QualityReport = field(default_factory=QualityReport)
    pixel_duplicates: int = 0  # dropped before embedding because a thumbnail repeated
    embedding_duplicates: int = 0  # dropped after embedding because an embedding repeated
    thinned: int = 0  # dropped to stay within max_frames
    kept: int = 0

    @property
    def no_usable_frames(self) -> bool:
        """The video produced no frame at all: tell the user."""
        return self.kept == 0

    def summary(self) -> str:
        text = (
            f"{self.quality.summary()}; {self.pixel_duplicates} pixel and "
            f"{self.embedding_duplicates} embedding duplicates removed"
        )
        if self.thinned:
            text += f"; {self.thinned} thinned to the frame cap"
        return text + f"; {self.kept} frames kept"


@dataclass(frozen=True)
class VideoFrames:
    frames: List[KeptFrame]
    report: PipelineReport


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode a frame as JPEG")
    return buffer.tobytes()


def select_video_frames(
    path, embed: EmbedFn, config: Optional[PipelineConfig] = None
) -> VideoFrames:
    """
    The frames that represent one video, with their embeddings and a report.

    Raises ValueError if the file cannot be read or `embed` returns the wrong number of
    vectors. A video whose frames are all junk returns no frames, and the report says so
    (`report.no_usable_frames`).
    """
    config = config or PipelineConfig.from_env()
    report = PipelineReport()
    deduper = FrameDeduper(config.dedupe)
    kept: List[KeptFrame] = []
    pending = []  # (SelectedFrame, jpeg) waiting for their embeddings

    def flush():
        if not pending:
            return
        vectors = embed([jpeg for _, jpeg in pending])
        if len(vectors) != len(pending):
            raise ValueError(f"embed returned {len(vectors)} vectors for {len(pending)} frames")
        for (selected, jpeg), vector in zip(pending, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            if deduper.embedding_duplicate(vector):
                report.embedding_duplicates += 1
                continue
            frame = selected.frame
            kept.append(KeptFrame(frame.index, frame.seconds, jpeg, vector, selected.quality))
        pending.clear()

    groups = sample_frame_groups(Path(path), config.sampling)
    for selected in iter_selected_frames(groups, config.quality, report.quality):
        if deduper.pixel_duplicate(thumbnail(selected.frame.image, config.dedupe.thumbnail_size)):
            report.pixel_duplicates += 1
            continue
        pending.append((selected, encode_jpeg(selected.frame.image, config.jpeg_quality)))
        if len(pending) >= config.embed_batch_size:
            flush()
    flush()

    frames = thin_evenly(kept, config.max_frames)
    report.thinned = len(kept) - len(frames)
    report.kept = len(frames)
    return VideoFrames(frames=frames, report=report)
