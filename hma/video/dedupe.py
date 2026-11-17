"""Remove redundant frames of one video, and thin what is left evenly if a cap is set.

A long static shot gives many near-identical frames; keeping all of them only inflates the
index and makes that shot count for more than it should when a video is scored by the
share of its frames that matched. Frames are compared in two steps, cheapest first:

1. tiny grayscale thumbnails, before any embedding is computed, so an obviously repeated
   frame never costs an embedding;
2. the embeddings of the frames that passed, by cosine distance.

In both steps the first frame of a group of near-duplicates wins, and each frame is compared
with every frame kept so far, so duplicates need not be adjacent. This module is
self-contained so it can move into a package of its own later.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, TypeVar

import cv2
import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class DedupeConfig:
    """
    pixel_diff_max: frames whose thumbnails differ by less than this (mean absolute
        difference of 0-255 grayscale values) are duplicates and are not embedded at all.
    embedding_distance_max: frames whose embeddings are closer than this (cosine distance)
        are duplicates.
    thumbnail_size: side of the square grayscale thumbnail the first step compares.
    """

    pixel_diff_max: float = 2.0
    embedding_distance_max: float = 0.02
    thumbnail_size: int = 16

    def __post_init__(self):
        if self.pixel_diff_max < 0 or self.embedding_distance_max < 0:
            raise ValueError("dedupe thresholds must not be negative")
        if self.thumbnail_size < 2:
            raise ValueError("thumbnail_size must be at least 2")

    @classmethod
    def from_env(cls) -> "DedupeConfig":
        """Read VIDEO_DEDUPE_PIXEL_DIFF_MAX and VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX;
        unset or empty values keep the defaults."""
        defaults = cls()

        def env(name, default):
            value = os.getenv(name, "").strip()
            return float(value) if value else default

        return cls(
            pixel_diff_max=env("VIDEO_DEDUPE_PIXEL_DIFF_MAX", defaults.pixel_diff_max),
            embedding_distance_max=env(
                "VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX", defaults.embedding_distance_max
            ),
        )


def thumbnail(image: np.ndarray, size: int = 16) -> np.ndarray:
    """A flattened size x size grayscale copy of a BGR frame, as float32."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


class _Rows:
    """A growing set of equal-length vectors that can be compared with a new one in one step."""

    def __init__(self):
        self._data: Optional[np.ndarray] = None
        self._count = 0

    def add(self, row: np.ndarray) -> None:
        if self._data is None:
            self._data = np.empty((16, row.shape[0]), dtype=np.float32)
        elif self._count == len(self._data):
            self._data = np.concatenate([self._data, np.empty_like(self._data)])
        self._data[self._count] = row
        self._count += 1

    @property
    def rows(self) -> np.ndarray:
        return self._data[: self._count]

    def __len__(self) -> int:
        return self._count


class FrameDeduper:
    """
    Decides, frame by frame and in video order, which frames repeat an earlier one.

    Both checks record a frame the first time they see something new, so call them once per
    frame, in order. The thumbnail check remembers every frame that got past it; the
    embedding check remembers only the frames it kept.
    """

    def __init__(self, config: Optional[DedupeConfig] = None):
        self.config = config or DedupeConfig()
        self._thumbnails = _Rows()
        self._embeddings = _Rows()

    def pixel_duplicate(self, thumb: np.ndarray) -> bool:
        """True if a frame already seen looks almost the same; otherwise remembers this one."""
        thumb = np.asarray(thumb, dtype=np.float32)
        if len(self._thumbnails):
            closest = np.abs(self._thumbnails.rows - thumb).mean(axis=1).min()
            if closest < self.config.pixel_diff_max:
                return True
        self._thumbnails.add(thumb)
        return False

    def embedding_duplicate(self, embedding: np.ndarray) -> bool:
        """True if a kept frame's embedding is closer than the threshold; otherwise keeps this one."""
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("cannot compare a zero embedding")
        vector = vector / norm
        if len(self._embeddings):
            closest = 1.0 - float((self._embeddings.rows @ vector).max())
            if closest < self.config.embedding_distance_max:
                return True
        self._embeddings.add(vector)
        return False


def thin_evenly(items: Sequence[T], max_items: Optional[int]) -> List[T]:
    """
    At most max_items of `items`, spread evenly over the sequence and including the first
    and last. `items` are the frames kept after de-duplication, in video order, so spreading
    over them spreads over the distinct content of the video, not over the clock: a long
    static shot is one item and is not allowed to use up the cap. None means no cap.
    """
    if max_items is None or len(items) <= max_items:
        return list(items)
    if max_items < 1:
        raise ValueError("max_items must be at least 1, or None for no cap")
    if max_items == 1:
        return [items[len(items) // 2]]
    positions = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    return [items[i] for i in positions]
