"""Index the frames of videos in an HMA bank and score a query video against them.

A video is a bag of frames. Each frame kept by the pipeline is one bank item, hashed with the
CLIP image signal, and a query video matches a candidate video in proportion to the share of
the query's own frames that found one of the candidate's frames.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hma.hma_client import HMAClient
from hma.logging_utils import _log_info, _log_warning
from hma.video.pipeline import PipelineConfig, select_video_frames

Match = tuple[str, float]  # bank content id, distance


@dataclass(frozen=True)
class FrameRef:
    """Which frame of which video a bank item is."""

    video: str
    index: int
    seconds: float


@dataclass
class VideoIndex:
    """What was added to the bank."""

    frames: dict[str, FrameRef] = field(default_factory=dict)  # bank content id -> frame
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # video -> (frames, dimensions)
    unusable: list[str] = field(default_factory=list)  # videos with no usable frame, left out


def encode_signal(embedding: np.ndarray) -> str:
    """The hex string HMA uses for a CLIP vector (float32 bytes)."""
    return np.asarray(embedding, dtype=np.float32).tobytes().hex()


def decode_signal(signal: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(signal), dtype=np.float32)


def hma_embedder(client: HMAClient, signal_type: str = "clip_float") -> Callable[[Sequence[bytes]], list[np.ndarray]]:
    """The embedding function for the pipeline: HMA hashes the frames and returns their vectors."""

    def embed(jpegs: Sequence[bytes]) -> list[np.ndarray]:
        return [decode_signal(signal) for signal in client.hash_image_bytes(list(jpegs), signal_type)]

    return embed


def index_videos(
    client: HMAClient,
    bank_name: str,
    videos: Iterable[str | Path],
    signal_type: str = "clip_float",
    config: PipelineConfig | None = None,
) -> VideoIndex:
    """Select the frames of each video and add them to the bank, one bank item per frame.

    The bank does not remember which video an item came from, so the returned index does.
    Raises if a frame cannot be added, because a frame missing from the bank would silently
    change every score.
    """
    config = config or PipelineConfig.from_env()
    embed = hma_embedder(client, signal_type)
    index = VideoIndex()
    for video in map(str, videos):
        result = select_video_frames(video, embed, config)
        summary = f"{video}: {result.report.summary()}"
        if result.report.no_usable_frames:
            _log_warning(f"{summary}. It is left out of the index.")
            index.unusable.append(video)
            continue
        (_log_warning if result.report.quality.slices_dropped else _log_info)(summary)
        for frame in result.frames:
            content_id = client.add_signal_to_bank(bank_name, signal_type, encode_signal(frame.embedding))
            if content_id is None:
                raise RuntimeError(f"Could not add frame {frame.index} of {video} to bank {bank_name}")
            index.frames[content_id] = FrameRef(video, frame.index, frame.seconds)
        index.embeddings[video] = np.stack([frame.embedding for frame in result.frames])
    return index


def lookup_frames(
    client: HMAClient,
    embeddings: Iterable[np.ndarray],
    max_distance: float,
    signal_type: str = "clip_float",
    max_workers: int = 8,
) -> list[list[Match]]:
    """For each frame, the bank items within max_distance of it. The distances are kept, so one
    lookup at the largest distance of interest serves every smaller one."""

    def lookup(embedding: np.ndarray) -> list[Match]:
        response = client.match_with_signal_threshold(encode_signal(embedding), max_distance, signal_type)
        if response.get("status") != "success":
            raise RuntimeError(f"Lookup failed: {response.get('error')}")
        return [(str(m["bank_content_id"]), float(m["distance"])) for m in response["matches"]]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lookup, embeddings))


def video_scores(
    frame_matches: Sequence[Sequence[Match]],
    frames: Mapping[str, FrameRef],
    max_distance: float,
    exclude_video: str | None = None,
) -> dict[str, float]:
    """Score each candidate video by the share of the query's frames that matched at least one
    of its frames within max_distance. A video with no match is absent, and the query's own
    video (exclude_video) is never scored."""
    if not frame_matches:
        return {}
    matched = Counter()
    for matches in frame_matches:
        videos = {frames[content_id].video for content_id, distance in matches if distance <= max_distance and content_id in frames}
        videos.discard(exclude_video)
        matched.update(videos)
    return {video: count / len(frame_matches) for video, count in matched.items()}
