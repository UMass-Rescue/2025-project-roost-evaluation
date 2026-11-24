"""Index the frames of videos in an HMA bank and rank the videos that match a query video.

A video is a bag of frames. Each frame kept by the pipeline is one bank item, hashed with the
CLIP image signal. The videos a query matches are ranked by a score that adds up how close each
of the query's frames came to one of their frames.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hma.hma_client import HMAClient
from hma.logging_utils import _log_info, _log_warning
from hma.video.frames import UnreadableVideoError
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
    A video that cannot be read, or has no usable frame, is listed as unusable and left out.
    Raises if a frame cannot be added, because a frame missing from the bank would silently
    change the matches of every query.
    """
    config = config or PipelineConfig.from_env()
    embed = hma_embedder(client, signal_type)
    index = VideoIndex()
    for video in map(str, videos):
        try:
            result = select_video_frames(video, embed, config)
        except UnreadableVideoError as error:
            _log_warning(f"{error}. It is left out of the index.")
            index.unusable.append(video)
            continue
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


def wait_until_searchable(
    client: HMAClient,
    index: VideoIndex,
    signal_type: str = "clip_float",
    timeout: float = 180.0,
    interval: float = 2.0,
) -> None:
    """Block until a lookup finds the frame added last, so every frame added before it is searchable too.

    HMA serves lookups from a copy of its index that it refreshes in the background, and the index
    size it reports can be ahead of that copy, so the size is not enough to know when to look up.
    Raises TimeoutError if the frame is not found within timeout seconds.
    """
    if not index.frames:
        return
    last_id = next(reversed(index.frames))
    probe = encode_signal(index.embeddings[index.frames[last_id].video][-1])
    deadline = time.monotonic() + timeout
    while True:
        response = client.match_with_signal_threshold(probe, 0.001, signal_type)
        if response.get("status") == "success" and any(str(m["bank_content_id"]) == last_id for m in response["matches"]):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Frame {last_id} was not searchable after {timeout:.0f}s; HMA's index has not caught up.")
        time.sleep(interval)


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


def rank_matches(
    frame_matches: Sequence[Sequence[Match]],
    frames: Mapping[str, FrameRef],
    max_distance: float,
    exclude_video: str | None = None,
) -> list[tuple[str, float]]:
    """The videos that matched a query and their scores, best first. Take the first k for the top k.

    A query frame matches a video when one of the video's frames is within max_distance of it. It
    counts once, at the distance of the closest such frame, and adds 1 - (distance / max_distance)
    to the video's score: 1 for an identical frame, 0 for one at max_distance. The highest score
    ranks first, then the name. A video with no matched frame is absent, and so is the query's own
    video (exclude_video).
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    scores: dict[str, float] = defaultdict(float)
    for matches in frame_matches:
        for video, distance in closest_distances(matches, frames, exclude_video).items():
            if distance <= max_distance:
                scores[video] += 1 - distance / max_distance
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def closest_distances(
    matches: Iterable[Match],
    frames: Mapping[str, FrameRef],
    exclude_video: str | None = None,
) -> dict[str, float]:
    """For each video that has a frame among the matches, the distance of its closest frame.
    The query's own video (exclude_video) is left out."""
    closest: dict[str, float] = {}
    for content_id, distance in matches:
        frame = frames.get(content_id)
        if frame is None or frame.video == exclude_video:
            continue
        if distance < closest.get(frame.video, float("inf")):
            closest[frame.video] = distance
    return closest
