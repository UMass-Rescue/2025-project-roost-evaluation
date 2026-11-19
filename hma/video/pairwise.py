"""A distance between two videos, computed from their frame embeddings.

The distance from video A to video B is the mean, over A's frames, of the cosine distance to the
closest frame of B. The distance between the two videos is the mean of the two directions.
It needs no threshold and no index, so it measures the embeddings alone.
"""

from typing import Iterator, Mapping

import numpy as np


def video_distances(embeddings: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return the videos (sorted) and the symmetric matrix of distances between them.

    The matrix is videos x videos, so memory grows with the square of the number of videos.
    """
    videos = sorted(embeddings)
    if not videos:
        raise ValueError("No videos to compare.")
    unit = [_normalised(embeddings[v]) for v in videos]
    frames = np.concatenate(unit)
    starts = np.cumsum([0] + [len(u) for u in unit[:-1]])
    directed = np.zeros((len(videos), len(videos)))
    for i, own in enumerate(unit):
        closest = np.minimum.reduceat(1 - own @ frames.T, starts, axis=1)  # own frames x videos
        directed[i] = closest.mean(axis=0)
    return videos, (directed + directed.T) / 2


def distance_pairs(videos: list[str], distances: np.ndarray) -> Iterator[tuple[str, str, float]]:
    """Each unordered pair of videos once, with its distance."""
    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            yield videos[i], videos[j], float(distances[i, j])


def _normalised(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not norms.all():
        raise ValueError("A zero embedding has no cosine distance.")
    return vectors / norms
