"""Rank videos, split the series, and sweep the frame distance: the retrieval side of the video evaluation.

Everything here works on the frame matches that frame_index.lookup_frames returns, so one lookup
at the largest frame distance serves every smaller one.
"""

import random
from bisect import bisect_left
from typing import Iterable, Iterator, Mapping, Sequence

from hma.metrics.map import compute_series_map
from hma.video.frame_index import FrameRef, Match, closest_distances, video_scores

SCORE_ONLY = "score_only"
CLOSEST_FRAME = "closest_frame"
RANKINGS = (SCORE_ONLY, CLOSEST_FRAME)

MatchesByQuery = Mapping[str, Sequence[Sequence[Match]]]


def distance_grid(max_distance: float, step: float) -> list[float]:
    """The frame distances to sweep: step, 2 * step, ... up to max_distance."""
    if step <= 0 or max_distance < step:
        raise ValueError("The distance step must be positive and no larger than the maximum distance.")
    return [round(i * step, 6) for i in range(1, int(round(max_distance / step)) + 1)]


def split_series(series: Iterable[str], test_fraction: float = 0.5, seed: int = 0) -> tuple[set[str], set[str]]:
    """Split series names into (train, test), shuffled with a fixed seed so every run splits alike."""
    names = sorted(series)
    if len(names) < 2:
        raise ValueError("Need at least two series to split into train and test.")
    if not 0 < test_fraction < 1:
        raise ValueError("The test fraction must be between 0 and 1.")
    random.Random(seed).shuffle(names)
    test_size = min(max(round(len(names) * test_fraction), 1), len(names) - 1)
    return set(names[test_size:]), set(names[:test_size])


def rank_videos(
    scores: Mapping[str, float],
    closest: Mapping[str, float],
    videos: Iterable[str],
    query: str,
    ranking: str,
    seed: int = 0,
) -> list[str]:
    """Every video but the query, best first.

    Both rankings put the higher score first. Candidates that still tie are put in a fixed random
    order (the same for the same seed and query, whatever folder the videos are in), so the file
    names never decide a ranking.
    CLOSEST_FRAME first breaks ties by the distance of the candidate's closest frame, and a
    candidate none of whose frames were looked up ranks last among its score.
    """
    if ranking not in RANKINGS:
        raise ValueError(f"Unknown ranking {ranking!r}; expected one of {RANKINGS}")
    ordered = sorted(videos)
    # Seeded by the query's place among the sorted videos, not by its path, so the order is the
    # same wherever the files are.
    others = [video for video in ordered if video != query]
    random.Random(f"{seed}/{bisect_left(ordered, query)}").shuffle(others)
    if ranking == CLOSEST_FRAME:
        return sorted(others, key=lambda v: (-scores.get(v, 0.0), closest.get(v, float("inf"))))
    return sorted(others, key=lambda v: -scores.get(v, 0.0))


def rankings_at(
    matches_by_query: MatchesByQuery,
    frames: Mapping[str, FrameRef],
    videos: Sequence[str],
    max_distance: float,
    seed: int = 0,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Rank every video for every query at one frame distance, once per ranking.

    Returns ranking name -> {query -> [(video, 1 - score), ...] best first}. The 1 - score is a
    distance in the sense the metrics use: smaller is more similar.
    """
    result: dict[str, dict[str, list[tuple[str, float]]]] = {ranking: {} for ranking in RANKINGS}
    for query, frame_matches in matches_by_query.items():
        scores = video_scores(frame_matches, frames, max_distance, exclude_video=query)
        closest = closest_distances(frame_matches, frames, exclude_video=query)
        for ranking in RANKINGS:
            ordered = rank_videos(scores, closest, videos, query, ranking, seed)
            result[ranking][query] = [(v, 1 - scores.get(v, 0.0)) for v in ordered]
    return result


def mean_map(
    series_to_videos: Mapping[str, set[str]],
    rankings: Mapping[str, list[tuple[str, float]]],
    max_k: int,
) -> list[float]:
    """mAP@k for k = 1..max_k, averaged over the series (each series' queries are the videos in it)."""
    per_series = compute_series_map(dict(series_to_videos), dict(rankings), max_k)
    return [sum(values[k] for values in per_series.values()) / len(per_series) for k in range(max_k)]


def score_pairs(
    matches_by_query: MatchesByQuery,
    frames: Mapping[str, FrameRef],
    videos: Sequence[str],
    max_distance: float,
    queries: Iterable[str] | None = None,
) -> Iterator[tuple[str, str, float]]:
    """Every (query, other video, 1 - score) at one frame distance, including the videos that matched nothing.

    Unmatched videos count (as distance 1) because a classifier that never finds a video is
    wrong about it, and precision/recall would flatter it if they were left out.
    """
    for query in matches_by_query if queries is None else queries:
        scores = video_scores(matches_by_query[query], frames, max_distance, exclude_video=query)
        for video in videos:
            if video != query:
                yield query, video, 1 - scores.get(video, 0.0)
