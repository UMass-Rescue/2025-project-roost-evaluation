"""Rank videos, split the series, and score the matches: what the video evaluation computes from the frame lookups.

Everything here works on the frame matches that frame_index.lookup_frames returns, so one lookup
at the largest frame distance serves every smaller one.
"""

import random
from typing import Iterable, Iterator, Mapping, NamedTuple, Sequence

from hma.metrics.map import compute_series_map
from hma.video.frame_index import FrameRef, Match, closest_distances, video_scores

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


def rank_videos(scores: Mapping[str, float], closest: Mapping[str, float], videos: Iterable[str], query: str) -> list[str]:
    """Every video but the query, best first: the higher score, and among equal scores the nearer closest frame.

    The closest frame is taken from all the lookup results (which reach to the largest distance
    swept, not only the distance of the score). A video none of whose frames were looked up ranks
    last among its score, and videos still tied after that are in name order.
    """
    others = sorted(video for video in videos if video != query)
    return sorted(others, key=lambda video: (-scores.get(video, 0.0), closest.get(video, float("inf"))))


def rankings_at(
    matches_by_query: MatchesByQuery,
    frames: Mapping[str, FrameRef],
    videos: Sequence[str],
    max_distance: float,
) -> dict[str, list[tuple[str, float]]]:
    """Rank every video for every query at one frame distance: query -> [(video, 1 - score), ...] best first.

    The 1 - score is a distance in the sense the metrics use: smaller is more similar.
    """
    result = {}
    for query, frame_matches in matches_by_query.items():
        scores = video_scores(frame_matches, frames, max_distance, exclude_video=query)
        closest = closest_distances(frame_matches, frames, exclude_video=query)
        result[query] = [(video, 1 - scores.get(video, 0.0)) for video in rank_videos(scores, closest, videos, query)]
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


class BestF1(NamedTuple):
    f1: float
    precision: float
    recall: float


def best_f1(labeled: Iterable[tuple[float, bool]]) -> BestF1:
    """The best F1 over every score threshold of (distance, is_same_series) pairs; a pair is a match at or below the threshold.

    Zeros when there are no same-series pairs.
    """
    pairs = sorted(labeled)
    positives = sum(is_same for _, is_same in pairs)
    best, tp, fp = BestF1(0.0, 0.0, 0.0), 0, 0
    if not positives:
        return best
    for i, (distance, is_same) in enumerate(pairs):
        tp, fp = tp + is_same, fp + (not is_same)
        if i + 1 < len(pairs) and pairs[i + 1][0] == distance:
            continue  # a threshold takes every pair at the same distance
        precision, recall = tp / (tp + fp), tp / positives
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        if f1 > best.f1:
            best = BestF1(f1, precision, recall)
    return best
