"""Turn the frame lookups of every query into rankings and video pairs for the metrics.

Everything here works on the frame matches that frame_index.lookup_frames returns, so one lookup
at the largest frame distance serves every smaller one.
"""

from itertools import chain
from typing import Iterator, Mapping, Sequence

from hma.video.frame_index import FrameRef, Match, closest_distances, rank_matches

MatchesByQuery = Mapping[str, Sequence[Sequence[Match]]]


def distance_grid(max_distance: float, step: float) -> list[float]:
    """The frame distances to sweep: step, 2 * step, ... up to max_distance."""
    if step <= 0 or max_distance < step:
        raise ValueError("The distance step must be positive and no larger than the maximum distance.")
    return [round(i * step, 6) for i in range(1, int(max_distance / step + 1e-9) + 1)]  # the sliver allows for rounding


def rankings_at(
    matches_by_query: MatchesByQuery, frames: Mapping[str, FrameRef], max_distance: float
) -> dict[str, list[tuple[str, float]]]:
    """The videos each query matches at one frame distance: query -> [(video, score), ...] best first."""
    return {
        query: rank_matches(frame_matches, frames, max_distance, query)
        for query, frame_matches in matches_by_query.items()
    }


def closest_pairs(matches_by_query: MatchesByQuery, frames: Mapping[str, FrameRef]) -> Iterator[tuple[str, str, float]]:
    """Every (query, other query, closest frame distance). A query matches a video at every frame distance
    from that one up, because one matched frame is enough.

    A video that no lookup returned is infinitely far. It is kept because a classifier that never
    finds a video is wrong about it, and recall would flatter it if it were left out.
    """
    for query, frame_matches in matches_by_query.items():
        closest = closest_distances(chain.from_iterable(frame_matches), frames, exclude_video=query)
        for video in matches_by_query:
            if video != query:
                yield query, video, closest.get(video, float("inf"))
