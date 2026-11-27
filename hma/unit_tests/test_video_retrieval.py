"""Tests for hma.video.retrieval."""

import pytest

from hma.video.retrieval import closest_pairs, distance_grid, rankings_at
from video_testing import FRAMES


class TestDistanceGrid:
    def test_steps_up_to_the_maximum(self):
        assert distance_grid(0.3, 0.05) == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    def test_a_step_that_does_not_divide_the_maximum_stops_short(self):
        assert distance_grid(0.3, 0.1) == [0.1, 0.2, 0.3]
        assert distance_grid(0.25, 0.1) == [0.1, 0.2]
        assert distance_grid(0.3, 0.16) == [0.16]

    @pytest.mark.parametrize("args", [(0.3, 0), (0.3, -0.1), (0.05, 0.1)])
    def test_rejects_an_impossible_grid(self, args):
        with pytest.raises(ValueError):
            distance_grid(*args)


MATCHES = {
    "a": [[("3", 0.04), ("4", 0.12)], [("5", 0.30)]],  # a's two frames: one finds b and c, one finds d far away
    "b": [[("1", 0.04)]],
    "c": [[("1", 0.12), ("5", 0.20)]],
    "d": [],  # a video with no usable frame
}


class TestRankingsAt:
    def test_ranks_the_matched_videos_of_every_query(self):
        score = pytest.approx(0.6)  # 1 - 0.04 / 0.1
        assert rankings_at(MATCHES, FRAMES, 0.1) == {"a": [("b", score)], "b": [("a", score)], "c": [], "d": []}

    def test_a_larger_distance_finds_more(self):
        assert [video for video, _ in rankings_at(MATCHES, FRAMES, 0.3)["a"]] == ["b", "c", "d"]


class TestClosestPairs:
    def test_every_query_is_paired_with_every_other_video_including_unmatched_ones(self):
        pairs = list(closest_pairs(MATCHES, FRAMES))
        assert len(pairs) == 4 * 3 and not any(a == b for a, b, _ in pairs)
        assert ("a", "b", 0.04) in pairs and ("a", "d", 0.30) in pairs
        assert ("b", "c", float("inf")) in pairs and ("d", "a", float("inf")) in pairs

    def test_a_pair_is_within_a_distance_exactly_when_the_ranking_returns_it(self):
        for distance in (0.04, 0.1, 0.15, 0.3):
            ranked = rankings_at(MATCHES, FRAMES, distance)
            returned = {(query, video) for query, videos in ranked.items() for video, _ in videos}
            assert returned == {(q, v) for q, v, d in closest_pairs(MATCHES, FRAMES) if d <= distance}
