"""Tests for hma.video.retrieval."""

import pytest

from hma.video.frame_index import FrameRef
from hma.video.retrieval import (
    CLOSEST_FRAME,
    SCORE_ONLY,
    distance_grid,
    mean_map,
    rank_videos,
    rankings_at,
    score_pairs,
    split_series,
)

# bank items 1-2 are frames of "a", 3 of "b", 4 of "c", 5 of "d"
FRAMES = {
    "1": FrameRef("a", 0, 0.5),
    "2": FrameRef("a", 6, 1.5),
    "3": FrameRef("b", 0, 0.5),
    "4": FrameRef("c", 0, 0.5),
    "5": FrameRef("d", 0, 0.5),
}
VIDEOS = ["a", "b", "c", "d"]
NAMES = {v: v.upper() for v in VIDEOS}


class TestDistanceGrid:
    def test_steps_up_to_the_maximum(self):
        assert distance_grid(0.3, 0.05) == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    def test_a_step_that_does_not_divide_the_maximum_stops_short(self):
        assert distance_grid(0.3, 0.1) == [0.1, 0.2, 0.3]
        assert distance_grid(0.25, 0.1) == [0.1, 0.2]

    @pytest.mark.parametrize("args", [(0.3, 0), (0.3, -0.1), (0.05, 0.1)])
    def test_rejects_an_impossible_grid(self, args):
        with pytest.raises(ValueError):
            distance_grid(*args)


class TestSplitSeries:
    def test_halves_the_series_without_overlap(self):
        train, test = split_series([f"s{i}" for i in range(10)])
        assert len(train) == len(test) == 5 and not train & test

    def test_the_same_seed_gives_the_same_split_whatever_the_input_order(self):
        names = [f"s{i}" for i in range(20)]
        assert split_series(names, seed=3) == split_series(reversed(names), seed=3)

    def test_a_different_seed_gives_a_different_split(self):
        names = [f"s{i}" for i in range(20)]
        assert split_series(names, seed=1) != split_series(names, seed=2)

    def test_both_halves_are_never_empty(self):
        train, test = split_series(["a", "b"], test_fraction=0.99)
        assert len(train) == len(test) == 1
        train, test = split_series(["a", "b", "c"], test_fraction=0.01)
        assert len(test) == 1 and len(train) == 2

    def test_the_test_fraction_sets_the_test_size(self):
        train, test = split_series([f"s{i}" for i in range(10)], test_fraction=0.3)
        assert (len(train), len(test)) == (7, 3)

    @pytest.mark.parametrize("kwargs", [{"test_fraction": 0}, {"test_fraction": 1}])
    def test_rejects_a_fraction_outside_zero_to_one(self, kwargs):
        with pytest.raises(ValueError):
            split_series(["a", "b", "c"], **kwargs)

    def test_needs_two_series(self):
        with pytest.raises(ValueError):
            split_series(["a"])


class TestRankVideos:
    def test_the_higher_score_comes_first_and_the_query_is_left_out(self):
        ranked = rank_videos({"b": 0.5, "c": 1.0}, {}, VIDEOS, "a", SCORE_ONLY)
        assert ranked[:2] == ["c", "b"] and set(ranked) == {"b", "c", "d"}

    def test_closest_frame_breaks_a_tie_in_score(self):
        scores = {"b": 0.5, "c": 0.5}
        assert rank_videos(scores, {"b": 0.09, "c": 0.04}, VIDEOS, "a", CLOSEST_FRAME)[:2] == ["c", "b"]
        assert rank_videos(scores, {"b": 0.04, "c": 0.09}, VIDEOS, "a", CLOSEST_FRAME)[:2] == ["b", "c"]

    def test_score_still_beats_the_closest_frame(self):
        ranked = rank_videos({"b": 1.0}, {"b": 0.9, "c": 0.0}, VIDEOS, "a", CLOSEST_FRAME)
        assert ranked[0] == "b"

    def test_a_video_with_no_looked_up_frame_ranks_last_among_its_score(self):
        ranked = rank_videos({}, {"c": 0.2}, VIDEOS, "a", CLOSEST_FRAME)
        assert ranked[0] == "c"

    def test_tied_videos_are_in_a_fixed_order_that_does_not_depend_on_the_input_order(self):
        names = [f"v{i:02d}" for i in range(30)]
        first = rank_videos({}, {}, names, "v00", SCORE_ONLY, seed=1)
        assert first == rank_videos({}, {}, reversed(names), "v00", SCORE_ONLY, seed=1)
        assert first != sorted(first)  # not simply alphabetical
        assert first != rank_videos({}, {}, names, "v00", SCORE_ONLY, seed=2)

    def test_the_tie_order_differs_between_queries(self):
        names = [f"v{i:02d}" for i in range(30)]
        assert rank_videos({}, {}, names, "v00", SCORE_ONLY)[:5] != rank_videos({}, {}, names, "v01", SCORE_ONLY)[:5]

    def test_rejects_an_unknown_ranking(self):
        with pytest.raises(ValueError):
            rank_videos({}, {}, VIDEOS, "a", "alphabetical")


MATCHES = {
    "a": [[("3", 0.04), ("4", 0.12)], [("5", 0.30)]],  # a's two frames: one finds b and c, one finds d far away
    "b": [[("1", 0.04)]],
    "c": [[("1", 0.12), ("5", 0.20)]],
    "d": [],  # a video with no usable frame
}


class TestRankingsAt:
    def test_ranks_every_other_video_under_both_rankings_using_the_names(self):
        result = rankings_at(MATCHES, FRAMES, VIDEOS, NAMES, 0.1)
        assert set(result) == {SCORE_ONLY, CLOSEST_FRAME}
        for ranked in result.values():
            assert set(ranked) == {"A", "B", "C", "D"}
            assert sorted(v for v, _ in ranked["A"]) == ["B", "C", "D"]

    def test_the_second_item_is_one_minus_the_score(self):
        ranked = rankings_at(MATCHES, FRAMES, VIDEOS, NAMES, 0.1)[CLOSEST_FRAME]
        assert ranked["A"][0] == ("B", pytest.approx(0.5))  # one of a's two frames is within 0.1 of b
        assert dict(ranked["A"])["C"] == 1.0  # c is 0.12 away, outside 0.1

    def test_a_larger_distance_finds_more(self):
        wide = dict(rankings_at(MATCHES, FRAMES, VIDEOS, NAMES, 0.15)[CLOSEST_FRAME]["A"])
        assert wide["C"] == pytest.approx(0.5)

    def test_a_query_with_no_frames_ranks_everything_at_distance_one(self):
        ranked = rankings_at(MATCHES, FRAMES, VIDEOS, NAMES, 0.1)[SCORE_ONLY]["D"]
        assert sorted(v for v, _ in ranked) == ["A", "B", "C"] and {d for _, d in ranked} == {1.0}

    def test_the_closest_frame_ranking_orders_unmatched_videos_by_their_closest_frame(self):
        ranked = rankings_at(MATCHES, FRAMES, VIDEOS, NAMES, 0.05)[CLOSEST_FRAME]["A"]
        assert [v for v, _ in ranked] == ["B", "C", "D"]  # only b is within 0.05; c is closer than d


class TestMeanMap:
    def test_averages_over_the_series(self):
        series = {"s1": {"A", "B"}, "s2": {"C", "D"}}
        rankings = {
            "A": [("B", 0.0), ("C", 1.0)],
            "B": [("C", 0.5), ("A", 1.0)],  # s1's positive is second: AP@1 0, AP@2 0.5
            "C": [("D", 0.0), ("A", 1.0)],
            "D": [("C", 0.0), ("A", 1.0)],
        }
        assert mean_map(series, rankings, 2) == pytest.approx([(0.5 + 1) / 2, (0.75 + 1) / 2])

    def test_only_the_given_series_are_scored(self):
        series = {"s1": {"A", "B"}}
        rankings = {"A": [("B", 0.0)], "B": [("A", 0.0)], "C": []}
        assert mean_map(series, rankings, 1) == [1.0]


class TestScorePairs:
    def test_every_query_is_paired_with_every_other_video_including_unmatched_ones(self):
        pairs = list(score_pairs(MATCHES, FRAMES, VIDEOS, NAMES, 0.1))
        assert len(pairs) == 4 * 3
        assert ("A", "B", pytest.approx(0.5)) in pairs
        assert ("A", "D", 1.0) in pairs
        assert not any(a == b for a, b, _ in pairs)

    def test_only_the_given_queries_are_used(self):
        pairs = list(score_pairs(MATCHES, FRAMES, VIDEOS, NAMES, 0.1, queries=["b"]))
        assert {a for a, _, _ in pairs} == {"B"} and len(pairs) == 3
