"""Tests for hma.video.frame_index, against a stand-in for the HMA service.

The stand-in hashes a frame as its normalised 8x8 grayscale picture and keeps the bank in memory,
so indexing and lookup run end to end on synthetic videos without a service.
"""

import cv2
import numpy as np
import pytest

from hma.video.frame_index import (
    VideoIndex,
    closest_distances,
    decode_signal,
    encode_signal,
    hma_embedder,
    index_videos,
    lookup_frames,
    rank_matches,
    wait_until_searchable,
)
from video_testing import CONFIG, FRAMES, FakeHMA, detailed, scenes_video, write_video


@pytest.fixture
def hma():
    return FakeHMA()


class TestSignalEncoding:
    def test_round_trips_exactly(self):
        vector = np.random.default_rng(0).normal(size=512).astype(np.float32)
        assert np.array_equal(decode_signal(encode_signal(vector)), vector)

    def test_is_the_hex_of_float32_bytes(self):
        assert encode_signal(np.array([1.0], dtype=np.float32)) == "0000803f"

    def test_a_float64_vector_is_stored_as_float32(self):
        assert len(encode_signal(np.ones(4))) == 4 * 4 * 2


class TestEmbedder:
    def test_returns_one_vector_per_frame_in_order(self, hma):
        frames = [cv2.imencode(".jpg", detailed(seed))[1].tobytes() for seed in (1, 2)]
        vectors = hma_embedder(hma)(frames)
        assert [v.shape for v in vectors] == [(64,), (64,)]
        assert not np.allclose(vectors[0], vectors[1])
        assert hma.hash_calls == [(2, "clip_float")]


class TestIndexVideos:
    def test_each_kept_frame_becomes_a_bank_item_that_knows_its_video(self, tmp_path, hma):
        video = scenes_video(tmp_path / "a.avi", [(1, 10), (2, 10)])
        index = index_videos(hma, "BANK", [video], config=CONFIG)
        assert len(hma.bank) == 2 and len(index.frames) == 2
        assert [(ref.video, ref.index) for ref in index.frames.values()] == [(video, 5), (video, 15)]
        assert index.frames["1"].seconds == pytest.approx(0.5)
        assert index.embeddings[video].shape == (2, 64)
        assert index.unusable == []

    def test_the_bank_holds_the_hash_of_the_frame(self, tmp_path, hma):
        video = scenes_video(tmp_path / "a.avi", [(1, 10)])
        index = index_videos(hma, "BANK", [video], config=CONFIG)
        assert np.array_equal(decode_signal(hma.bank["1"]), index.embeddings[video][0])

    def test_several_videos_share_one_bank(self, tmp_path, hma):
        a = scenes_video(tmp_path / "a.avi", [(1, 10)])
        b = scenes_video(tmp_path / "b.avi", [(2, 10), (3, 10)])
        index = index_videos(hma, "BANK", [a, b], config=CONFIG)
        assert sorted(ref.video for ref in index.frames.values()) == [a, b, b]
        assert set(index.embeddings) == {a, b}

    def test_paths_may_be_path_objects(self, tmp_path, hma):
        video = scenes_video(tmp_path / "a.avi", [(1, 10)])
        index = index_videos(hma, "BANK", [tmp_path / "a.avi"], config=CONFIG)
        assert set(index.embeddings) == {video}

    def test_a_video_with_no_usable_frame_is_reported_and_left_out(self, tmp_path, hma):
        flat = write_video(tmp_path / "flat.avi", [np.full((360, 640, 3), 128, np.uint8)] * 20)
        good = scenes_video(tmp_path / "good.avi", [(1, 10)])
        index = index_videos(hma, "BANK", [flat, good], config=CONFIG)
        assert index.unusable == [flat] and set(index.embeddings) == {good} and len(hma.bank) == 1

    def test_a_video_that_cannot_be_read_is_reported_and_left_out(self, tmp_path, hma):
        broken = tmp_path / "broken.avi"
        broken.write_text("this is not a video")
        good = scenes_video(tmp_path / "good.avi", [(1, 10)])
        index = index_videos(hma, "BANK", [str(broken), good], config=CONFIG)
        assert index.unusable == [str(broken)] and set(index.embeddings) == {good} and len(hma.bank) == 1

    def test_an_error_that_is_not_about_the_file_is_not_swallowed(self, tmp_path, hma, monkeypatch):
        video = scenes_video(tmp_path / "a.avi", [(1, 10)])
        monkeypatch.setattr(hma, "hash_image_bytes", lambda images, signal_type: [])  # one vector per frame is required
        with pytest.raises(ValueError, match="vectors"):
            index_videos(hma, "BANK", [video], config=CONFIG)

    def test_a_failed_add_is_an_error(self, tmp_path, hma):
        hma.fail_adds = True
        video = scenes_video(tmp_path / "a.avi", [(1, 10)])
        with pytest.raises(RuntimeError, match="Could not add"):
            index_videos(hma, "BANK", [video], config=CONFIG)

    def test_uses_the_environment_when_no_config_is_given(self, tmp_path, hma, monkeypatch):
        monkeypatch.setenv("VIDEO_SCENE_DETECTOR", "none")
        monkeypatch.setenv("VIDEO_CANDIDATE_WINDOW_SECONDS", "0")
        monkeypatch.setenv("VIDEO_MAX_FRAMES", "1")
        video = scenes_video(tmp_path / "a.avi", [(s, 10) for s in (1, 2, 3)])
        assert len(index_videos(hma, "BANK", [video]).frames) == 1


class TestLookupFrames:
    def test_returns_the_matches_of_each_frame_with_their_distances(self, tmp_path, hma):
        a = scenes_video(tmp_path / "a.avi", [(1, 10), (2, 10)])
        index = index_videos(hma, "BANK", [a], config=CONFIG)
        found = lookup_frames(hma, index.embeddings[a], 0.1)
        assert [[content_id for content_id, _ in matches] for matches in found] == [["1"], ["2"]]
        assert all(distance == pytest.approx(0, abs=1e-5) for matches in found for _, distance in matches)

    def test_a_larger_distance_finds_more(self, tmp_path, hma):
        a = scenes_video(tmp_path / "a.avi", [(1, 10), (2, 10)])
        index = index_videos(hma, "BANK", [a], config=CONFIG)
        assert [len(m) for m in lookup_frames(hma, index.embeddings[a], 2.0)] == [2, 2]

    def test_no_frames_no_lookups(self, hma):
        assert lookup_frames(hma, [], 0.1) == []

    def test_a_failed_lookup_is_an_error(self):
        class Failing:
            def match_with_signal_threshold(self, *args):
                return {"status": "failure", "error": "down"}

        with pytest.raises(RuntimeError, match="down"):
            lookup_frames(Failing(), [np.ones(4, dtype=np.float32)], 0.1)


class TestWaitUntilSearchable:
    @pytest.fixture
    def indexed(self, tmp_path, hma):
        video = scenes_video(tmp_path / "a.avi", [(1, 10), (2, 10)])
        return index_videos(hma, "BANK", [video], config=CONFIG)

    def test_returns_at_once_when_the_last_frame_is_found(self, indexed, hma, monkeypatch):
        monkeypatch.setattr("hma.video.frame_index.time.sleep", lambda s: pytest.fail("should not wait"))
        wait_until_searchable(hma, indexed)

    def test_waits_until_a_lookup_finds_the_last_frame(self, indexed, hma, monkeypatch):
        calls, sleeps = [], []
        lookup = hma.match_with_signal_threshold

        def stale_then_fresh(signal, threshold, signal_type):
            calls.append(threshold)
            return {"status": "success", "matches": []} if len(calls) < 3 else lookup(signal, threshold, signal_type)

        monkeypatch.setattr(hma, "match_with_signal_threshold", stale_then_fresh)
        monkeypatch.setattr("hma.video.frame_index.time.sleep", sleeps.append)
        wait_until_searchable(hma, indexed, interval=7)
        assert len(calls) == 3 and sleeps == [7, 7]

    def test_a_failed_lookup_is_retried(self, indexed, hma, monkeypatch):
        calls = []
        lookup = hma.match_with_signal_threshold

        def flaky(signal, threshold, signal_type):
            calls.append(1)
            return {"status": "failure", "error": "busy"} if len(calls) == 1 else lookup(signal, threshold, signal_type)

        monkeypatch.setattr(hma, "match_with_signal_threshold", flaky)
        monkeypatch.setattr("hma.video.frame_index.time.sleep", lambda s: None)
        wait_until_searchable(hma, indexed)
        assert len(calls) == 2

    def test_times_out_when_the_frame_never_appears(self, indexed, hma, monkeypatch):
        monkeypatch.setattr(hma, "match_with_signal_threshold", lambda *a: {"status": "success", "matches": []})
        clock = iter(range(0, 1000, 10))
        monkeypatch.setattr("hma.video.frame_index.time.monotonic", lambda: next(clock))
        monkeypatch.setattr("hma.video.frame_index.time.sleep", lambda s: None)
        with pytest.raises(TimeoutError, match="not searchable"):
            wait_until_searchable(hma, indexed, timeout=30)

    def test_an_empty_index_needs_no_wait(self, hma):
        wait_until_searchable(hma, VideoIndex())


def names(matches, max_distance=0.1, **kwargs):
    return [video for video, _ in rank_matches(matches, FRAMES, max_distance, **kwargs)]


def scores(matches, max_distance=0.1):
    return dict(rank_matches(matches, FRAMES, max_distance))


def query_matching(b, c):
    """The lookups of a query whose frames match video b at the distances b, and video c at the distances c."""
    return [[("3", distance)] for distance in b] + [[("4", distance)] for distance in c]


class TestRankMatches:
    def test_each_matched_frame_adds_its_closeness_relative_to_the_distance(self):
        assert scores(query_matching([0.02, 0.06], []) + [[]]) == {"b": pytest.approx(0.8 + 0.4)}

    @pytest.mark.parametrize(
        "better, worse",
        [
            ([0.05] * 8, [0.05] * 3),  # more frames at the same distance
            ([0.02] * 5, [0.06] * 5),  # as many frames, but closer
            ([0.04] * 5, [0.01]),  # several good frames against one very close one
            ([0.02] * 3, [0.09] * 5),  # close frames against more that barely match
        ],
    )
    def test_more_and_closer_frames_rank_first(self, better, worse):
        assert names(query_matching(better, worse)) == ["b", "c"]
        assert names(query_matching(worse, better)) == ["c", "b"]

    def test_videos_that_tie_are_in_name_order(self):
        assert names([[("4", 0.05), ("3", 0.05)]]) == ["b", "c"]

    def test_a_query_frame_counts_once_at_its_closest_frame(self):
        assert scores([[("1", 0.05), ("2", 0.02)]]) == {"a": pytest.approx(0.8)}

    def test_only_matches_within_the_distance_count(self):
        matches = query_matching([0.05, 0.2], [])
        assert scores(matches, 0.1) == {"b": pytest.approx(0.5)}
        assert scores(matches, 0.4) == {"b": pytest.approx(0.875 + 0.5)}

    def test_a_frame_at_exactly_the_distance_matches_and_adds_nothing(self):
        assert scores([[("3", 0.1)]]) == {"b": 0.0}

    def test_the_distance_can_change_the_order(self):
        matches = query_matching([0.01], [0.09] * 5)
        assert names(matches, 0.1) == ["b", "c"] and names(matches, 0.2) == ["c", "b"]

    @pytest.mark.parametrize("max_distance", [0, -0.1])
    def test_the_distance_must_be_positive(self, max_distance):
        with pytest.raises(ValueError, match="positive"):
            rank_matches([], FRAMES, max_distance)

    def test_a_short_clip_is_found_in_a_long_query(self):
        assert names([[]] * 500 + [[("3", 0.01)]]) == ["b"]

    def test_a_video_with_no_match_is_absent(self):
        assert names([[("3", 0.2)]]) == [] and names([]) == [] and names([[]]) == []

    def test_the_query_video_is_never_returned(self):
        assert names([[("1", 0.0), ("3", 0.01)]], exclude_video="a") == ["b"]

    def test_unknown_bank_items_are_ignored(self):
        assert names([[("99", 0.0)]]) == []


class TestClosestDistances:
    def test_each_video_gets_the_distance_of_its_closest_frame(self):
        matches = [("1", 0.08), ("3", 0.30), ("2", 0.02), ("3", 0.10)]
        assert closest_distances(matches, FRAMES) == {"a": 0.02, "b": 0.10}

    def test_the_query_video_is_left_out(self):
        assert closest_distances([("1", 0.0), ("3", 0.2)], FRAMES, exclude_video="a") == {"b": 0.2}

    def test_unknown_bank_items_and_empty_results_are_ignored(self):
        assert closest_distances([("99", 0.0)], FRAMES) == {} and closest_distances([], FRAMES) == {}


def test_a_query_finds_the_video_that_shares_its_scene(tmp_path, hma):
    """End to end: the clip that continues a scene is found and the clip of other scenes is not."""
    first = scenes_video(tmp_path / "first.avi", [(1, 10), (2, 10)])
    same_scene = scenes_video(tmp_path / "same.avi", [(2, 10), (9, 10)])
    other = scenes_video(tmp_path / "other.avi", [(7, 10), (8, 10)])
    index = index_videos(hma, "BANK", [first, same_scene, other], config=CONFIG)
    matches = lookup_frames(hma, index.embeddings[first], 0.05)
    ranked = rank_matches(matches, index.frames, 0.05, exclude_video=first)
    assert ranked == [(same_scene, pytest.approx(1, abs=0.01))]
