"""Tests for hma.video.frame_index, against a stand-in for the HMA service.

The stand-in hashes a frame as its normalised 8x8 grayscale picture and keeps the bank in memory,
so indexing and lookup run end to end on synthetic videos without a service.
"""

import cv2
import numpy as np
import pytest

from hma.video.frame_index import (
    FrameRef,
    decode_signal,
    encode_signal,
    hma_embedder,
    index_videos,
    lookup_frames,
    video_scores,
)
from hma.video.frames import SamplingConfig
from hma.video.pipeline import PipelineConfig
from video_testing import detailed, scenes_video, write_video

# one frame per second, no scene detection, no neighbouring candidates: easy to count
CONFIG = PipelineConfig(sampling=SamplingConfig(scene_detector="none", max_frame_gap_seconds=1.0, candidate_window_seconds=0))


class FakeHMA:
    def __init__(self):
        self.bank = {}  # content id -> signal
        self.hash_calls = []
        self.fail_adds = False

    def hash_image_bytes(self, images, signal_type):
        self.hash_calls.append((len(images), signal_type))
        signals = []
        for image in images:
            gray = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            small -= small.mean()
            signals.append(encode_signal(small / np.linalg.norm(small)))
        return signals

    def add_signal_to_bank(self, bank_name, signal_type, signal):
        if self.fail_adds:
            return None
        content_id = str(len(self.bank) + 1)
        self.bank[content_id] = signal
        return content_id

    def match_with_signal_threshold(self, signal, threshold, signal_type):
        query = decode_signal(signal)
        matches = []
        for content_id, stored in self.bank.items():
            distance = 1 - float(query @ decode_signal(stored))
            if distance <= threshold:
                matches.append({"bank_content_id": int(content_id), "distance": str(distance)})
        return {"status": "success", "matches": matches}


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


FRAMES = {
    "1": FrameRef("a", 0, 0.5),
    "2": FrameRef("a", 6, 1.5),
    "3": FrameRef("b", 0, 0.5),
    "4": FrameRef("c", 0, 0.5),
}


class TestVideoScores:
    def test_the_score_is_the_share_of_query_frames_that_matched_the_video(self):
        matches = [[("3", 0.01)], [("3", 0.02), ("4", 0.03)], [], [("4", 0.05)]]
        assert video_scores(matches, FRAMES, 0.1) == {"b": 0.5, "c": 0.5}

    def test_several_matching_frames_of_one_video_count_a_query_frame_once(self):
        matches = [[("1", 0.01), ("2", 0.02)]]
        assert video_scores(matches, FRAMES, 0.1) == {"a": 1.0}

    def test_only_matches_within_the_distance_count(self):
        matches = [[("3", 0.05)], [("3", 0.2)]]
        assert video_scores(matches, FRAMES, 0.1) == {"b": 0.5}
        assert video_scores(matches, FRAMES, 0.3) == {"b": 1.0}
        assert video_scores(matches, FRAMES, 0.05) == {"b": 0.5}  # a distance equal to the limit counts

    def test_the_query_video_is_never_scored(self):
        matches = [[("1", 0.0), ("3", 0.01)]]
        assert video_scores(matches, FRAMES, 0.1, exclude_video="a") == {"b": 1.0}

    def test_unknown_bank_items_are_ignored(self):
        assert video_scores([[("99", 0.0)]], FRAMES, 0.1) == {}

    def test_a_query_with_no_frames_scores_nothing(self):
        assert video_scores([], FRAMES, 0.1) == {}

    def test_a_video_with_no_match_is_absent(self):
        assert "c" not in video_scores([[("3", 0.0)]], FRAMES, 0.1)


def test_a_query_finds_the_video_that_shares_its_scene(tmp_path, hma):
    """End to end: the clip that continues a scene scores above the clip of another scene."""
    first = scenes_video(tmp_path / "first.avi", [(1, 10), (2, 10)])
    same_scene = scenes_video(tmp_path / "same.avi", [(2, 10), (9, 10)])
    other = scenes_video(tmp_path / "other.avi", [(7, 10), (8, 10)])
    index = index_videos(hma, "BANK", [first, same_scene, other], config=CONFIG)
    matches = lookup_frames(hma, index.embeddings[first], 0.05)
    scores = video_scores(matches, index.frames, 0.05, exclude_video=first)
    assert scores == {same_scene: 0.5}
