"""Tests for hma.video.pairwise."""

import numpy as np
import pytest

from hma.video.pairwise import distance_pairs, video_distances


def frames(*angles):
    """Unit vectors in 2D at the given angles in degrees."""
    radians = np.radians(angles)
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


class TestVideoDistances:
    def test_identical_videos_are_at_distance_zero(self):
        videos, matrix = video_distances({"a": frames(0, 90), "b": frames(0, 90)})
        assert videos == ["a", "b"] and matrix[0, 1] == pytest.approx(0, abs=1e-12)

    def test_is_the_mean_closest_frame_distance_in_both_directions(self):
        a = frames(0, 90)
        b = frames(0)
        _, matrix = video_distances({"a": a, "b": b})
        a_to_b = (0 + (1 - np.cos(np.radians(90)))) / 2  # a's frames: 0 deg is at 0, 90 deg is 1 away
        b_to_a = 0.0  # b's only frame has an identical frame in a
        assert matrix[0, 1] == pytest.approx((a_to_b + b_to_a) / 2)

    def test_is_symmetric_with_a_zero_diagonal(self):
        rng = np.random.default_rng(0)
        embeddings = {name: rng.normal(size=(n, 16)) for name, n in (("a", 3), ("b", 1), ("c", 4))}
        _, matrix = video_distances(embeddings)
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0, atol=1e-12)

    def test_vectors_need_not_be_unit_length(self):
        _, scaled = video_distances({"a": frames(0, 40) * 5, "b": frames(10) * 0.1})
        _, unit = video_distances({"a": frames(0, 40), "b": frames(10)})
        assert np.allclose(scaled, unit)

    def test_the_videos_are_sorted_and_frames_of_different_counts_line_up(self):
        videos, matrix = video_distances({"c": frames(0), "a": frames(0, 90, 180), "b": frames(90)})
        assert videos == ["a", "b", "c"]
        assert matrix[videos.index("a"), videos.index("b")] < matrix[videos.index("b"), videos.index("c")] + 1  # sanity
        assert matrix[0, 1] == pytest.approx((((1 - 0) + 0 + 1) / 3 + 0) / 2)  # a's frames: 1, 0, 1 away from b

    def test_no_videos_is_an_error(self):
        with pytest.raises(ValueError):
            video_distances({})

    def test_a_zero_embedding_is_an_error(self):
        with pytest.raises(ValueError, match="zero"):
            video_distances({"a": np.zeros((1, 4)), "b": np.ones((1, 4))})


def test_distance_pairs_lists_each_pair_once():
    matrix = np.array([[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]])
    assert list(distance_pairs(["a", "b", "c"], matrix)) == [("a", "b", 0.1), ("a", "c", 0.2), ("b", "c", 0.3)]
