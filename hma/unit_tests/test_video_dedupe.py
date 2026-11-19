"""Tests for hma.video.dedupe."""

import numpy as np
import pytest

from hma.video.dedupe import DedupeConfig, FrameDeduper, thin_evenly, thumbnail


def unit(*values):
    return np.array(values, dtype=np.float32)


class TestDedupeConfig:
    def test_defaults(self):
        config = DedupeConfig()
        assert (config.pixel_diff_max, config.embedding_distance_max, config.thumbnail_size) == (2.0, 0.02, 16)

    @pytest.mark.parametrize(
        "kwargs", [{"pixel_diff_max": -1}, {"embedding_distance_max": -0.1}, {"thumbnail_size": 1}]
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            DedupeConfig(**kwargs)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEO_DEDUPE_PIXEL_DIFF_MAX", "4")
        monkeypatch.setenv("VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX", "0.05")
        assert DedupeConfig.from_env() == DedupeConfig(pixel_diff_max=4.0, embedding_distance_max=0.05)

    def test_from_env_keeps_defaults_when_unset_or_empty(self, monkeypatch):
        monkeypatch.delenv("VIDEO_DEDUPE_PIXEL_DIFF_MAX", raising=False)
        monkeypatch.setenv("VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX", "")
        assert DedupeConfig.from_env() == DedupeConfig()


class TestThumbnail:
    def test_is_a_flat_float_vector_of_size_squared(self):
        image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        thumb = thumbnail(image, 16)
        assert thumb.shape == (256,) and thumb.dtype == np.float32

    def test_gray_and_colour_input_agree_for_a_gray_picture(self):
        gray = np.random.default_rng(1).integers(0, 256, (48, 64), dtype=np.uint8)
        colour = np.stack([gray] * 3, axis=-1)
        assert np.allclose(thumbnail(gray), thumbnail(colour), atol=1.0)

    def test_does_not_depend_on_resolution(self):
        small = np.random.default_rng(2).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        large = np.kron(small, np.ones((4, 4, 1), dtype=np.uint8))
        assert np.abs(thumbnail(small) - thumbnail(large)).mean() < 1.0


class TestPixelDuplicate:
    def test_the_first_frame_is_new_and_an_identical_one_repeats_it(self):
        deduper = FrameDeduper()
        thumb = np.full(256, 100.0, dtype=np.float32)
        assert deduper.pixel_duplicate(thumb) is False
        assert deduper.pixel_duplicate(thumb.copy()) is True

    def test_a_small_difference_is_a_duplicate_and_a_large_one_is_not(self):
        deduper = FrameDeduper(DedupeConfig(pixel_diff_max=2.0))
        base = np.full(256, 100.0, dtype=np.float32)
        deduper.pixel_duplicate(base)
        assert deduper.pixel_duplicate(base + 1.5) is True
        assert deduper.pixel_duplicate(base + 5.0) is False

    def test_a_frame_is_compared_with_every_earlier_frame_not_just_the_last(self):
        deduper = FrameDeduper()
        a, b = np.full(256, 20.0, dtype=np.float32), np.full(256, 200.0, dtype=np.float32)
        assert [deduper.pixel_duplicate(x) for x in (a, b, a.copy())] == [False, False, True]

    def test_the_threshold_is_exclusive(self):
        deduper = FrameDeduper(DedupeConfig(pixel_diff_max=2.0))
        base = np.zeros(256, dtype=np.float32)
        deduper.pixel_duplicate(base)
        assert deduper.pixel_duplicate(base + 2.0) is False  # a mean difference of exactly 2.0 is not below it


class TestEmbeddingDuplicate:
    def test_identical_embeddings_repeat(self):
        deduper = FrameDeduper()
        assert deduper.embedding_duplicate(unit(1, 0, 0)) is False
        assert deduper.embedding_duplicate(unit(1, 0, 0)) is True

    def test_scale_does_not_matter(self):
        deduper = FrameDeduper()
        deduper.embedding_duplicate(unit(1, 2, 3))
        assert deduper.embedding_duplicate(unit(10, 20, 30)) is True

    def test_close_is_a_duplicate_and_far_is_not(self):
        deduper = FrameDeduper(DedupeConfig(embedding_distance_max=0.05))
        deduper.embedding_duplicate(unit(1, 0, 0))
        near = unit(1, 0.1, 0)  # cosine distance about 0.005
        far = unit(1, 1, 0)  # cosine distance about 0.29
        assert deduper.embedding_duplicate(near) is True
        assert deduper.embedding_duplicate(far) is False

    def test_a_frame_is_compared_with_every_kept_frame(self):
        deduper = FrameDeduper()
        a, b = unit(1, 0, 0), unit(0, 1, 0)
        assert [deduper.embedding_duplicate(x) for x in (a, b, a.copy(), b.copy())] == [False, False, True, True]

    def test_a_duplicate_is_not_remembered(self):
        # c is within the threshold of b but not of a, and b was dropped, so c is new
        deduper = FrameDeduper(DedupeConfig(embedding_distance_max=0.02))
        a = unit(1, 0)
        b = unit(np.cos(0.15), np.sin(0.15))  # cosine distance from a: about 0.011, a duplicate
        c = unit(np.cos(0.30), np.sin(0.30))  # about 0.011 from b but 0.045 from a
        assert deduper.embedding_duplicate(a) is False
        assert deduper.embedding_duplicate(b) is True
        assert deduper.embedding_duplicate(c) is False

    def test_a_zero_embedding_is_an_error(self):
        with pytest.raises(ValueError):
            FrameDeduper().embedding_duplicate(unit(0, 0, 0))

    def test_still_finds_the_first_frame_after_many_others(self):
        # more frames than the initial capacity, so the storage has to grow
        deduper = FrameDeduper(DedupeConfig(pixel_diff_max=0.0, embedding_distance_max=0.01))
        rng = np.random.default_rng(0)
        vectors = [rng.normal(size=64).astype(np.float32) for _ in range(100)]
        assert not any(deduper.embedding_duplicate(v) for v in vectors)
        assert deduper.embedding_duplicate(vectors[0]) is True
        assert deduper.embedding_duplicate(vectors[99]) is True


class TestThinEvenly:
    def test_no_cap_or_few_items_returns_everything(self):
        items = list("abcdef")
        assert thin_evenly(items, None) == items
        assert thin_evenly(items, 6) == items
        assert thin_evenly(items, 20) == items

    def test_spreads_over_the_whole_sequence_including_both_ends(self):
        assert thin_evenly(list(range(10)), 4) == [0, 3, 6, 9]
        assert thin_evenly(list(range(101)), 5) == [0, 25, 50, 75, 100]

    def test_the_result_is_ordered_unique_and_the_right_size(self):
        for n in range(2, 60):
            for m in range(2, n):
                result = thin_evenly(list(range(n)), m)
                assert len(result) == m and result == sorted(set(result))
                assert result[0] == 0 and result[-1] == n - 1

    def test_a_cap_of_one_takes_the_middle(self):
        assert thin_evenly(list(range(9)), 1) == [4]

    def test_a_cap_below_one_is_an_error(self):
        with pytest.raises(ValueError):
            thin_evenly([1, 2, 3], 0)

    def test_does_not_change_its_input(self):
        items = list(range(10))
        thin_evenly(items, 3)
        assert items == list(range(10))
