"""Tests for hma.video.pipeline, on synthetic videos with a stand-in embedding function.

A scene is one textured picture held for a while, so it is static (all its frames repeat)
and different scenes differ completely. The stand-in embedding is the frame's tiny grayscale
picture, so equal pictures embed equally and different ones are far apart.
"""

import cv2
import numpy as np
import pytest

from hma.video.dedupe import DedupeConfig
from hma.video.frames import SamplingConfig
from hma.video.pipeline import (
    KeptFrame,
    PipelineConfig,
    PipelineReport,
    encode_jpeg,
    select_video_frames,
)
from video_testing import detailed, scenes_video, write_video

# one frame per second, no scene detection, no neighbouring candidates: easy to count
SIMPLE = SamplingConfig(scene_detector="none", max_frame_gap_seconds=1.0, candidate_window_seconds=0)


def config(**overrides):
    return PipelineConfig(sampling=SIMPLE, **overrides)


class Embedder:
    """Embeds JPEG bytes as their 8x8 grayscale picture, and records how it was called."""

    def __init__(self):
        self.batches = []

    def __call__(self, jpegs):
        self.batches.append(len(jpegs))
        vectors = []
        for jpeg in jpegs:
            gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            vectors.append(small - small.mean())
        return vectors

    @property
    def frames_embedded(self):
        return sum(self.batches)


class TestSelectVideoFrames:
    def test_a_static_video_keeps_one_frame(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 60)])  # 6 s, one picture
        embed = Embedder()
        result = select_video_frames(video, embed, config())
        assert len(result.frames) == 1
        report = result.report
        assert (report.pixel_duplicates, report.embedding_duplicates, report.thinned, report.kept) == (5, 0, 0, 1)
        assert embed.frames_embedded == 1  # the five repeats never cost an embedding

    def test_each_distinct_scene_keeps_one_frame(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 30), (2, 30)])
        result = select_video_frames(video, Embedder(), config())
        assert len(result.frames) == 2
        assert result.frames[0].seconds < 3 <= result.frames[1].seconds
        assert result.report.pixel_duplicates == 4

    def test_embedding_duplicates_are_removed_after_embedding(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 10), (2, 10)])  # different pictures
        embed_calls = []

        def same_vector(jpegs):
            embed_calls.append(len(jpegs))
            return [np.ones(8, dtype=np.float32)] * len(jpegs)

        result = select_video_frames(video, same_vector, config())
        assert len(result.frames) == 1
        assert result.report.pixel_duplicates == 0 and result.report.embedding_duplicates == 1
        assert sum(embed_calls) == 2

    def test_kept_frames_carry_what_the_next_step_needs(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 10)])
        (frame,) = select_video_frames(video, Embedder(), config()).frames
        assert isinstance(frame, KeptFrame)
        decoded = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == detailed(1).shape
        assert frame.embedding.dtype == np.float32 and frame.embedding.shape == (64,)
        assert frame.index == 5 and frame.seconds == pytest.approx(0.5)
        assert not frame.quality.junk

    def test_frames_come_back_in_video_order(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in (1, 2, 3, 4)])
        frames = select_video_frames(video, Embedder(), config()).frames
        assert [f.seconds for f in frames] == sorted(f.seconds for f in frames)
        assert len(frames) == 4

    def test_frames_are_embedded_in_batches(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in range(1, 8)])  # 7 distinct frames
        embed = Embedder()
        result = select_video_frames(video, embed, config(embed_batch_size=3))
        assert embed.batches == [3, 3, 1] and len(result.frames) == 7

    def test_no_cap_by_default(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in range(1, 41)])
        assert len(select_video_frames(video, Embedder(), config()).frames) == 40

    def test_a_cap_thins_evenly_across_the_video(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in range(1, 11)])  # 10 distinct frames, 1 s apart
        result = select_video_frames(video, Embedder(), config(max_frames=4))
        assert [int(f.seconds) for f in result.frames] == [0, 3, 6, 9]  # first and last, evenly between
        assert result.report.thinned == 6 and result.report.kept == 4

    def test_the_cap_applies_to_distinct_frames_not_to_the_clock(self, tmp_path):
        # a long static shot, then three quick distinct scenes: the shot is one frame and uses one slot
        video = scenes_video(tmp_path / "v.avi", [(1, 100), (2, 10), (3, 10), (4, 10)])
        result = select_video_frames(video, Embedder(), config(max_frames=4))
        assert len(result.frames) == 4 and result.report.thinned == 0

    def test_a_video_of_only_junk_yields_nothing_and_says_so(self, tmp_path):
        flat = np.full((360, 640, 3), 128, np.uint8)
        video = write_video(tmp_path / "flat.avi", [flat] * 30)
        embed = Embedder()
        result = select_video_frames(video, embed, config())
        assert result.frames == []
        assert result.report.no_usable_frames and result.report.quality.no_usable_frames
        assert embed.frames_embedded == 0
        assert "NO USABLE FRAMES" in result.report.summary()

    def test_junk_slices_are_dropped_and_the_rest_is_kept(self, tmp_path):
        flat = np.full((360, 640, 3), 0, np.uint8)
        video = write_video(tmp_path / "mixed.avi", [flat] * 10 + [detailed(1)] * 10 + [flat] * 10)
        result = select_video_frames(video, Embedder(), config())
        assert len(result.frames) == 1 and result.report.quality.slices_dropped == 2
        assert not result.report.no_usable_frames

    def test_the_counts_add_up(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 30), (2, 30), (3, 30)])
        report = select_video_frames(video, Embedder(), config(max_frames=2)).report
        selected = report.quality.slices - report.quality.slices_dropped
        assert selected == report.pixel_duplicates + report.embedding_duplicates + report.thinned + report.kept

    def test_the_summary_mentions_every_step(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in range(1, 6)])
        text = select_video_frames(video, Embedder(), config(max_frames=2)).report.summary()
        for part in ("slices", "frames assessed", "pixel", "embedding duplicates", "thinned", "frames kept"):
            assert part in text

    def test_the_embedding_function_must_return_one_vector_per_frame(self, tmp_path):
        video = scenes_video(tmp_path / "v.avi", [(1, 10), (2, 10)])
        with pytest.raises(ValueError, match="vectors"):
            select_video_frames(video, lambda jpegs: [np.ones(4)], config())

    def test_an_unreadable_file(self, tmp_path):
        with pytest.raises(ValueError):
            select_video_frames(tmp_path / "missing.avi", Embedder(), config())

    def test_uses_the_environment_when_no_config_is_given(self, tmp_path, monkeypatch):
        video = scenes_video(tmp_path / "v.avi", [(s, 10) for s in range(1, 6)])
        monkeypatch.setenv("VIDEO_SCENE_DETECTOR", "none")
        monkeypatch.setenv("VIDEO_CANDIDATE_WINDOW_SECONDS", "0")
        monkeypatch.setenv("VIDEO_MAX_FRAMES", "2")
        assert len(select_video_frames(video, Embedder()).frames) == 2


class TestPipelineConfig:
    def test_defaults_have_no_cap(self):
        config = PipelineConfig()
        assert config.max_frames is None
        assert (config.embed_batch_size, config.jpeg_quality) == (32, 90)

    @pytest.mark.parametrize("kwargs", [{"max_frames": 0}, {"embed_batch_size": 0}, {"jpeg_quality": 0}, {"jpeg_quality": 101}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs)

    def test_from_env_reads_the_pipeline_and_every_step(self, monkeypatch):
        monkeypatch.setenv("VIDEO_MAX_FRAMES", "12")
        monkeypatch.setenv("VIDEO_EMBED_BATCH_SIZE", "8")
        monkeypatch.setenv("VIDEO_JPEG_QUALITY", "75")
        monkeypatch.setenv("VIDEO_MAX_FRAME_GAP_SECONDS", "2")
        monkeypatch.setenv("VIDEO_BLUR_MIN", "40")
        monkeypatch.setenv("VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX", "0.05")
        config = PipelineConfig.from_env()
        assert (config.max_frames, config.embed_batch_size, config.jpeg_quality) == (12, 8, 75)
        assert config.sampling.max_frame_gap_seconds == 2.0
        assert config.quality.blur_min == 40.0
        assert config.dedupe == DedupeConfig(embedding_distance_max=0.05)

    @pytest.mark.parametrize("value", ["", "0", "  "])
    def test_an_empty_or_zero_cap_means_no_cap(self, monkeypatch, value):
        monkeypatch.setenv("VIDEO_MAX_FRAMES", value)
        assert PipelineConfig.from_env().max_frames is None


class TestEncodeJpeg:
    def test_round_trips_to_a_similar_image(self):
        image = detailed(3)
        decoded = cv2.imdecode(np.frombuffer(encode_jpeg(image, 95), np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == image.shape
        assert np.abs(decoded.astype(int) - image.astype(int)).mean() < 15

    def test_lower_quality_is_smaller(self):
        image = detailed(3)
        assert len(encode_jpeg(image, 30)) < len(encode_jpeg(image, 95))


def test_an_empty_report_has_no_usable_frames():
    assert PipelineReport().no_usable_frames
