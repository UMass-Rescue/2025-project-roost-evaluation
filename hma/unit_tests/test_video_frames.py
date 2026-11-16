"""Tests for hma.video.frames, using small synthetic videos generated on the fly.

Each video is a sequence of flat colour blocks, so a scene cut is an unambiguous
colour change and the expected frame positions can be worked out by hand.
"""

import inspect
import random

import cv2
import numpy as np
import pytest

from hma.video.frames import (
    SamplingConfig,
    candidate_frame_indices,
    detect_scenes,
    iter_frames,
    probe_video,
    sample_frames,
)

RED = (0, 0, 255)  # BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
FPS = 10
SIZE = (64, 48)  # width, height


def make_video(path, segments, fps=FPS, size=SIZE):
    """Write a video made of (bgr_colour, frame_count) blocks and return its path as a string.

    MJPG in an .avi container is used because OpenCV can write it with both its macOS
    (AVFoundation) and Linux (FFmpeg) backends; mp4v/.mp4 is not available on macOS."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV cannot write MJPG/avi video in this environment"
    for colour, count in segments:
        frame = np.full((size[1], size[0], 3), colour, np.uint8)
        for _ in range(count):
            writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def two_scenes(tmp_path):
    """3 s of red then 3 s of blue at 10 fps: 60 frames, one cut at frame 30."""
    return make_video(tmp_path / "two.avi", [(RED, 30), (BLUE, 30)])


class TestEnvironment:
    def test_opencv_is_older_than_4_13(self):
        # PySceneDetect 0.7.3 crashes on OpenCV 4.14 and 5.0 (NaN video position at the end of
        # the stream); 4.12 works. See the pin in requirements.txt before raising this.
        major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
        assert (major, minor) < (4, 13)


class TestSamplingConfig:
    def test_defaults(self):
        config = SamplingConfig()
        assert (config.scene_detector, config.scene_threshold) == ("adaptive", None)
        assert (config.min_scene_seconds, config.max_frame_gap_seconds) == (1.0, 1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scene_detector": "histogram"},
            {"min_scene_seconds": 0},
            {"max_frame_gap_seconds": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            SamplingConfig(**kwargs)

    def test_from_env_reads_every_setting(self, monkeypatch):
        monkeypatch.setenv("VIDEO_SCENE_DETECTOR", "Adaptive")
        monkeypatch.setenv("VIDEO_SCENE_THRESHOLD", "4.5")
        monkeypatch.setenv("VIDEO_MIN_SCENE_SECONDS", "0.5")
        monkeypatch.setenv("VIDEO_MAX_FRAME_GAP_SECONDS", "2")
        assert SamplingConfig.from_env() == SamplingConfig("adaptive", 4.5, 0.5, 2.0)

    def test_from_env_ignores_unset_and_empty_values(self, monkeypatch):
        for name in ("VIDEO_SCENE_DETECTOR", "VIDEO_MIN_SCENE_SECONDS", "VIDEO_MAX_FRAME_GAP_SECONDS"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VIDEO_SCENE_THRESHOLD", "  ")
        assert SamplingConfig.from_env() == SamplingConfig()

    def test_from_env_rejects_a_bad_detector(self, monkeypatch):
        monkeypatch.setenv("VIDEO_SCENE_DETECTOR", "nope")
        with pytest.raises(ValueError):
            SamplingConfig.from_env()


class TestProbe:
    def test_reads_rate_count_and_size(self, tmp_path):
        info = probe_video(make_video(tmp_path / "v.avi", [(RED, 30)]))
        assert info.fps == pytest.approx(FPS)
        assert info.frame_count == 30
        assert (info.width, info.height) == SIZE
        assert info.duration_seconds == pytest.approx(3.0)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError):
            probe_video(tmp_path / "missing.avi")

    def test_file_that_is_not_a_video(self, tmp_path):
        bogus = tmp_path / "bogus.avi"
        bogus.write_text("this is not a video")
        with pytest.raises(ValueError):
            probe_video(bogus)


class TestCandidateFrameIndices:
    def test_three_second_scene_gets_three_midpoints(self):
        assert candidate_frame_indices([(0, 18)], fps=6, max_frame_gap_seconds=1.0) == [3, 9, 15]

    def test_scene_shorter_than_the_gap_gets_its_midpoint(self):
        assert candidate_frame_indices([(0, 2)], fps=10, max_frame_gap_seconds=1.0) == [1]
        assert candidate_frame_indices([(40, 43)], fps=10, max_frame_gap_seconds=1.0) == [41]

    def test_each_scene_is_sampled_on_its_own(self):
        assert candidate_frame_indices([(0, 30), (30, 60)], 10, 1.0) == [5, 15, 25, 35, 45, 55]

    def test_a_larger_gap_gives_fewer_frames(self):
        assert candidate_frame_indices([(0, 30)], fps=10, max_frame_gap_seconds=2.0) == [7, 22]

    def test_just_over_a_multiple_of_the_gap_adds_a_slice(self):
        assert len(candidate_frame_indices([(0, 31)], fps=10, max_frame_gap_seconds=1.0)) == 4

    def test_empty_and_inverted_scenes_are_skipped(self):
        assert candidate_frame_indices([(5, 5), (9, 3)], 10, 1.0) == []

    def test_indices_are_sorted_unique_and_inside_their_scene(self):
        rng = random.Random(0)
        for _ in range(200):
            starts = sorted(rng.sample(range(0, 5000), 4))
            scenes = [(s, s + rng.randint(1, 400)) for s in starts]
            scenes = [(s, min(e, nxt)) for (s, e), nxt in zip(scenes, starts[1:] + [10**9])]
            gap = rng.choice([0.5, 1.0, 2.5])
            indices = candidate_frame_indices(scenes, rng.choice([6, 24, 29.97]), gap)
            assert indices == sorted(set(indices))
            for index in indices:
                assert any(s <= index < e for s, e in scenes)
            # every scene gets at least one frame
            for s, e in scenes:
                assert any(s <= i < e for i in indices)


class TestDetectScenes:
    def test_a_hard_cut_is_found(self, two_scenes):
        info = probe_video(two_scenes)
        scenes = detect_scenes(two_scenes, info, SamplingConfig())
        assert len(scenes) == 2
        assert scenes[0][0] == 0 and scenes[-1][1] == 60
        assert abs(scenes[0][1] - 30) <= 1 and scenes[0][1] == scenes[1][0]

    def test_content_detector_also_finds_a_hard_cut(self, two_scenes):
        scenes = detect_scenes(two_scenes, probe_video(two_scenes), SamplingConfig(scene_detector="content"))
        assert len(scenes) == 2

    @pytest.mark.parametrize(
        "detector, class_name, option, expected",
        [("adaptive", "AdaptiveDetector", "adaptive_threshold", 2.0), ("content", "ContentDetector", "threshold", 27.0)],
    )
    def test_each_detector_gets_its_own_default_threshold(self, two_scenes, monkeypatch, detector, class_name, option, expected):
        import scenedetect

        seen = {}
        real = getattr(scenedetect, class_name)

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(scenedetect, class_name, spy)
        detect_scenes(two_scenes, probe_video(two_scenes), SamplingConfig(scene_detector=detector))
        assert seen[option] == expected
        assert seen["min_scene_len"] == 10  # 1.0 s at 10 fps

    def test_an_explicit_threshold_overrides_the_default(self, two_scenes, monkeypatch):
        import scenedetect

        seen = {}
        real = scenedetect.AdaptiveDetector
        monkeypatch.setattr(scenedetect, "AdaptiveDetector", lambda **kw: (seen.update(kw), real(**kw))[1])
        detect_scenes(two_scenes, probe_video(two_scenes), SamplingConfig(scene_threshold=3.5))
        assert seen["adaptive_threshold"] == 3.5

    def test_no_cut_is_one_scene(self, tmp_path):
        path = make_video(tmp_path / "still.avi", [(RED, 30)])
        assert detect_scenes(path, probe_video(path), SamplingConfig()) == [(0, 30)]

    def test_detector_none_is_one_scene_without_scenedetect(self, two_scenes):
        scenes = detect_scenes(two_scenes, probe_video(two_scenes), SamplingConfig(scene_detector="none"))
        assert scenes == [(0, 60)]

    def test_a_high_threshold_suppresses_the_cut(self, two_scenes):
        scenes = detect_scenes(two_scenes, probe_video(two_scenes), SamplingConfig(scene_threshold=1000.0))
        assert scenes == [(0, 60)]

    def test_a_cut_inside_the_minimum_scene_length_is_ignored(self, tmp_path):
        # red lasts 0.5 s, shorter than the 1.0 s minimum, so the red-to-blue cut does not split
        path = make_video(tmp_path / "short_first.avi", [(RED, 5), (BLUE, 25)])
        assert len(detect_scenes(path, probe_video(path), SamplingConfig(min_scene_seconds=1.0))) == 1
        assert len(detect_scenes(path, probe_video(path), SamplingConfig(min_scene_seconds=0.3))) == 2


class TestIterFrames:
    def test_yields_the_requested_frames_lazily_and_in_order(self, two_scenes):
        frames = iter_frames(two_scenes, [45, 5, 5, 25], FPS)
        assert inspect.isgenerator(frames)
        got = list(frames)
        assert [f.index for f in got] == [5, 25, 45]
        assert [f.seconds for f in got] == pytest.approx([0.5, 2.5, 4.5])
        assert got[0].image.shape == (SIZE[1], SIZE[0], 3)

    def test_frames_have_the_colour_of_their_scene(self, two_scenes):
        red_frame, blue_frame = (f.image for f in iter_frames(two_scenes, [10, 50], FPS))
        assert red_frame[..., 2].mean() > 200 and red_frame[..., 0].mean() < 60  # BGR
        assert blue_frame[..., 0].mean() > 200 and blue_frame[..., 2].mean() < 60

    def test_indices_past_the_end_are_dropped(self, two_scenes):
        assert [f.index for f in iter_frames(two_scenes, [5, 500], FPS)] == [5]

    def test_unreadable_file(self, tmp_path):
        with pytest.raises(ValueError):
            list(iter_frames(tmp_path / "missing.avi", [0], FPS))


class TestSampleFrames:
    def test_samples_each_scene(self, two_scenes):
        frames = list(sample_frames(two_scenes, SamplingConfig()))
        assert [f.index for f in frames] == [5, 15, 25, 35, 45, 55]
        assert all(f.image[..., 2].mean() > 150 for f in frames[:3])  # red scene
        assert all(f.image[..., 0].mean() > 150 for f in frames[3:])  # blue scene

    def test_detector_none_samples_uniformly(self, tmp_path):
        path = make_video(tmp_path / "three.avi", [(RED, 20), (BLUE, 20), (GREEN, 20)])
        frames = list(sample_frames(path, SamplingConfig(scene_detector="none", max_frame_gap_seconds=2.0)))
        assert [f.index for f in frames] == [10, 30, 50]

    def test_uses_the_environment_when_no_config_is_given(self, two_scenes, monkeypatch):
        monkeypatch.setenv("VIDEO_SCENE_DETECTOR", "none")
        monkeypatch.setenv("VIDEO_MAX_FRAME_GAP_SECONDS", "3")
        assert [f.index for f in sample_frames(two_scenes)] == [15, 45]

    def test_a_short_clip_gets_its_frames(self, tmp_path):
        # like the 3 s, 6 fps series clips: one scene, three frames
        path = make_video(tmp_path / "clip.avi", [(GREEN, 18)], fps=6)
        assert [f.index for f in sample_frames(path, SamplingConfig())] == [3, 9, 15]

    def test_unreadable_file(self, tmp_path):
        with pytest.raises(ValueError):
            sample_frames(tmp_path / "missing.avi", SamplingConfig())
