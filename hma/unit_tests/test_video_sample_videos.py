"""Tests for hma.video.sample_videos."""

import cv2
import numpy as np
import pytest

from hma.video import sample_videos
from hma.video.pipeline import PipelineConfig, select_video_frames
from hma.video.sample_videos import make_sample_videos, scene_image
from hma.video.series_labels_utils import find_video_series


def gray_embedding(jpegs):
    """Embeds JPEG bytes as their 8x8 grayscale picture, centred so that different pictures are far apart."""
    vectors = []
    for jpeg in jpegs:
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        vectors.append(small - small.mean())
    return vectors


def first_frames(path):
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(frame)


class TestSceneImage:
    def test_the_same_scene_is_always_the_same_picture(self):
        assert np.array_equal(scene_image(3), scene_image(3))

    def test_different_scenes_differ_clearly(self):
        assert np.abs(scene_image(1).astype(int) - scene_image(2).astype(int)).mean() > 20

    def test_has_the_size_of_a_video_frame(self):
        assert scene_image(0).shape == (sample_videos.HEIGHT, sample_videos.WIDTH, 3)


class TestMakeSampleVideos:
    def test_writes_three_clips_for_each_series(self, tmp_path):
        paths = make_sample_videos(tmp_path, series=4)
        assert len(paths) == 12
        series = find_video_series(tmp_path)
        assert sorted(series) == [f"series_{i:04d}" for i in range(1, 5)]
        assert all(len(clips) == 3 for clips in series.values())

    def test_the_same_call_writes_the_same_files(self, tmp_path):
        first = make_sample_videos(tmp_path / "one")
        second = make_sample_videos(tmp_path / "two")
        for a, b in zip(first, second):
            assert open(a, "rb").read() == open(b, "rb").read()

    def test_neighbouring_clips_of_a_series_share_a_scene_and_other_series_share_none(self, tmp_path):
        paths = make_sample_videos(tmp_path)
        clip = {p.rsplit("/", 2)[-2] + "/" + p.rsplit("/", 1)[-1]: first_frames(p) for p in paths}
        seconds = sample_videos.FPS  # frames per scene
        a, b, c = (clip[f"series_0001/clip_00{i}.avi"] for i in (1, 2, 3))
        distance = lambda x, y: np.abs(x.astype(int) - y.astype(int)).mean()
        assert distance(a[-1], b[0]) < 3  # a's second scene is b's first (up to video compression)
        assert distance(b[-1], c[0]) < 3
        assert distance(a[0], c[0]) > 20 and distance(a[0], a[-1]) > 20
        other = clip["series_0002/clip_001.avi"]
        assert min(distance(x, y) for x in (a[0], a[-1], c[-1]) for y in (other[0], other[-1])) > 20
        assert len(a) == 2 * seconds

    def test_the_pipeline_keeps_one_frame_of_each_scene(self, tmp_path):
        """The picture passes every quality filter and the two scenes of a clip are not duplicates."""
        for path in make_sample_videos(tmp_path, series=2):
            result = select_video_frames(path, gray_embedding, PipelineConfig.from_env())
            assert len(result.frames) == 2 and result.report.quality.slices_dropped == 0
            assert result.frames[0].seconds < 1 <= result.frames[1].seconds

    def test_more_series_get_more_folders(self, tmp_path):
        assert len(make_sample_videos(tmp_path, series=6)) == 18
