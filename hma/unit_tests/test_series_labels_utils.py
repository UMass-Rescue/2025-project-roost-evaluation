"""Tests for the series label helpers shared by the image and video evaluations."""

import json
from pathlib import Path

import pytest

from hma import series_labels_utils as shared
from hma.image import series_labels_utils as image_labels
from hma.video import series_labels_utils as video_labels


def touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


class TestGetMediaFiles:
    def test_filters_by_extension_case_insensitively(self, tmp_path):
        touch(tmp_path, "a.JPG", "b.png", "c.mp4", "d.txt")
        assert [Path(p).name for p in image_labels.get_image_files(tmp_path)] == ["a.JPG", "b.png"]
        assert [Path(p).name for p in video_labels.get_video_files(tmp_path)] == ["c.mp4"]

    def test_recurses_into_series_folders(self, tmp_path):
        touch(tmp_path, "s1/a.mp4", "s1/b.mp4", "s2/c.mp4", "top.mp4")
        assert [Path(p).name for p in video_labels.get_video_files(tmp_path)] == ["a.mp4", "b.mp4", "c.mp4", "top.mp4"]

    def test_stays_flat_without_series_folders(self, tmp_path):
        touch(tmp_path, "a.mp4", "empty/readme.txt")
        assert [Path(p).name for p in video_labels.get_video_files(tmp_path)] == ["a.mp4"]

    def test_is_sorted_and_unique(self, tmp_path):
        touch(tmp_path, "s/b.mp4", "s/a.mp4")
        files = video_labels.get_video_files(tmp_path)
        assert files == sorted(set(files))


class TestFindSeries:
    def test_one_series_per_folder_with_at_least_two_files(self, tmp_path):
        touch(tmp_path, "s1/a.mp4", "s1/b.mp4", "s2/c.mp4", "s3/d.mp4", "s3/e.mp4", "s3/notes.txt")
        series = video_labels.find_video_series(tmp_path)
        assert list(series) == ["s1", "s3"]
        assert [Path(p).name for p in series["s3"]] == ["d.mp4", "e.mp4"]

    def test_video_paths_are_the_ones_get_video_files_finds(self, tmp_path):
        touch(tmp_path, "s1/a.mp4", "s1/b.mp4")
        found = video_labels.get_video_files(tmp_path)
        assert sum(video_labels.find_video_series(tmp_path).values(), []) == found

    def test_relative_input_keeps_its_form(self, tmp_path, monkeypatch):
        touch(tmp_path, "vids/s1/a.mp4", "vids/s1/b.mp4")
        monkeypatch.chdir(tmp_path)
        assert video_labels.find_video_series(Path("vids"))["s1"] == ["vids/s1/a.mp4", "vids/s1/b.mp4"]
        assert video_labels.get_video_files(Path("vids")) == ["vids/s1/a.mp4", "vids/s1/b.mp4"]

    def test_paths_inside_the_label_root_become_relative(self, tmp_path):
        touch(tmp_path, "vids/s1/a.mp4", "vids/s1/b.mp4")
        series = shared.find_series(tmp_path / "vids", shared.VIDEO_EXTENSIONS, tmp_path)
        assert series == {"s1": ["./vids/s1/a.mp4", "./vids/s1/b.mp4"]}

    def test_paths_outside_the_label_root_stay_absolute(self, tmp_path):
        touch(tmp_path, "s1/a.mp4", "s1/b.mp4")
        series = shared.find_series(tmp_path, shared.VIDEO_EXTENSIONS, tmp_path / "elsewhere")
        assert series["s1"][0] == (tmp_path / "s1/a.mp4").as_posix()

    def test_no_series_is_an_error(self, tmp_path):
        touch(tmp_path, "s1/a.mp4")
        with pytest.raises(ValueError, match="No series"):
            video_labels.find_video_series(tmp_path)


class TestEnsureLabelsFile:
    def build(self, tmp_path, labels_path=None):
        return shared.ensure_labels_file(
            labels_path or tmp_path / "given.json",
            tmp_path / "media",
            shared.IMAGE_EXTENSIONS,
            tmp_path / "labels" / "default.json",
            tmp_path,
        )

    def test_an_existing_labels_file_is_used(self, tmp_path):
        given = tmp_path / "given.json"
        given.write_text("{}")
        assert self.build(tmp_path) == given

    def test_an_existing_default_is_used_when_the_given_one_is_missing(self, tmp_path):
        default = tmp_path / "labels" / "default.json"
        default.parent.mkdir()
        default.write_text("{}")
        assert self.build(tmp_path) == default

    def test_labels_are_built_from_series_folders(self, tmp_path):
        touch(tmp_path, "media/s1/a.png", "media/s1/b.png")
        built = self.build(tmp_path)
        assert built == tmp_path / "labels" / "default.json"
        assert json.loads(built.read_text()) == {"s1": ["./media/s1/a.png", "./media/s1/b.png"]}

    def test_flat_input_without_labels_is_an_error(self, tmp_path):
        touch(tmp_path, "media/a.png")
        with pytest.raises(ValueError, match="appear flat"):
            self.build(tmp_path)


class TestImageLabels:
    def test_the_checked_in_labels_are_found_when_none_are_given(self, tmp_path):
        images = image_labels._IMAGE_ROOT / "resources" / "images"
        found = image_labels.ensure_labels_file(tmp_path / "none.json", images)
        assert found == image_labels._IMAGE_ROOT / "resources/labels/images_series_labels.json"
        assert found.exists()

    def test_a_given_labels_file_wins(self, tmp_path):
        given = tmp_path / "given.json"
        given.write_text("{}")
        assert image_labels.ensure_labels_file(given, tmp_path) == given
