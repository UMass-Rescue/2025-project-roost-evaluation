"""Tests for hma.video.smoke_test, with a stand-in for HMA."""

import hashlib

import cv2
import numpy as np
import pytest

from hma.results_utils import read_csv_rows, write_csv_rows
from hma.video import evaluate, smoke_test
from hma.video.evaluate import EvaluationConfig
from hma.video.frame_index import encode_signal
from video_testing import FakeHMA


class FakeService(FakeHMA):
    """The fake HMA plus the environment reset and bank calls the evaluation makes.

    It hashes a frame as a pseudo-random unit vector seeded by the frame's coarse content, so frames
    of one scene are identical and frames of different scenes are nearly orthogonal, as they are in CLIP.
    """

    def hash_image_bytes(self, images, signal_type):
        signals = []
        for image in images:
            colour = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            coarse = (cv2.resize(colour, (4, 4), interpolation=cv2.INTER_AREA) // 32).tobytes()
            vector = np.random.default_rng(int.from_bytes(hashlib.sha256(coarse).digest()[:8], "big")).normal(size=64)
            signals.append(encode_signal((vector / np.linalg.norm(vector)).astype(np.float32)))
        return signals

    def cleanup_test_environment(self, signal_type=None):
        return True

    def setup_bank(self, bank_name):
        return True


@pytest.fixture(autouse=True)
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(evaluate, "HMAClient", FakeService)
    monkeypatch.setattr(smoke_test, "setup_logging", lambda *args: None)


def test_passes_on_the_sample_videos(capsys):
    assert smoke_test.main() == 0
    out = capsys.readouterr().out
    assert "$ python -m hma.video.sample_videos" in out and "python hma/video/evaluate.py" in out
    assert "✓ Video smoke test passed: 12 videos, 24 frames" in out


def test_prints_the_commands_that_reproduce_it(capsys, tmp_path):
    smoke_test.main()
    out = capsys.readouterr().out
    assert f"VIDEO_INPUT_DIR={tmp_path / 'out' / 'video_smoke_samples'}" in out and "BANK_NAME=VIDEO_SMOKE_TEST" in out


def test_fails_with_a_nonzero_status_when_a_result_is_wrong(capsys, monkeypatch):
    real_check = smoke_test.check_results
    monkeypatch.setattr(smoke_test, "check_results", lambda summary: real_check(summary) + ["a broken result"])
    assert smoke_test.main() == 1
    assert "✗ Video smoke test failed" in capsys.readouterr().out


class TestCheckResults:
    @pytest.fixture
    def summary(self):
        from pathlib import Path
        import os

        smoke_test.make_sample_videos(Path(os.environ["OUTPUT_DIR"]) / "samples")
        os.environ["TEST_RUN_TIMESTAMP"] = "run"
        return evaluate.run(Path(os.environ["OUTPUT_DIR"]) / "samples", "BANK", EvaluationConfig())

    def test_a_good_run_has_no_problems(self, summary):
        assert smoke_test.check_results(summary) == []

    def test_a_missing_file_is_reported(self, summary):
        (summary["results_dir"] / "pairwise_clip_float_compare.csv").unlink()
        assert smoke_test.check_results(summary) == ["missing pairwise_clip_float_compare.csv"]

    def test_unusable_videos_are_reported(self, summary):
        assert "no usable frame" in smoke_test.check_results({**summary, "unusable": 2})[0]

    def test_a_wrong_first_hit_is_reported(self, summary):
        path = summary["results_dir"] / "retrieval_clip_float_results.csv"
        path.write_text(path.read_text().replace("1.000000", "0.500000"))
        problems = smoke_test.check_results(summary)
        assert problems and all("map@1 is 0.500000" in p for p in problems)

    def test_a_wrong_pairwise_result_is_reported(self, summary):
        path = summary["results_dir"] / "map_by_series_pairwise_clip_float_results.csv"
        path.write_text(path.read_text().replace("1.000000", "0.250000"))
        assert any("pairwise map@1" in p for p in smoke_test.check_results(summary))

    def test_a_match_curve_without_a_precise_point_is_reported(self, summary):
        path = summary["results_dir"] / "match_precision_recall_clip_float_results.csv"
        curve = read_csv_rows(path)
        for row in curve:
            row["precision"] = "0.5"
        write_csv_rows(path, curve)
        assert any("match curve" in p for p in smoke_test.check_results(summary))
