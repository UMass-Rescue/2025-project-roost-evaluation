"""Tests for hma.video.evaluate, end to end on synthetic videos against a stand-in for HMA.

There are four series of three clips. Within a series, clip a shows scenes (1, 2), clip b scenes (2, 3)
and clip c scenes (3, 4), so neighbouring clips share a scene and clips a and c share nothing.
Scenes of different series never repeat, so the only matches are inside a series.
"""

import numpy as np
import pytest

from hma.results_utils import read_csv_rows
from hma.video.evaluate import EvaluationConfig, evaluate as run_evaluation
from hma.video.frame_index import index_videos
from hma.video.series_labels_utils import find_video_series, get_video_files
from video_testing import CONFIG, FakeHMA, scenes_video, write_video

CONFIG_SMALL = EvaluationConfig(distance_max=0.2, distance_step=0.1)


@pytest.fixture(autouse=True)
def output(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "run")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return tmp_path / "out" / "evaluation_results" / "run"


def make_series(root, count=4, extra=None):
    for s in range(count):
        base = 10 * (s + 1)
        (root / f"series_{s}").mkdir(parents=True, exist_ok=True)
        for clip, first in zip("abc", (0, 1, 2)):
            scenes_video(root / f"series_{s}" / f"clip_{clip}.avi", [(base + first + 1, 10), (base + first + 2, 10)])
    if extra:
        extra(root)
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_series(tmp_path / "videos")


def run(dataset, config=CONFIG_SMALL):
    videos = get_video_files(dataset)
    series = find_video_series(dataset)
    client = FakeHMA()
    index = index_videos(client, "BANK", videos, config=CONFIG)
    return run_evaluation(client, index, videos, series, config), index


class TestEvaluate:
    def test_writes_every_result(self, dataset, output):
        run(dataset)
        names = {p.name for p in output.iterdir()}
        assert names == {
            "retrieval_sweep_clip_float_results.csv",
            "frame_distance_choice_clip_float_results.csv",
            "map_by_series_score_only_clip_float_results.csv",
            "map_by_series_closest_frame_clip_float_results.csv",
            "match_precision_recall_clip_float_results.csv",
            "match_precision_recall_clip_float_curve.png",
            "match_precision_recall_selected_clip_float_results.csv",
            "match_precision_recall_selected_clip_float_curve.png",
            "pairwise_clip_float_compare.csv",
            "map_by_series_pairwise_clip_float_results.csv",
            "precision_recall_clip_float_results.csv",
            "precision_recall_clip_float_curve.png",
            "distance_distribution_clip_float.png",
        } | {n for n in names if n.startswith("anon")}

    def test_the_sweep_has_a_row_per_distance_ranking_and_series_group(self, dataset, output):
        run(dataset)
        sweep = read_csv_rows(output / "retrieval_sweep_clip_float_results.csv")
        assert len(sweep) == 2 * 2 * 3
        assert {r["frame_distance"] for r in sweep} == {"0.1", "0.2"}
        assert {r["ranking"] for r in sweep} == {"score_only", "closest_frame"}
        assert {r["series"] for r in sweep} == {"all", "train", "test"}
        assert [c for c in sweep[0] if c.startswith("map@")] == [f"map@{k}" for k in range(1, 6)]

    def test_the_first_hit_is_always_right(self, dataset, output):
        """Every query's best match is a clip of its own series, whatever the tie order."""
        run(dataset)
        for row in read_csv_rows(output / "retrieval_sweep_clip_float_results.csv"):
            assert float(row["map@1"]) == 1.0

    def test_the_split_and_the_chosen_distance_are_reported(self, dataset, output):
        chosen, _ = run(dataset)
        choice = read_csv_rows(output / "frame_distance_choice_clip_float_results.csv")
        assert [r["choice"] for r in choice] == ["default", "selected on train"]
        assert choice[0]["frame_distance"] == "0.1" and float(choice[1]["frame_distance"]) == chosen.distance
        assert chosen.distance in CONFIG_SMALL.grid
        assert len(chosen.test_videos) == 6  # two of the four series

    def test_the_per_series_tables_list_every_series_and_a_mean(self, dataset, output):
        run(dataset)
        table = read_csv_rows(output / "map_by_series_closest_frame_clip_float_results.csv")
        assert [r["series"] for r in table][-1] == "mean" and len(table) == 4 + 1

    def test_the_match_curve_reaches_full_precision(self, dataset, output):
        run(dataset)
        curve = read_csv_rows(output / "match_precision_recall_clip_float_results.csv")
        assert max(float(r["precision"]) for r in curve) == 1.0
        assert max(float(r["recall"]) for r in curve) == 1.0

    def test_the_selected_curve_covers_the_test_queries_only(self, dataset, output):
        run(dataset)
        everything = read_csv_rows(output / "match_precision_recall_clip_float_results.csv")
        selected = read_csv_rows(output / "match_precision_recall_selected_clip_float_results.csv")
        positives = lambda curve: max(int(r["tp"]) + int(r["fn"]) for r in curve)
        assert positives(selected) < positives(everything)

    def test_the_pairwise_results_are_anonymised_by_default(self, dataset, output):
        run(dataset)
        text = (output / "pairwise_clip_float_compare.csv").read_text()
        assert "clip_a" not in text and "series_" not in text
        assert len(read_csv_rows(output / "pairwise_clip_float_compare.csv")) == 12 * 11 // 2

    def test_the_pairwise_results_can_show_the_paths(self, dataset, output, monkeypatch):
        monkeypatch.setenv("DEANONYMIZE_IMAGE_PATHS", "1")
        run(dataset)
        pairs = read_csv_rows(output / "pairwise_clip_float_compare.csv")
        assert pairs[0]["video1"].endswith(".avi") and "series_" in pairs[0]["video1"]

    def test_clips_that_share_a_scene_are_closer_than_clips_of_other_series(self, dataset, output, monkeypatch):
        monkeypatch.setenv("DEANONYMIZE_IMAGE_PATHS", "1")
        run(dataset)
        pairs = read_csv_rows(output / "pairwise_clip_float_compare.csv")
        series_of = lambda path: path.split("/")[-2]
        clip_of = lambda path: path.split("/")[-1][5]
        neighbours = [
            float(r["distance"])
            for r in pairs
            if series_of(r["video1"]) == series_of(r["video2"]) and {clip_of(r["video1"]), clip_of(r["video2"])} != {"a", "c"}
        ]
        others = [float(r["distance"]) for r in pairs if series_of(r["video1"]) != series_of(r["video2"])]
        assert len(neighbours) == 4 * 2 and max(neighbours) < min(others)

    def test_a_video_with_no_usable_frame_still_counts_in_the_match_and_pairwise_metrics(self, tmp_path, output):
        flat = lambda root: write_video(root / "series_0" / "clip_d.avi", [np.full((360, 640, 3), 128, np.uint8)] * 20)
        dataset = make_series(tmp_path / "with_flat", extra=flat)
        _, index = run(dataset)
        assert [v.split("/")[-1] for v in index.unusable] == ["clip_d.avi"]
        curve = read_csv_rows(output / "match_precision_recall_clip_float_results.csv")
        # three series of three clips and one of four: 3 * 6 + 12 ordered same-series pairs, the flat clip's included
        assert max(int(r["tp"]) + int(r["fn"]) for r in curve) == 30
        pairwise = read_csv_rows(output / "precision_recall_clip_float_results.csv")
        # 3 series of three clips and one of four: 3 * 3 + 6 same-series pairs, the pairs with the flat clip included
        assert max(int(r["tp"]) + int(r["fn"]) for r in pairwise) == 15
        assert len(read_csv_rows(output / "pairwise_clip_float_compare.csv")) == 12 * 11 // 2  # pairwise covers the 12 clips with frames


class TestEvaluationConfig:
    def test_defaults(self):
        config = EvaluationConfig()
        assert config.frame_distance == 0.10
        assert config.grid == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        assert config.lookup_distance == 0.30
        assert (config.max_k, config.test_fraction, config.seed) == (5, 0.5, 0)

    def test_the_lookup_distance_covers_a_headline_beyond_the_grid(self):
        assert EvaluationConfig(frame_distance=0.4).lookup_distance == 0.4

    @pytest.mark.parametrize(
        "kwargs",
        [{"frame_distance": 0}, {"max_k": 2}, {"max_workers": 0}, {"distance_step": 0}, {"test_fraction": 1.0}],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            EvaluationConfig(**kwargs)

    def test_from_env(self, monkeypatch):
        for name, value in {
            "VIDEO_FRAME_DISTANCE": "0.12",
            "VIDEO_DISTANCE_MAX": "0.4",
            "VIDEO_DISTANCE_STEP": "0.1",
            "MAX_K": "6",
            "VIDEO_TEST_FRACTION": "0.25",
            "VIDEO_SPLIT_SEED": "7",
            "MAX_WORKERS": "2",
        }.items():
            monkeypatch.setenv(name, value)
        assert EvaluationConfig.from_env() == EvaluationConfig(0.12, 0.4, 0.1, 6, 0.25, 7, 2)

    def test_from_env_keeps_defaults_when_unset_or_empty(self, monkeypatch):
        monkeypatch.setenv("VIDEO_FRAME_DISTANCE", "")
        monkeypatch.delenv("MAX_K", raising=False)
        assert EvaluationConfig.from_env() == EvaluationConfig()
