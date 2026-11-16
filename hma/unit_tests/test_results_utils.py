"""Tests for hma.results_utils (result CSV writing, content-id mapping, anonymisation)."""

import csv

import pytest

from hma import results_utils


@pytest.fixture(autouse=True)
def results_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260101_000000")
    monkeypatch.setenv("ANON_ID_MAP_FILEPATH", str(tmp_path / "anon.json"))
    return tmp_path


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_content_id_mapping_round_trips(results_env):
    mapping = {"1": "a.jpg", "2": "b.jpg"}
    path = results_utils.save_content_id_mapping(mapping, "clip_float")
    assert path.parent == results_env / "evaluation_results" / "20260101_000000"
    assert results_utils.load_content_id_mapping(path.parent, "clip_float") == mapping


def test_load_content_id_mapping_missing_file_is_empty(results_env):
    assert results_utils.load_content_id_mapping(results_env, "clip_float") == {}


def test_write_results_topk_rows_are_anonymised_by_default():
    results = [{"image": "/data/a.jpg", "k": 2, "matches": [
        {"bank_content_id": 1, "distance": "0.1"}, {"bank_content_id": 2, "distance": "0.2"}]}]
    out = results_utils.write_results(results, "topk_test_clip_float_results.csv")
    rows = _rows(out)
    assert [r["bank_content_id"] for r in rows] == ["1", "2"]
    assert rows[0]["image"] == "1" and rows[0]["k"] == "2"  # path replaced by a stable id


def test_write_results_keeps_paths_when_deanonymised(monkeypatch):
    monkeypatch.setenv("DEANONYMIZE_IMAGE_PATHS", "1")
    results = [{"image": "/data/a.jpg", "threshold": 0.3, "matches": []},
               {"image": "/data/a.jpg", "threshold": 0.3, "matches": [{"bank_content_id": 5, "distance": "0.0"}]}]
    rows = _rows(results_utils.write_results(results, "threshold_test_clip_float_results.csv"))
    assert rows == [{"image": "/data/a.jpg", "threshold": "0.3", "bank_content_id": "5", "distance": "0.0"}]


def test_write_results_pairwise_unwraps_distance_and_drops_missing():
    results = [
        {"image1": "a.jpg", "image2": "b.jpg", "matched": True, "distance": {"distance": "0.05"}},
        {"image1": "a.jpg", "image2": "c.jpg", "matched": False, "distance": None},
    ]
    rows = _rows(results_utils.write_results(results, "pairwise_clip_float_compare.csv"))
    assert len(rows) == 1
    assert rows[0]["distance"] == "0.05" and rows[0]["matched"] == "True"


def test_anonymisation_ids_are_stable_across_calls():
    first = results_utils.write_results([{"image": "x.jpg", "k": 1, "matches": [{"bank_content_id": 1, "distance": 0}]}], "topk_a.csv")
    second = results_utils.write_results([{"image": "x.jpg", "k": 1, "matches": [{"bank_content_id": 1, "distance": 0}]}], "topk_b.csv")
    assert _rows(first)[0]["image"] == _rows(second)[0]["image"]


def test_write_results_rejects_non_dict_items_when_anonymising():
    with pytest.raises(TypeError):
        results_utils.write_results(["not a dict"], "topk_c.csv")


def test_decode_clip_hex():
    import numpy as np

    vec = np.array([1.0, -2.0], dtype=np.float32)
    assert results_utils.decode_clip_hex(vec.tobytes().hex()) == [1.0, -2.0]
    assert results_utils.decode_clip_hex("zz") is None
