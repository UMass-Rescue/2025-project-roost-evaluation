"""Tests for the in-memory metric entry points in hma.metrics.

The pairwise/CSV entry points and the in-memory ones must agree, since the
former now delegate to the latter.
"""

import csv
import json

import pytest

from hma.metrics.common import (
    classify_pairs_by_series,
    classify_pairwise_by_series,
)
from hma.metrics.distance_distribution import compute_distance_distribution_from_labeled
from hma.metrics.map import (
    build_rankings,
    compute_map_from_pairwise,
    compute_map_from_rankings,
)
from hma.metrics.precision_recall import (
    compute_precision_recall_from_labeled,
    compute_precision_recall_from_pairwise,
)

SERIES = {"A": {"a1", "a2", "a3"}, "B": {"b1", "b2"}}


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pairwise_entries():
    # a-items are close to each other, b-items are close to each other, cross pairs are far
    return [
        {"image1": "a1", "image2": "a2", "distance": 0.10},
        {"image1": "a1", "image2": "a3", "distance": 0.20},
        {"image1": "a2", "image2": "a3", "distance": 0.15},
        {"image1": "b1", "image2": "b2", "distance": 0.05},
        {"image1": "a1", "image2": "b1", "distance": 0.80},
        {"image1": "a1", "image2": "b2", "distance": 0.85},
        {"image1": "a2", "image2": "b1", "distance": 0.70},
        {"image1": "a2", "image2": "b2", "distance": 0.90},
        {"image1": "a3", "image2": "b1", "distance": 0.60},
        {"image1": "a3", "image2": "b2", "distance": 0.95},
    ]


def _write_labels_and_pairwise(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({s: sorted(v) for s, v in SERIES.items()}))
    pairwise = tmp_path / "pairwise.csv"
    with open(pairwise, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image1", "image2", "distance", "matched"])
        w.writeheader()
        for e in _pairwise_entries():
            w.writerow({**e, "matched": False})
    return str(labels), str(pairwise)


class TestClassifyPairs:
    def test_labels_same_and_different_series(self):
        pairs = [("a1", "a2", 0.1), ("a1", "b1", 0.8)]
        assert classify_pairs_by_series(SERIES, pairs) == [(0.1, True), (0.8, False)]

    def test_skips_items_outside_every_series_and_normalises_paths(self):
        pairs = [("./a1", "/a2", 0.1), ("a1", "unlabelled", 0.4)]
        assert classify_pairs_by_series(SERIES, pairs) == [(0.1, True)]

    def test_items_can_be_taken_exactly_as_they_are(self):
        series = {"s": {"./a/1.mp4", "./a/2.mp4"}}
        pairs = [("./a/1.mp4", "./a/2.mp4", 0.2)]
        assert classify_pairs_by_series(series, pairs, normalize=str) == [(0.2, True)]
        assert classify_pairs_by_series(series, pairs) == []  # the default would have rewritten the paths

    def test_pairwise_entries_version_agrees_and_applies_the_anon_map(self):
        entries = [{"image1": "1", "image2": "2", "distance": "0.25"}]
        by_entries = classify_pairwise_by_series(SERIES, entries, {"1": "a1", "2": "b2"})
        assert by_entries == classify_pairs_by_series(SERIES, [("a1", "b2", 0.25)]) == [(0.25, False)]


class TestMapFromRankings:
    def test_perfect_rankings_give_map_one(self, tmp_path):
        rankings = {
            "a1": [("a2", 0.1), ("a3", 0.2), ("b1", 0.8), ("b2", 0.9)],
            "a2": [("a1", 0.1), ("a3", 0.15), ("b1", 0.7), ("b2", 0.9)],
            "a3": [("a2", 0.15), ("a1", 0.2), ("b1", 0.6), ("b2", 0.95)],
            "b1": [("b2", 0.05), ("a3", 0.6), ("a2", 0.7), ("a1", 0.8)],
            "b2": [("b1", 0.05), ("a1", 0.85), ("a2", 0.9), ("a3", 0.95)],
        }
        out = tmp_path / "map.csv"
        compute_map_from_rankings(SERIES, rankings, str(out))
        rows = {r["series"]: r for r in _rows(out)}
        assert float(rows["A"]["map@1"]) == pytest.approx(1.0)
        assert float(rows["B"]["map@1"]) == pytest.approx(1.0)
        assert float(rows["mean"]["map@1"]) == pytest.approx(1.0)

    def test_a_wrong_first_neighbour_lowers_map_at_one(self, tmp_path):
        rankings = {
            "b1": [("a1", 0.1), ("b2", 0.2)],  # nearest is wrong
            "b2": [("b1", 0.05)],
        }
        out = tmp_path / "map.csv"
        compute_map_from_rankings({"B": {"b1", "b2"}}, rankings, str(out))
        row = _rows(out)[0]
        assert float(row["map@1"]) == pytest.approx(0.5)  # b1 misses at rank 1, b2 hits

    def test_the_querys_own_entry_is_ignored(self, tmp_path):
        rankings = {"b1": [("b1", 0.0), ("b2", 0.2)], "b2": [("b2", 0.0), ("b1", 0.2)]}
        out = tmp_path / "map.csv"
        compute_map_from_rankings({"B": {"b1", "b2"}}, rankings, str(out))
        assert float(_rows(out)[0]["map@1"]) == pytest.approx(1.0)

    def test_matches_the_pairwise_entry_point(self, tmp_path):
        labels, pairwise = _write_labels_and_pairwise(tmp_path)
        via_pairwise = tmp_path / "via_pairwise.csv"
        compute_map_from_pairwise(labels, pairwise, str(via_pairwise))
        via_rankings = tmp_path / "via_rankings.csv"
        compute_map_from_rankings(SERIES, build_rankings(_pairwise_entries()), str(via_rankings))
        assert via_pairwise.read_text() == via_rankings.read_text()


class TestPrecisionRecallFromLabeled:
    def test_sweep_reaches_full_recall_and_counts_pairs(self, tmp_path):
        labeled = classify_pairs_by_series(
            SERIES, [(e["image1"], e["image2"], e["distance"]) for e in _pairwise_entries()]
        )
        out = tmp_path / "pr.csv"
        compute_precision_recall_from_labeled(labeled, str(out))
        rows = _rows(out)
        first, last = rows[0], rows[-1]
        assert (int(first["tp"]), int(first["fp"])) == (1, 0)  # only the closest pair is under the lowest threshold
        assert float(first["precision"]) == 1.0
        # 4 same-series pairs (3 in A, 1 in B), 6 cross pairs
        assert int(last["tp"]) == 4 and int(last["fp"]) == 6
        assert float(last["recall"]) == 1.0 and float(last["fpr"]) == 1.0

    def test_matches_the_pairwise_entry_point(self, tmp_path):
        labels, pairwise = _write_labels_and_pairwise(tmp_path)
        via_pairwise = tmp_path / "via_pairwise.csv"
        compute_precision_recall_from_pairwise(labels, pairwise, str(via_pairwise))
        labeled = classify_pairwise_by_series(SERIES, _pairwise_entries())
        via_labeled = tmp_path / "via_labeled.csv"
        compute_precision_recall_from_labeled(labeled, str(via_labeled))
        assert via_pairwise.read_text() == via_labeled.read_text()

    def test_no_pairs_writes_only_a_header(self, tmp_path):
        out = tmp_path / "pr.csv"
        compute_precision_recall_from_labeled([], str(out))
        assert _rows(out) == []


def test_distance_distribution_splits_same_and_different(tmp_path):
    labeled = [(0.1, True), (0.2, True), (0.9, False)]
    same, diff = compute_distance_distribution_from_labeled(labeled, str(tmp_path / "dist.png"))
    assert same == [0.1, 0.2] and diff == [0.9]
