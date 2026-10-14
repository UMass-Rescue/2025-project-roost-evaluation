"""Tests for sentinel.metrics' precision/recall threshold sweep."""

import csv

from sentinel.metrics import compute_precision_recall

# Genuinely separable fake scores: every grooming score is higher than every
# innocuous score, so some threshold classifies all of them correctly.
SEPARABLE_SCORES = [
    (0.9, "grooming"),
    (0.8, "grooming"),
    (0.7, "grooming"),
    (0.6, "grooming"),
    (0.1, "innocuous"),
    (0.2, "innocuous"),
    (0.15, "innocuous"),
    (0.3, "innocuous"),
]


def test_compute_precision_recall_writes_csv(tmp_path):
    output_csv = tmp_path / "pr.csv"
    rows = compute_precision_recall(SEPARABLE_SCORES, str(output_csv))

    assert output_csv.exists()
    assert len(rows) > 0

    with open(output_csv) as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert set(csv_rows[0].keys()) == {
        "threshold",
        "precision",
        "recall",
        "fpr",
        "tp",
        "fp",
        "fn",
        "tn",
    }


def test_compute_precision_recall_writes_plot_when_requested(tmp_path):
    output_csv = tmp_path / "pr.csv"
    output_plot = tmp_path / "pr.png"
    compute_precision_recall(SEPARABLE_SCORES, str(output_csv), str(output_plot))

    assert output_plot.exists()


def test_compute_precision_recall_skips_plot_when_not_requested(tmp_path):
    output_csv = tmp_path / "pr.csv"
    compute_precision_recall(SEPARABLE_SCORES, str(output_csv))

    assert not (tmp_path / "pr.png").exists()


def test_lowest_threshold_has_perfect_recall(tmp_path):
    output_csv_rows = compute_precision_recall(SEPARABLE_SCORES, str(tmp_path / "pr.csv"))
    min_threshold_row = min(output_csv_rows, key=lambda r: r["threshold"])

    # At the lowest observed score, every example is predicted positive.
    assert min_threshold_row["recall"] == 1.0
    assert min_threshold_row["fn"] == 0


def test_highest_threshold_has_zero_false_positives(tmp_path):
    output_csv_rows = compute_precision_recall(SEPARABLE_SCORES, str(tmp_path / "pr.csv"))
    max_threshold_row = max(output_csv_rows, key=lambda r: r["threshold"])

    # At the highest observed score, almost nothing is predicted positive.
    assert max_threshold_row["fp"] == 0


def test_perfectly_separable_scores_reach_perfect_precision_and_recall(tmp_path):
    # A threshold strictly between the two clusters should classify everything correctly.
    rows = compute_precision_recall(SEPARABLE_SCORES, str(tmp_path / "pr.csv"))
    best_row = max(rows, key=lambda r: (r["precision"], r["recall"]))

    assert best_row["precision"] == 1.0
    assert best_row["recall"] == 1.0


def test_empty_input_produces_no_rows(tmp_path):
    output_csv = tmp_path / "pr.csv"
    rows = compute_precision_recall([], str(output_csv))

    assert rows == []
    assert output_csv.exists()  # header-only CSV is still written
