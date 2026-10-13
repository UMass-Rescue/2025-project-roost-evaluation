"""Tests for sentinel.dataset, using a small fake fixture dataset.

The fixture dataset deliberately includes files on disk that are NOT in
labeled_conversations.csv (an unreviewed grooming-folder file, an
unreviewed 10l conversation) to verify the loader only trusts the manifest,
never the raw folder contents.
"""

import os
from pathlib import Path

import pytest

from sentinel.dataset import (
    Conversation,
    DatasetNotFoundError,
    ground_truth_label,
    load_1l_paths,
    load_10l_eval_set,
    load_10l_test_set,
    stage_1l_banks,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "fake_dataset"


def test_load_1l_paths_returns_only_manifest_rows():
    grooming, innocuous = load_1l_paths(dataset_root=FIXTURE_ROOT)

    assert {p.name for p in grooming} == {"fixture_g1.txt", "fixture_g2.txt"}
    assert {p.name for p in innocuous} == {"fixture_i1.txt", "fixture_i2.txt"}

    # The unreviewed file exists on disk but must never be returned.
    assert "fixture_g_excluded.txt" not in {p.name for p in grooming}


def test_load_1l_paths_are_resolvable_files():
    grooming, innocuous = load_1l_paths(dataset_root=FIXTURE_ROOT)

    for path in grooming + innocuous:
        assert path.exists()
        assert path.is_file()


def test_load_10l_test_set_returns_only_manifest_rows():
    conversations = load_10l_test_set(dataset_root=FIXTURE_ROOT)

    assert len(conversations) == 5
    names = {c.path.name for c in conversations}
    assert names == {
        "conversation_001.txt",
        "conversation_002.txt",
        "conversation_003.txt",
        "conversation_004.txt",
        "conversation_005.txt",
    }

    # The unreviewed conversation exists on disk but must never be returned.
    assert "conversation_201.txt" not in names


def test_load_10l_test_set_fields():
    conversations = load_10l_test_set(dataset_root=FIXTURE_ROOT)
    by_name = {c.path.name: c for c in conversations}

    grooming_convo = by_name["conversation_001.txt"]
    assert grooming_convo.split == "10l"
    assert grooming_convo.generation_label == "grooming"
    assert grooming_convo.human_classification == "definitely grooming"

    innocuous_convo = by_name["conversation_002.txt"]
    assert innocuous_convo.generation_label == "innocuous"
    assert innocuous_convo.human_classification == "definitely innocuous"


def test_conversation_lines_reads_file_content():
    conversations = load_10l_test_set(dataset_root=FIXTURE_ROOT)
    by_name = {c.path.name: c for c in conversations}

    lines = by_name["conversation_001.txt"].lines()
    assert lines == [
        "FIXTURE_SPEAKER_A: fixture placeholder line 1",
        "FIXTURE_SPEAKER_B: fixture placeholder line 2",
        "FIXTURE_SPEAKER_A: fixture placeholder line 3",
    ]


def test_conversation_lines_missing_file_raises():
    missing = Conversation(
        path=FIXTURE_ROOT / "conversations" / "synth_chat_snippets_10l" / "does_not_exist.txt",
        split="10l",
        generation_label="grooming",
        human_classification="definitely grooming",
    )
    with pytest.raises(DatasetNotFoundError):
        missing.lines()


def test_missing_dataset_root_raises_clear_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="labeled_conversations.csv"):
        load_1l_paths(dataset_root=tmp_path)


def test_sentinel_dataset_dir_env_var_override(monkeypatch):
    monkeypatch.setenv("SENTINEL_DATASET_DIR", str(FIXTURE_ROOT))

    # No explicit dataset_root passed -- must fall back to the env var.
    grooming, innocuous = load_1l_paths()
    assert len(grooming) == 2
    assert len(innocuous) == 2


def test_sentinel_dataset_dir_env_var_nonexistent_path_raises(monkeypatch):
    monkeypatch.setenv("SENTINEL_DATASET_DIR", "/nonexistent/path/for/sure")

    with pytest.raises(DatasetNotFoundError):
        load_1l_paths()


def _conversation(human_classification: str) -> Conversation:
    return Conversation(
        path=Path("/fake/path.txt"),
        split="10l",
        generation_label="grooming",
        human_classification=human_classification,
    )


@pytest.mark.parametrize(
    "human_classification,expected_lenient,expected_strict",
    [
        ("definitely grooming", "grooming", "grooming"),
        ("probably grooming", "grooming", None),
        ("too difficult to tell", None, None),
        ("probably innocuous", "innocuous", None),
        ("definitely innocuous", "innocuous", "innocuous"),
    ],
)
def test_ground_truth_label_certainty_thresholds(
    human_classification, expected_lenient, expected_strict
):
    conversation = _conversation(human_classification)
    assert ground_truth_label(conversation, certainty_threshold="lenient") == expected_lenient
    assert ground_truth_label(conversation, certainty_threshold="strict") == expected_strict


def test_ground_truth_label_defaults_to_lenient(monkeypatch):
    monkeypatch.delenv("SENTINEL_CERTAINTY_THRESHOLD", raising=False)
    conversation = _conversation("probably grooming")
    assert ground_truth_label(conversation) == "grooming"


def test_ground_truth_label_reads_env_var(monkeypatch):
    monkeypatch.setenv("SENTINEL_CERTAINTY_THRESHOLD", "strict")
    conversation = _conversation("probably grooming")
    assert ground_truth_label(conversation) is None


def test_ground_truth_label_unknown_threshold_raises():
    conversation = _conversation("definitely grooming")
    with pytest.raises(ValueError, match="Unknown certainty threshold"):
        ground_truth_label(conversation, certainty_threshold="made_up")


def test_load_10l_eval_set_lenient_excludes_only_too_difficult():
    paired = load_10l_eval_set(dataset_root=FIXTURE_ROOT, certainty_threshold="lenient")
    names_to_labels = {c.path.name: label for c, label in paired}

    assert names_to_labels == {
        "conversation_001.txt": "grooming",
        "conversation_002.txt": "innocuous",
        "conversation_003.txt": "grooming",
        "conversation_004.txt": "innocuous",
    }
    assert "conversation_005.txt" not in names_to_labels  # too difficult to tell


def test_load_10l_eval_set_strict_excludes_probable_rows_too():
    paired = load_10l_eval_set(dataset_root=FIXTURE_ROOT, certainty_threshold="strict")
    names_to_labels = {c.path.name: label for c, label in paired}

    assert names_to_labels == {
        "conversation_001.txt": "grooming",
        "conversation_002.txt": "innocuous",
    }


def test_stage_1l_banks_copies_confirmed_files_only(tmp_path):
    positive_dir, negative_dir = stage_1l_banks(data_dir=tmp_path, dataset_root=FIXTURE_ROOT)

    assert {p.name for p in positive_dir.iterdir()} == {"fixture_g1.txt", "fixture_g2.txt"}
    assert {p.name for p in negative_dir.iterdir()} == {"fixture_i1.txt", "fixture_i2.txt"}

    # Staged copies are real files, not symlinks pointing outside the staging dir.
    staged_file = positive_dir / "fixture_g1.txt"
    assert staged_file.read_text() == (
        FIXTURE_ROOT
        / "conversations"
        / "synth_chat_snippets_1l"
        / "grooming"
        / "fixture_g1.txt"
    ).read_text()


def test_stage_1l_banks_limit_caps_each_folder(tmp_path):
    positive_dir, negative_dir = stage_1l_banks(
        data_dir=tmp_path, dataset_root=FIXTURE_ROOT, limit=1
    )

    assert len(list(positive_dir.iterdir())) == 1
    assert len(list(negative_dir.iterdir())) == 1


def test_stage_1l_banks_clears_previous_contents(tmp_path):
    positive_dir, _ = stage_1l_banks(data_dir=tmp_path, dataset_root=FIXTURE_ROOT, limit=1)
    assert len(list(positive_dir.iterdir())) == 1

    # Re-staging without limit should replace, not accumulate.
    positive_dir, _ = stage_1l_banks(data_dir=tmp_path, dataset_root=FIXTURE_ROOT)
    assert len(list(positive_dir.iterdir())) == 2


def test_stage_1l_banks_data_dir_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path))
    positive_dir, negative_dir = stage_1l_banks(dataset_root=FIXTURE_ROOT)

    assert positive_dir == tmp_path / "texts" / "positive"
    assert len(list(positive_dir.iterdir())) == 2
    assert len(list(negative_dir.iterdir())) == 2


def test_stage_1l_banks_missing_data_dir_env_var_raises(monkeypatch):
    monkeypatch.delenv("SENTINEL_DATA_DIR", raising=False)
    with pytest.raises(DatasetNotFoundError, match="SENTINEL_DATA_DIR"):
        stage_1l_banks(dataset_root=FIXTURE_ROOT)
