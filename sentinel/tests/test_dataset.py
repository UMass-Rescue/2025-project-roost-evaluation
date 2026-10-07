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
    load_1l_paths,
    load_10l_test_set,
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

    assert len(conversations) == 2
    names = {c.path.name for c in conversations}
    assert names == {"conversation_001.txt", "conversation_002.txt"}

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
