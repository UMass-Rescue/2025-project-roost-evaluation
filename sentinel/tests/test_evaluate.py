"""Tests for sentinel.evaluate, mocking the Sentinel API's HTTP layer.

These never make real network calls -- SentinelEvaluator's HTTP methods are
exercised against a mocked `requests` session, and _score_conversations()
is exercised against a stub evaluator. The live end-to-end path (actually
calling a running sentinel-app) is verified manually, not in this suite.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from sentinel.dataset import Conversation
from sentinel.evaluate import SentinelEvaluator, _score_conversations


def _conversation(tmp_path, path_name: str, lines: list) -> Conversation:
    """A real Conversation backed by an actual temp file (it's frozen, so
    .lines() can't be monkeypatched -- write real content instead)."""
    file_path = tmp_path / path_name
    file_path.write_text("\n".join(lines))
    return Conversation(
        path=file_path,
        split="10l",
        generation_label="grooming",
        human_classification="definitely grooming",
    )


def _mock_response(json_data: dict, ok: bool = True, status_code: int = 200):
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = str(json_data)
    response.raise_for_status.return_value = None
    return response


@patch("sentinel.evaluate.requests.get")
def test_health_calls_health_endpoint(mock_get):
    mock_get.return_value = _mock_response({"status": "healthy", "banks_loaded": True})

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    result = evaluator.health()

    mock_get.assert_called_once_with("http://fake-host:8000/health")
    assert result == {"status": "healthy", "banks_loaded": True}


@patch("sentinel.evaluate.requests.post")
def test_create_banks_sends_expected_payload(mock_post):
    mock_post.return_value = _mock_response(
        {"success": True, "positive_count": 5, "negative_count": 5}
    )

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    result = evaluator.create_banks("/data/texts/positive", "/data/texts/negative", "/data/banks/x")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://fake-host:8000/banks/create"
    assert kwargs["json"] == {
        "positive_folder": "/data/texts/positive",
        "negative_folder": "/data/texts/negative",
        "output_path": "/data/banks/x",
        "model_name": "all-MiniLM-L6-v2",
    }
    assert result["success"] is True


@patch("sentinel.evaluate.requests.post")
def test_create_banks_raises_on_failure_response(mock_post):
    mock_post.return_value = _mock_response({}, ok=False, status_code=500)

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    with pytest.raises(RuntimeError, match="Failed to create banks"):
        evaluator.create_banks("/pos", "/neg", "/out")


@patch("sentinel.evaluate.requests.post")
def test_score_sends_texts_and_optional_params(mock_post):
    mock_post.return_value = _mock_response(
        {"rare_class_affinity_score": 0.42, "observation_scores": {}, "num_observations": 2}
    )

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    result = evaluator.score(["hello", "world"], top_k=3, min_score_to_consider=0.2)

    args, kwargs = mock_post.call_args
    assert args[0] == "http://fake-host:8000/score"
    assert kwargs["json"] == {
        "texts": ["hello", "world"],
        "top_k": 3,
        "min_score_to_consider": 0.2,
    }
    assert result["rare_class_affinity_score"] == 0.42


@patch("sentinel.evaluate.requests.post")
def test_score_omits_optional_params_when_not_given(mock_post):
    mock_post.return_value = _mock_response(
        {"rare_class_affinity_score": 0.1, "observation_scores": {}, "num_observations": 1}
    )

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    evaluator.score(["hello"])

    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"texts": ["hello"]}


@patch("sentinel.evaluate.requests.post")
def test_score_raises_on_failure_response(mock_post):
    mock_post.return_value = _mock_response({}, ok=False, status_code=503)

    evaluator = SentinelEvaluator(base_url="http://fake-host:8000")
    with pytest.raises(RuntimeError, match="Failed to score conversation"):
        evaluator.score(["hello"])


class _StubEvaluator:
    """Stand-in for SentinelEvaluator that returns a fixed score per call."""

    def __init__(self, score: float):
        self.score_value = score
        self.calls = []

    def score(self, texts, top_k=None, min_score_to_consider=None):
        self.calls.append(texts)
        return {"rare_class_affinity_score": self.score_value, "observation_scores": {}, "num_observations": len(texts)}


def test_score_conversations_pairs_scores_with_labels(tmp_path):
    conversations_with_labels = [
        (_conversation(tmp_path, "a.txt", ["line1", "line2"]), "grooming"),
        (_conversation(tmp_path, "b.txt", ["line1"]), "innocuous"),
    ]
    evaluator = _StubEvaluator(score=0.5)

    labeled_scores = _score_conversations(evaluator, conversations_with_labels)

    assert labeled_scores == [(0.5, "grooming"), (0.5, "innocuous")]
    assert len(evaluator.calls) == 2


def test_score_conversations_skips_empty_conversations(tmp_path):
    conversations_with_labels = [
        (_conversation(tmp_path, "empty.txt", []), "grooming"),
        (_conversation(tmp_path, "b.txt", ["line1"]), "innocuous"),
    ]
    evaluator = _StubEvaluator(score=0.5)

    labeled_scores = _score_conversations(evaluator, conversations_with_labels)

    assert labeled_scores == [(0.5, "innocuous")]
    assert len(evaluator.calls) == 1


def _reload_evaluate_module():
    """sentinel.evaluate reads SENTINEL_SCORE_TOP_K/MIN_SCORE_TO_CONSIDER at
    import time, so env var changes need a reload to take effect."""
    import sentinel.evaluate as evaluate_module

    return importlib.reload(evaluate_module)


@pytest.fixture(autouse=True)
def _restore_evaluate_module_after_reload_tests(monkeypatch):
    """Reloading sentinel.evaluate mutates shared module-level globals
    (SCORE_TOP_K, MIN_SCORE_TO_CONSIDER). Reload it back to a clean,
    env-unset state after every test in this file so state never leaks
    across tests regardless of run order."""
    yield
    monkeypatch.delenv("SENTINEL_SCORE_TOP_K", raising=False)
    monkeypatch.delenv("SENTINEL_MIN_SCORE_TO_CONSIDER", raising=False)
    _reload_evaluate_module()


def test_score_params_unset_by_default(monkeypatch):
    monkeypatch.delenv("SENTINEL_SCORE_TOP_K", raising=False)
    monkeypatch.delenv("SENTINEL_MIN_SCORE_TO_CONSIDER", raising=False)
    module = _reload_evaluate_module()

    assert module.SCORE_TOP_K is None
    assert module.MIN_SCORE_TO_CONSIDER is None


def test_score_params_read_from_env_vars(monkeypatch):
    monkeypatch.setenv("SENTINEL_SCORE_TOP_K", "10")
    monkeypatch.setenv("SENTINEL_MIN_SCORE_TO_CONSIDER", "0.05")
    module = _reload_evaluate_module()

    assert module.SCORE_TOP_K == 10
    assert module.MIN_SCORE_TO_CONSIDER == 0.05


def test_score_params_min_score_can_be_explicitly_zero(monkeypatch):
    monkeypatch.setenv("SENTINEL_MIN_SCORE_TO_CONSIDER", "0.0")
    monkeypatch.delenv("SENTINEL_SCORE_TOP_K", raising=False)
    module = _reload_evaluate_module()

    assert module.MIN_SCORE_TO_CONSIDER == 0.0
    assert "0.0" in module._score_params_description()
