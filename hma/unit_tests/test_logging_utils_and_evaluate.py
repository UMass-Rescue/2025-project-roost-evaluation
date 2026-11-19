"""Tests for hma.logging_utils and for the names hma.evaluate keeps exposing.

The evaluation suites under hma/tests/ import these names from hma.evaluate,
so moving the code into shared modules must not break those imports.
"""

import logging

import pytest

from hma import logging_utils
from hma.hma_client import HMAClient


@pytest.fixture(autouse=True)
def reset_logging_state():
    logging_utils.logger = None
    logging_utils.log_file = None
    yield
    logging_utils.logger = None
    logging_utils.log_file = None
    logging.getLogger("evaluation").handlers.clear()


def test_get_logger_without_setup_is_silent():
    logger = logging_utils.get_logger()
    assert logger.name == "evaluation"
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert logging_utils.get_log_file() is None


def test_setup_logging_writes_a_log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    logging_utils.setup_logging("unit")
    logging_utils._log_info("hello from the test")
    log_file = logging_utils.get_log_file()
    assert log_file.parent == tmp_path / "test_run_logs"
    assert log_file.name.startswith("unit_")
    for handler in logging_utils.get_logger().handlers:
        handler.flush()
    assert "hello from the test" in log_file.read_text()


def test_image_evaluate_still_exposes_what_the_suites_import():
    from hma.image.evaluate import Evaluator, get_logger, _log_debug, _log_info, _log_warning  # noqa: F401

    assert issubclass(Evaluator, HMAClient)
    assert callable(get_logger)
    assert "match_uploaded_files" in Evaluator.__dict__
    # the shared client methods (including the DB reset) are inherited, not redefined
    for name in ("hash_local_content_batch", "create_fresh_database", "cleanup_test_environment"):
        assert hasattr(Evaluator, name)
        assert name not in Evaluator.__dict__
