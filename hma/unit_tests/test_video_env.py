"""Tests for hma.video.env."""

import pytest

from hma.video.env import env_value


def test_converts_the_value(monkeypatch):
    monkeypatch.setenv("SETTING", " 2.5 ")
    assert env_value("SETTING", 1.0) == 2.5
    monkeypatch.setenv("SETTING", "7")
    assert env_value("SETTING", 1, int) == 7
    monkeypatch.setenv("SETTING", "Adaptive")
    assert env_value("SETTING", "x", str) == "Adaptive"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_empty_gives_the_default(monkeypatch, value):
    monkeypatch.delenv("SETTING", raising=False)
    if value is not None:
        monkeypatch.setenv("SETTING", value)
    assert env_value("SETTING", 3.0) == 3.0
    assert env_value("SETTING", None) is None


def test_a_value_that_does_not_convert_is_an_error(monkeypatch):
    monkeypatch.setenv("SETTING", "fast")
    with pytest.raises(ValueError):
        env_value("SETTING", 1.0)
