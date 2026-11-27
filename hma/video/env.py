"""Reading settings from environment variables."""

import os
from typing import Callable, TypeVar

T = TypeVar("T")


def env_value(name: str, default, cast: Callable[[str], T] = float):
    """The variable converted with cast, or default when it is unset or empty."""
    value = os.getenv(name, "").strip()
    return cast(value) if value else default
