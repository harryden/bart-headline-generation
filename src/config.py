"""Configuration loading for experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/default.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a JSON experiment config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_config_path(path: str | None) -> Path:
    """Return the explicit config path or the project default."""
    return Path(path) if path else DEFAULT_CONFIG_PATH
