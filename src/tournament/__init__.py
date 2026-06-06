"""Tournament module helpers."""

from __future__ import annotations

from typing import Any


def _load_config() -> dict[str, Any]:
    import yaml

    with open("config.yaml") as f:
        return yaml.safe_load(f)
