from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load YAML and apply ``section.key=value`` overrides."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The top-level configuration must be a mapping.")
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        cursor = config
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override inside non-mapping key {key!r}.")
            cursor = child
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

