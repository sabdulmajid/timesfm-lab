"""Strict helpers for loading committed experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration cannot be validated."""


REQUIRED_PROVENANCE = frozenset({"seed", "model_revision", "dataset_revision"})


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and enforce the shared provenance fields."""

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    missing = REQUIRED_PROVENANCE.difference(config)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ConfigError(f"{path} is missing required field(s): {fields}")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ConfigError("seed must be an integer")
    for field in ("model_revision", "dataset_revision"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ConfigError(f"{field} must be a non-empty string")
    return config
