"""Model construction shared by controlled and recovery studies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .compact_timesfm3 import CompactTimesFM3Config, CompactTimesFM3Student
from .student import StudentConfig, TimesFMStudent


def build_student(values: Mapping[str, Any]) -> nn.Module:
    config = dict(values)
    architecture = str(config.get("architecture", "factorized_last_token"))
    if architecture == "compact_timesfm3":
        return CompactTimesFM3Student(CompactTimesFM3Config(**config))
    if architecture == "factorized_last_token":
        config.pop("architecture", None)
        return TimesFMStudent(StudentConfig(**config))
    raise ValueError(f"unsupported student architecture {architecture!r}")
