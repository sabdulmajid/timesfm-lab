#!/usr/bin/env python3
"""Summarize measured teacher/student size, latency, throughput, and VRAM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCOPES = ("model_only", "end_to_end")


def _load(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record["status"] != "succeeded":
        raise ValueError(f"benchmark did not succeed: {path}")
    return record


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {result}")
    return result


def _memory(result: dict[str, Any], scope: str) -> dict[str, int]:
    memory = result["memory_bytes"]
    if scope in memory:
        memory = memory[scope]
    return {
        "peak_allocated": int(memory["peak_allocated"]),
        "peak_reserved": int(memory["peak_reserved"]),
    }


def _scope_summary(
    teacher: dict[str, Any], student: dict[str, Any], variant: str, name: str, scope: str
) -> dict[str, float | int]:
    teacher_latency = _finite_positive(
        teacher["latency_ms"][scope]["p50"], f"teacher {name}/{scope} latency"
    )
    student_latency = _finite_positive(
        student["latency_ms"][scope]["p50"], f"student {variant}/{name}/{scope} latency"
    )
    teacher_throughput = _finite_positive(
        teacher["throughput"][scope]["forecast_points_per_second"],
        f"teacher {name}/{scope} throughput",
    )
    student_throughput = _finite_positive(
        student["throughput"][scope]["forecast_points_per_second"],
        f"student {variant}/{name}/{scope} throughput",
    )
    teacher_memory = _memory(teacher, scope)
    student_memory = _memory(student, scope)
    return {
        "teacher_latency_p50_ms": teacher_latency,
        "student_latency_p50_ms": student_latency,
        "student_latency_speedup": teacher_latency / student_latency,
        "teacher_forecast_points_per_second": teacher_throughput,
        "student_forecast_points_per_second": student_throughput,
        "student_throughput_multiple": student_throughput / teacher_throughput,
        "teacher_peak_allocated_bytes": teacher_memory["peak_allocated"],
        "student_peak_allocated_bytes": student_memory["peak_allocated"],
        "student_peak_allocated_fraction": (
            student_memory["peak_allocated"] / teacher_memory["peak_allocated"]
        ),
        "teacher_peak_reserved_bytes": teacher_memory["peak_reserved"],
        "student_peak_reserved_bytes": student_memory["peak_reserved"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = _load(args.teacher)
    students = [_load(path) for path in args.student]
    teacher_parameters = int(teacher["extra"]["model_parameter_count"])
    teacher_results = {row["name"]: row for row in teacher["extra"]["results"]}

    methods: dict[str, Any] = {}
    for student in students:
        variant = str(student["extra"]["variant"])
        if variant in methods:
            raise ValueError(f"duplicate student variant: {variant}")
        student_parameters = int(student["extra"]["model_parameter_count"])
        student_results = {row["name"]: row for row in student["extra"]["results"]}
        if set(student_results) != set(teacher_results):
            raise ValueError(f"benchmark shape mismatch for {variant}")
        shapes: dict[str, Any] = {}
        for name, teacher_row in teacher_results.items():
            student_row = student_results[name]
            shape_fields = ("batch", "variates", "context", "horizon")
            if any(student_row[field] != teacher_row[field] for field in shape_fields):
                raise ValueError(f"benchmark tensor mismatch for {variant}/{name}")
            shapes[name] = {
                "shape": {field: int(teacher_row[field]) for field in shape_fields},
                "measurements": {
                    scope: _scope_summary(teacher_row, student_row, variant, name, scope)
                    for scope in SCOPES
                },
            }
        methods[variant] = {
            "student_parameter_count": student_parameters,
            "teacher_parameter_count": teacher_parameters,
            "teacher_to_student_parameter_ratio": teacher_parameters / student_parameters,
            "student_parameter_reduction_percent": 100.0
            * (1.0 - student_parameters / teacher_parameters),
            "shapes": shapes,
        }

    result = {
        "status": "succeeded",
        "methodology": {
            "metric_direction": (
                "higher speedup/throughput multiple is better; lower memory is better"
            ),
            "teacher": teacher["extra"]["method"],
            "student": students[0]["extra"]["method"],
            "precision": {
                "teacher": teacher["extra"]["precision"],
                "student": students[0]["extra"]["precision"],
            },
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
