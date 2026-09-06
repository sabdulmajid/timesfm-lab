#!/usr/bin/env python3
"""Validate and aggregate isolated teacher/student performance-recovery trials."""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config

SCOPES = ("model_only", "end_to_end")
STATISTICS = ("p50", "p95", "mean")


def _load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("status") != "succeeded":
        raise ValueError(f"benchmark record did not succeed: {path}")
    if record.get("extra", {}).get("protocol_version") != 1:
        raise ValueError(f"unsupported benchmark protocol: {path}")
    record["_path"] = str(path)
    return record


def _distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("latency samples must be finite and positive")
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "samples": samples,
    }


def _geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive inputs")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _coefficient_of_variation(values: list[float]) -> float:
    mean = float(np.mean(values))
    return float(np.std(values, ddof=1) / mean) if len(values) > 1 else 0.0


def _validate_trials(
    records: list[dict[str, Any]], expected_kind: str, expected_trials: int
) -> dict[str, Any]:
    if len(records) != expected_trials:
        raise ValueError(
            f"expected {expected_trials} {expected_kind} trials, received {len(records)}"
        )
    extras = [record["extra"] for record in records]
    if any(extra["model_kind"] != expected_kind for extra in extras):
        raise ValueError(f"non-{expected_kind} record supplied in {expected_kind} group")
    trials = sorted(int(extra["trial"]) for extra in extras)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError(f"{expected_kind} trial identifiers are not 1..{expected_trials}")
    invariant_fields = (
        "config_sha256",
        "target_authority_sha256",
        "suite_input_sha256",
        "label",
        "parameter_count",
        "trainable_parameter_count",
        "checkpoint_sha256",
        "student_config_sha256",
        "student_factory",
        "compile",
        "compile_mode",
        "precision",
        "deployment_implementation",
        "input_policy",
        "covariate_policy",
    )
    for field in invariant_fields:
        values = {json.dumps(extra.get(field), sort_keys=True) for extra in extras}
        if len(values) != 1:
            raise ValueError(f"{expected_kind} trials disagree on {field}: {values}")
    gpu_uuids = {extra["gpu"]["uuid"] for extra in extras}
    if len(gpu_uuids) != 1:
        raise ValueError(f"{expected_kind} trials used different GPUs: {gpu_uuids}")
    process_ids = [int(extra["process_id"]) for extra in extras]
    if len(set(process_ids)) != len(process_ids):
        raise ValueError(f"{expected_kind} trials did not use fresh processes")
    return {
        field: extras[0].get(field) for field in invariant_fields
    } | {
        "gpu": extras[0]["gpu"],
        "process_ids": process_ids,
        "git_commits": sorted({record["git_commit"] for record in records}),
        "record_paths": [record["_path"] for record in records],
    }


def _rows_by_name(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["name"]): row for row in record["extra"]["results"]}
    if len(rows) != len(record["extra"]["results"]):
        raise ValueError(f"duplicate workload in {record['_path']}")
    return rows


def _merged_model(
    records: list[dict[str, Any]], expected_names: list[str]
) -> dict[str, dict[str, Any]]:
    record_rows = [_rows_by_name(record) for record in records]
    if any(list(rows) != expected_names for rows in record_rows):
        raise ValueError("benchmark workload order does not match the frozen configuration")
    merged: dict[str, dict[str, Any]] = {}
    identity_fields = (
        "groups",
        "dataset",
        "term",
        "instance_indices",
        "batch",
        "variates",
        "context",
        "horizon",
        "past_covariates_included",
        "past_covariate_variates",
        "input_sha256",
    )
    for name in expected_names:
        rows = [mapping[name] for mapping in record_rows]
        for field in identity_fields:
            values = {json.dumps(row[field], sort_keys=True) for row in rows}
            if len(values) != 1:
                raise ValueError(f"trials disagree on {name}/{field}")
        hashes = {row["correctness"]["output_sha256"] for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"non-deterministic output hashes across trials for {name}")
        semantic_checks = [row.get("semantic_reference") for row in rows]
        if any(check is not None for check in semantic_checks):
            if not all(
                check is not None and check.get("allclose") is True
                for check in semantic_checks
            ):
                raise ValueError(f"incomplete semantic-reference check for {name}")
            semantic_hashes = {
                check["reference_output_sha256"]
                for check in semantic_checks
                if check is not None
            }
            if len(semantic_hashes) != 1:
                raise ValueError(f"non-deterministic semantic reference for {name}")
            semantic_summary = {
                "allclose": True,
                "maximum_absolute_difference": max(
                    float(check["maximum_absolute_difference"])
                    for check in semantic_checks
                    if check is not None
                ),
                "reference_output_sha256": next(iter(semantic_hashes)),
            }
        else:
            semantic_summary = None
        scopes: dict[str, Any] = {}
        for scope in SCOPES:
            raw_samples = [
                float(value)
                for row in rows
                for value in row["latency_ms"][scope]["samples"]
            ]
            trial_p50 = [float(row["latency_ms"][scope]["p50"]) for row in rows]
            memory_fields = (
                "baseline_allocated",
                "baseline_reserved",
                "peak_allocated",
                "peak_reserved",
                "incremental_peak_allocated",
                "incremental_peak_reserved",
            )
            scopes[scope] = {
                "latency_ms": _distribution(raw_samples),
                "trial_p50_ms": trial_p50,
                "trial_p50_coefficient_of_variation": _coefficient_of_variation(trial_p50),
                "memory_bytes": {
                    field: max(int(row["memory_bytes"][scope][field]) for row in rows)
                    for field in memory_fields
                },
            }
        merged[name] = {
            "shape": {field: rows[0][field] for field in identity_fields if field != "groups"},
            "groups": rows[0]["groups"],
            "output_sha256": next(iter(hashes)),
            "semantic_reference": semantic_summary,
            "cold_latency_ms": {
                "first_end_to_end_samples": [
                    float(row["cold_latency_ms"]["first_end_to_end"]) for row in rows
                ],
                "first_model_only_after_end_to_end_samples": [
                    float(row["cold_latency_ms"]["first_model_only_after_end_to_end"])
                    for row in rows
                ],
            },
            "measurements": scopes,
        }
    return merged


def _memory_ratio(teacher: dict[str, int], student: dict[str, int], field: str) -> float:
    denominator = int(teacher[field])
    if denominator <= 0:
        raise ValueError(f"teacher memory is nonpositive for {field}")
    return int(student[field]) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--teacher", type=Path, action="append", required=True)
    parser.add_argument("--student", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    teacher_records = [_load_record(path) for path in args.teacher]
    student_records = [_load_record(path) for path in args.student]
    for record in teacher_records + student_records:
        if record["model_revision"] != config["model_revision"]:
            raise ValueError(f"model revision mismatch in {record['_path']}")
        if record["dataset_revision"] != config["dataset_revision"]:
            raise ValueError(f"dataset revision mismatch in {record['_path']}")
        for row in record["extra"]["results"]:
            if int(row["warmup_iterations"]) != int(
                config["measurement"]["warmup_iterations"]
            ):
                raise ValueError(f"warmup mismatch in {record['_path']}")
            expected_repeats = int(config["measurement"]["steady_state_repetitions"])
            if int(row["steady_state_repetitions"]) != expected_repeats:
                raise ValueError(f"repeat-count mismatch in {record['_path']}")
            for scope in SCOPES:
                if len(row["latency_ms"][scope]["samples"]) != expected_repeats:
                    raise ValueError(f"raw-sample count mismatch in {record['_path']}")
    expected_trials = int(config["measurement"]["independent_trials"])
    teacher_kind = str(teacher_records[0]["extra"]["model_kind"])
    if teacher_kind not in {"teacher_stock", "teacher_optimized"}:
        raise ValueError(f"invalid teacher reference kind: {teacher_kind}")
    teacher_identity = _validate_trials(teacher_records, teacher_kind, expected_trials)
    student_identity = _validate_trials(student_records, "student", expected_trials)
    if teacher_identity["config_sha256"] != student_identity["config_sha256"]:
        raise ValueError("teacher and student used different frozen configurations")
    if teacher_identity["suite_input_sha256"] != student_identity["suite_input_sha256"]:
        raise ValueError("teacher and student received different target histories")
    if teacher_identity["gpu"]["uuid"] != student_identity["gpu"]["uuid"]:
        raise ValueError("teacher and student did not run on the same physical GPU")

    all_records = teacher_records + student_records
    intervals = sorted(
        (
            record["started_at"],
            record["ended_at"],
            record["extra"]["model_kind"],
            int(record["extra"]["trial"]),
        )
        for record in all_records
    )
    if any(previous[1] > following[0] for previous, following in pairwise(intervals)):
        raise ValueError("teacher/student benchmark processes overlapped")
    observed_order = [item[2] for item in intervals]
    expected_order = [
        str(value)
        for value in config["measurement"]["trial_order"]
        if value in {teacher_kind, "student"}
    ]
    if observed_order != expected_order:
        raise ValueError(f"expected serial order {expected_order}, observed {observed_order}")

    workload_specs = config["workloads"]
    expected_names = [str(item["name"]) for item in workload_specs]
    teacher = _merged_model(teacher_records, expected_names)
    student = _merged_model(student_records, expected_names)
    if teacher_kind == "teacher_optimized" and any(
        teacher[name]["semantic_reference"] is None for name in expected_names
    ):
        raise ValueError("optimized teacher lacks stock-output equivalence evidence")
    workload_results: dict[str, Any] = {}
    for name in expected_names:
        if teacher[name]["shape"] != student[name]["shape"]:
            raise ValueError(f"teacher/student shape or input mismatch for {name}")
        scopes: dict[str, Any] = {}
        for scope in SCOPES:
            teacher_scope = teacher[name]["measurements"][scope]
            student_scope = student[name]["measurements"][scope]
            comparison: dict[str, Any] = {
                "teacher": teacher_scope,
                "student": student_scope,
                "latency_speedup": {
                    statistic: (
                        float(teacher_scope["latency_ms"][statistic])
                        / float(student_scope["latency_ms"][statistic])
                    )
                    for statistic in STATISTICS
                },
                "student_memory_fraction": {
                    field: _memory_ratio(
                        teacher_scope["memory_bytes"], student_scope["memory_bytes"], field
                    )
                    for field in ("peak_allocated", "peak_reserved")
                },
            }
            shape = teacher[name]["shape"]
            teacher_mean_seconds = float(teacher_scope["latency_ms"]["mean"]) / 1000.0
            student_mean_seconds = float(student_scope["latency_ms"]["mean"]) / 1000.0
            points = int(shape["batch"]) * int(shape["variates"]) * int(shape["horizon"])
            comparison["throughput"] = {
                "teacher_forecast_points_per_second": points / teacher_mean_seconds,
                "student_forecast_points_per_second": points / student_mean_seconds,
                "student_multiple": teacher_mean_seconds / student_mean_seconds,
            }
            scopes[scope] = comparison
        workload_results[name] = {
            "shape": teacher[name]["shape"],
            "groups": teacher[name]["groups"],
            "teacher_output_sha256": teacher[name]["output_sha256"],
            "student_output_sha256": student[name]["output_sha256"],
            "cold_latency_ms": {
                "teacher": teacher[name]["cold_latency_ms"],
                "student": student[name]["cold_latency_ms"],
            },
            "measurements": scopes,
        }

    groups = sorted({group for item in workload_specs for group in item["groups"]})
    aggregates: dict[str, Any] = {}
    for group in groups:
        names = [str(item["name"]) for item in workload_specs if group in item["groups"]]
        aggregates[group] = {
            "workloads": names,
            "workload_count": len(names),
            "method": config["measurement"]["aggregate"]["method"],
            "latency_speedup": {
                scope: {
                    statistic: _geometric_mean(
                        [
                            float(
                                workload_results[name]["measurements"][scope][
                                    "latency_speedup"
                                ][statistic]
                            )
                            for name in names
                        ]
                    )
                    for statistic in STATISTICS
                }
                for scope in SCOPES
            },
        }

    aggregate_config = config["measurement"]["aggregate"]
    primary_group = str(aggregate_config["primary_group"])
    primary_scope = str(aggregate_config["primary_scope"])
    primary_statistic = str(aggregate_config["primary_statistic"])
    primary_speedup = float(
        aggregates[primary_group]["latency_speedup"][primary_scope][primary_statistic]
    )
    target = float(config["measurement"]["target_end_to_end_speedup"])
    gating_teacher_kind = f"teacher_{config['teacher']['references']['gating']}"
    gating_reference = teacher_kind == gating_teacher_kind
    result = {
        "status": "succeeded",
        "protocol_version": 1,
        "teacher_reference_kind": teacher_kind,
        "config_path": str(args.config),
        "scope": config["scope"],
        "serial_execution": {
            "verified_no_overlap": True,
            "expected_order": expected_order,
            "observed_intervals": [
                {"started_at": row[0], "ended_at": row[1], "model_kind": row[2], "trial": row[3]}
                for row in intervals
            ],
            "physical_gpu": teacher_identity["gpu"],
        },
        "teacher": teacher_identity,
        "student": student_identity,
        "parameter_comparison": {
            "teacher": int(teacher_identity["parameter_count"]),
            "student": int(student_identity["parameter_count"]),
            "teacher_to_student_ratio": (
                int(teacher_identity["parameter_count"])
                / int(student_identity["parameter_count"])
            ),
        },
        "workloads": workload_results,
        "aggregates": aggregates,
        "speed_target": {
            "definition": {
                "group": primary_group,
                "scope": primary_scope,
                "statistic": primary_statistic,
                "aggregation": aggregate_config["method"],
            },
            "threshold": target,
            "measured_speedup": primary_speedup,
            "gating_reference": gating_reference,
            "passed": gating_reference and primary_speedup >= target,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
