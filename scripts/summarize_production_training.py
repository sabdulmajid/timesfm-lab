#!/usr/bin/env python3
"""Verify matched production training and summarize validation learning curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


VARIANTS = ("gt", "kd", "dual_view", "cvrd")


def _load(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record["status"] != "succeeded":
        raise ValueError(f"training did not succeed: {path}")
    if record["extra"]["variant"] not in VARIANTS:
        raise ValueError(f"unsupported variant in {path}")
    return record


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _equal(records: dict[str, dict[str, Any]], field: str, values: dict[str, Any]) -> Any:
    canonical = {_canonical(value) for value in values.values()}
    if len(canonical) != 1:
        raise ValueError(f"matched-control violation for {field}: {values}")
    return next(iter(values.values()))


def _percent(left: float, right: float) -> float:
    return 100.0 * (left / right - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [_load(path) for path in args.result]
    records = {record["extra"]["variant"]: record for record in loaded}
    if set(records) != set(VARIANTS) or len(loaded) != len(VARIANTS):
        raise ValueError(f"expected exactly {VARIANTS}, received {tuple(records)}")

    common = {
        "seed": _equal(records, "seed", {key: value["seed"] for key, value in records.items()}),
        "model_revision": _equal(
            records,
            "model_revision",
            {key: value["model_revision"] for key, value in records.items()},
        ),
        "dataset_revision": _equal(
            records,
            "dataset_revision",
            {key: value["dataset_revision"] for key, value in records.items()},
        ),
        "initialization_sha256": _equal(
            records,
            "initialization_sha256",
            {key: value["extra"]["initialization_sha256"] for key, value in records.items()},
        ),
        "training_sequence_sha256": _equal(
            records,
            "rank0_training_sequence_sha256",
            {
                key: value["extra"]["rank0_training_sequence_sha256"]
                for key, value in records.items()
            },
        ),
        "steps": _equal(
            records,
            "training.steps",
            {key: value["extra"]["training"]["steps"] for key, value in records.items()},
        ),
        "windows_processed": _equal(
            records,
            "training.windows_processed",
            {
                key: value["extra"]["training"]["windows_processed"]
                for key, value in records.items()
            },
        ),
        "optimizer": _equal(
            records,
            "training.optimizer",
            {key: value["extra"]["training"]["optimizer"] for key, value in records.items()},
        ),
        "schedule": _equal(
            records,
            "training.schedule",
            {key: value["extra"]["training"]["schedule"] for key, value in records.items()},
        ),
        "precision": _equal(
            records,
            "training.precision",
            {key: value["extra"]["training"]["precision"] for key, value in records.items()},
        ),
        "layout": _equal(
            records,
            "training.layout",
            {key: value["extra"]["training"]["layout"] for key, value in records.items()},
        ),
        "world_size": _equal(
            records,
            "training.world_size",
            {key: value["extra"]["training"]["world_size"] for key, value in records.items()},
        ),
        "datasets": _equal(
            records,
            "datasets",
            {key: value["extra"]["datasets"] for key, value in records.items()},
        ),
    }
    if int(common["steps"]) != 200_000:
        raise ValueError(f"production comparison requires 200000 steps, got {common['steps']}")

    methods: dict[str, Any] = {}
    for variant in VARIANTS:
        record = records[variant]
        curve = record["extra"]["learning_curve"]
        if [int(row["step"]) for row in curve] != list(range(10_000, 200_001, 10_000)):
            raise ValueError(f"incomplete or irregular learning curve for {variant}")
        best = min(curve, key=lambda row: float(row["validation"]["student_pinball"]))
        final = curve[-1]
        methods[variant] = {
            "parameter_count": int(record["extra"]["parameter_count"]),
            "loss_weights": record["extra"]["training"]["loss_weights"],
            "best_checkpoint": record["extra"]["best_checkpoint"],
            "best_step": int(best["step"]),
            "best_windows_processed": int(best["windows_processed"]),
            "best_validation_pinball": float(best["validation"]["student_pinball"]),
            "final_validation_pinball": float(final["validation"]["student_pinball"]),
            "teacher_validation_pinball": float(final["validation"]["teacher_pinball"]),
            "student_minus_teacher_percent_at_best": _percent(
                float(best["validation"]["student_pinball"]),
                float(best["validation"]["teacher_pinball"]),
            ),
            "elapsed_seconds": float(record["extra"]["training"]["elapsed_seconds"]),
            "windows_per_second": float(record["extra"]["training"]["windows_per_second"]),
            "learning_curve": [
                {
                    "step": int(row["step"]),
                    "windows_processed": int(row["windows_processed"]),
                    "validation_pinball": float(row["validation"]["student_pinball"]),
                }
                for row in curve
            ],
        }

    dual = methods["dual_view"]["best_validation_pinball"]
    cvrd = methods["cvrd"]["best_validation_pinball"]
    result = {
        "status": "succeeded",
        "metric_direction": "lower is better",
        "matched_control_audit": {"status": "passed", **common},
        "methods": methods,
        "critical_comparisons": {
            "cvrd_minus_dual_view_percent": _percent(cvrd, dual),
            "cvrd_improves_over_dual_view": bool(cvrd < dual),
            "interpretation": (
                "CVRD adds held-out pretrain forecast accuracy beyond the matched Dual-View control"
                if cvrd < dual
                else "the explicit CVRD response term does not add held-out pretrain forecast accuracy beyond Dual-View"
            ),
        },
    }
    if not all(math.isfinite(method["best_validation_pinball"]) for method in methods.values()):
        raise ValueError("non-finite validation result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
