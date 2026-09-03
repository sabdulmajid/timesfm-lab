#!/usr/bin/env python3
"""Audit exact context/horizon requirements for a configured GIFT-Eval scope."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config


def _distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(array, return_counts=True)
    return {
        "count": len(values),
        "min": int(array.min()),
        "p25": int(np.percentile(array, 25)),
        "p50": int(np.percentile(array, 50)),
        "p75": int(np.percentile(array, 75)),
        "max": int(array.max()),
        "exact_counts": {
            str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    student = load_config(args.student_config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())

    from gift_eval.data import Dataset  # type: ignore[import-untyped]

    configurations = []
    all_contexts: list[int] = []
    all_horizons: list[int] = []
    for item in config["evaluation"]["datasets"]:
        dataset = Dataset(name=str(item["name"]), term=str(item["term"]), to_univariate=False)
        contexts = [
            int(np.asarray(entry["target"]).shape[-1]) for entry in dataset.test_data.input
        ]
        horizon = int(dataset.prediction_length)
        all_contexts.extend(contexts)
        all_horizons.extend([horizon] * len(contexts))
        configurations.append(
            {
                "configuration": f"{item['name']}/{item['term']}",
                "instances": len(contexts),
                "actual_target_variates": int(dataset.target_dim),
                "horizon": horizon,
                "context": _distribution(contexts),
            }
        )
    max_student_horizon = int(student["student"]["max_horizon"])
    max_required_horizon = max(all_horizons)
    result = {
        "status": "success",
        "scope": config["evaluation"]["reportable_scope"],
        "dataset_revision": config["dataset_revision"],
        "configurations": len(configurations),
        "instances": len(all_contexts),
        "horizon_distribution": _distribution(all_horizons),
        "context_distribution": _distribution(all_contexts),
        "target_variate_distribution": dict(
            sorted(Counter(item["actual_target_variates"] for item in configurations).items())
        ),
        "student_support": {
            "max_context": int(student["student"]["max_context"]),
            "max_horizon": max_student_horizon,
            "scope_horizon_supported": max_required_horizon <= max_student_horizon,
        },
        "training_horizon_decision": {
            "maximum": 64,
            "reason": (
                "the complete short-horizon scope requires at most 60 points; training the "
                "teacher's native 64-point output patch permits exact slicing for every shorter "
                "configuration"
            ),
            "medium_long_excluded": (
                "official medium/long horizons are 480-900, beyond the current student's "
                f"max_horizon={max_student_horizon}"
            ),
        },
        "per_configuration": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
