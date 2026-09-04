#!/usr/bin/env python3
"""Build exact lower-is-better teacher/student GIFT-Eval comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME

METRICS = (MASE_NAME, MWQL_NAME)


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric aggregation requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _scope(record: dict[str, Any]) -> str:
    explicit = record["extra"].get("scope")
    if explicit is not None:
        return str(explicit)
    count = len(record["extra"]["results"])
    inferred = {
        19: "complete_short_horizon_multivariate_subset",
        55: "complete_short_horizon_gift_eval",
    }
    if count not in inferred:
        raise ValueError(f"cannot infer evaluation scope from {count} configurations")
    return inferred[count]


def _method(record: dict[str, Any]) -> str:
    variant = record["extra"].get("variant")
    return f"student_{variant}" if variant is not None else "timesfm3_teacher"


def _run_summary(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "succeeded" or record["extra"].get("failures"):
        raise ValueError(f"evaluation did not fully succeed: {path}")
    rows = record["extra"]["results"]
    aggregate = {
        metric: _geometric_mean([float(row[metric]) for row in rows]) for metric in METRICS
    }
    return {
        "path": str(path.resolve()),
        "run_id": record["run_id"],
        "method": _method(record),
        "mode": str(record["extra"]["mode"]),
        "scope": _scope(record),
        "seed": int(record["seed"]),
        "model_revision": record["model_revision"],
        "dataset_revision": record["dataset_revision"],
        "parameter_count": record["extra"].get("parameter_count"),
        "checkpoint_sha256": record["extra"].get("checkpoint_sha256"),
        "aggregation": "unweighted geometric mean across configurations",
        "aggregate": aggregate,
        "per_configuration": rows,
    }


def _percent_difference(left: float, right: float) -> float:
    return 100.0 * (left / right - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [_run_summary(path, json.loads(path.read_text())) for path in args.result]
    keyed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for run in runs:
        key = (run["scope"], run["method"], run["mode"], run["seed"])
        if key in keyed:
            raise ValueError(f"duplicate evaluation for {key}")
        keyed[key] = run

    comparisons = []
    scopes = sorted({run["scope"] for run in runs})
    for scope in scopes:
        scope_runs = [run for run in runs if run["scope"] == scope]
        methods = sorted({run["method"] for run in scope_runs})
        for method in methods:
            seeds = sorted({run["seed"] for run in scope_runs if run["method"] == method})
            for seed in seeds:
                mv = keyed.get((scope, method, "multivariate", seed))
                uv = keyed.get((scope, method, "univariate", seed))
                if mv is not None and uv is not None:
                    comparisons.append(
                        {
                            "scope": scope,
                            "left": f"{method}_multivariate",
                            "right": f"{method}_univariate",
                            "seed": seed,
                            "interpretation": (
                                "negative means the multivariate mode has lower error"
                            ),
                            "percent_difference": {
                                metric: _percent_difference(
                                    mv["aggregate"][metric], uv["aggregate"][metric]
                                )
                                for metric in METRICS
                            },
                        }
                    )
        teacher = next(
            (
                run
                for run in scope_runs
                if run["method"] == "timesfm3_teacher" and run["mode"] == "multivariate"
            ),
            None,
        )
        if teacher is not None:
            for run in scope_runs:
                if not run["method"].startswith("student_") or run["mode"] != "multivariate":
                    continue
                comparisons.append(
                    {
                        "scope": scope,
                        "left": f"{run['method']}_multivariate",
                        "right": "timesfm3_teacher_multivariate",
                        "seed": run["seed"],
                        "interpretation": "positive means the student has higher error",
                        "percent_difference": {
                            metric: _percent_difference(
                                run["aggregate"][metric], teacher["aggregate"][metric]
                            )
                            for metric in METRICS
                        },
                    }
                )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["scope"], run["method"], run["mode"])].append(run)
    uncertainty = []
    for (scope, method, mode), group in sorted(groups.items()):
        values = {metric: [run["aggregate"][metric] for run in group] for metric in METRICS}
        uncertainty.append(
            {
                "scope": scope,
                "method": method,
                "mode": mode,
                "seeds": sorted(run["seed"] for run in group),
                "metrics": {
                    metric: {
                        "mean": float(np.mean(metric_values)),
                        "sample_standard_deviation": (
                            float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else None
                        ),
                        "minimum": float(np.min(metric_values)),
                        "maximum": float(np.max(metric_values)),
                    }
                    for metric, metric_values in values.items()
                },
            }
        )

    result = {
        "status": "succeeded",
        "metric_direction": "lower is better",
        "aggregation": "unweighted geometric mean across configurations",
        "runs": runs,
        "comparisons": comparisons,
        "multi_seed_uncertainty": uncertainty,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
