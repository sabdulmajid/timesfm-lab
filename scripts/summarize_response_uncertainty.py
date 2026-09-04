#!/usr/bin/env python3
"""Aggregate paired Dual-View/CVRD response-transfer diagnostics across seeds."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "response_nmae",
    "pearson",
    "spearman",
    "sign_agreement_fraction",
    "directional_cosine_mean_over_windows",
    "directional_cosine_global",
    "magnitude_mae_normalized",
)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_standard_deviation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = _mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def _summary(values: list[float]) -> dict[str, float | None]:
    deviation = _sample_standard_deviation(values)
    return {
        "mean": _mean(values),
        "sample_standard_deviation": deviation,
        "standard_error": deviation / math.sqrt(len(values)) if deviation is not None else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[tuple[str, int], dict[str, Any]] = {}
    teacher_reference: str | None = None
    for path in args.result:
        record = json.loads(path.read_text())
        if record["status"] != "succeeded" or "student" not in record["summary"]:
            raise ValueError(f"response diagnostic did not succeed: {path}")
        label = str(record["methodology"]["student_label"])
        variant = "dual_view" if label.startswith("dual_view") else "cvrd" if label.startswith("cvrd") else None
        if variant is None:
            raise ValueError(f"expected Dual-View/CVRD diagnostic, received {label!r}")
        seed_match = re.search(r"-seed(\d+)$", label)
        seed = int(seed_match.group(1)) if seed_match else 42
        key = (variant, seed)
        if key in records:
            raise ValueError(f"duplicate response diagnostic for {key}")
        records[key] = record
        teacher = json.dumps(
            {key: value for key, value in record["summary"].items() if key != "student"},
            sort_keys=True,
        )
        if teacher_reference is None:
            teacher_reference = teacher
        elif teacher != teacher_reference:
            raise ValueError(f"teacher diagnostic population mismatch: {path}")

    seeds = sorted(seed for variant, seed in records if variant == "dual_view")
    if seeds != sorted(seed for variant, seed in records if variant == "cvrd") or len(seeds) < 3:
        raise ValueError("expected paired Dual-View/CVRD diagnostics for at least three seeds")

    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    paired: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        dual = records[("dual_view", seed)]["summary"]["student"]
        cvrd = records[("cvrd", seed)]["summary"]["student"]
        dual_correlations = dual["response_correlations"]
        cvrd_correlations = cvrd["response_correlations"]
        dual_values = {
            "response_nmae": dual["response_nmae"],
            "pearson": dual_correlations["pearson"],
            "spearman": dual_correlations["spearman"],
            "sign_agreement_fraction": dual["sign_agreement_fraction"],
            "directional_cosine_mean_over_windows": dual["directional_cosine_mean_over_windows"],
            "directional_cosine_global": dual["directional_cosine_global"],
            "magnitude_mae_normalized": dual["magnitude_mae_normalized"],
        }
        cvrd_values = {
            "response_nmae": cvrd["response_nmae"],
            "pearson": cvrd_correlations["pearson"],
            "spearman": cvrd_correlations["spearman"],
            "sign_agreement_fraction": cvrd["sign_agreement_fraction"],
            "directional_cosine_mean_over_windows": cvrd["directional_cosine_mean_over_windows"],
            "directional_cosine_global": cvrd["directional_cosine_global"],
            "magnitude_mae_normalized": cvrd["magnitude_mae_normalized"],
        }
        for metric in METRICS:
            by_method["dual_view"][metric].append(float(dual_values[metric]))
            by_method["cvrd"][metric].append(float(cvrd_values[metric]))
            paired[metric].append(float(cvrd_values[metric]) - float(dual_values[metric]))

    result = {
        "status": "succeeded",
        "seeds": seeds,
        "metric_direction": {
            "response_nmae": "lower is better",
            "magnitude_mae_normalized": "lower is better",
            "correlation_sign_cosine": "higher is better",
        },
        "methods": {
            method: {metric: _summary(values) for metric, values in metrics.items()}
            for method, metrics in by_method.items()
        },
        "paired_cvrd_minus_dual_view": {
            metric: {"values_by_seed": dict(zip(seeds, values)), **_summary(values)}
            for metric, values in paired.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
