#!/usr/bin/env python3
"""Compare student cross-variate response fidelity on production validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VARIANTS = ("gt", "kd", "dual_view", "cvrd")


def _student_metrics(section: dict[str, Any]) -> dict[str, Any]:
    student = section["student"]
    correlations = student["response_correlations"]
    return {
        "response_nmae": student["response_nmae"],
        "pearson": correlations["pearson"],
        "spearman": correlations["spearman"],
        "sign_agreement_fraction": student["sign_agreement_fraction"],
        "directional_cosine_mean_over_windows": student[
            "directional_cosine_mean_over_windows"
        ],
        "directional_cosine_global": student["directional_cosine_global"],
        "magnitude_mae_normalized": student["magnitude_mae_normalized"],
    }


def _load(path: Path) -> tuple[str, dict[str, Any]]:
    record = json.loads(path.read_text())
    if record["status"] != "succeeded" or "student" not in record["summary"]:
        raise ValueError(f"student diagnostic did not succeed: {path}")
    variant = str(record["methodology"]["student_label"])
    if variant not in VARIANTS:
        raise ValueError(f"unexpected seed-42 variant label {variant!r}: {path}")
    return variant, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = dict(_load(path) for path in args.result)
    if set(records) != set(VARIANTS) or len(args.result) != len(VARIANTS):
        raise ValueError(f"expected exactly {VARIANTS}, received {tuple(records)}")

    reference = records["gt"]
    reference_teacher = json.dumps(
        {key: value for key, value in reference["summary"].items() if key != "student"},
        sort_keys=True,
    )
    for variant, record in records.items():
        teacher = json.dumps(
            {key: value for key, value in record["summary"].items() if key != "student"},
            sort_keys=True,
        )
        if teacher != reference_teacher:
            raise ValueError(f"teacher diagnostic population mismatch for {variant}")

    result: dict[str, Any] = {
        "status": "succeeded",
        "metric_direction": {
            "response_nmae": "lower is better",
            "magnitude_mae_normalized": "lower is better",
            "correlation_sign_cosine": "higher is better",
        },
        "teacher": {
            "near_zero_fraction": reference["summary"]["near_zero_fraction"],
            "absolute_response_normalized": reference["summary"][
                "absolute_response_normalized"
            ],
            "accuracy": reference["summary"]["teacher_accuracy"],
        },
        "methods": {},
    }
    for variant in VARIANTS:
        record = records[variant]
        result["methods"][variant] = {
            "aggregate": _student_metrics(record["summary"]),
            "by_domain": {
                name: _student_metrics(section)
                for name, section in sorted(record["by_domain"].items())
            },
            "by_horizon_step": {
                step: _student_metrics(section)
                for step, section in sorted(
                    record["by_horizon_step"].items(), key=lambda item: int(item[0])
                )
            },
            "by_quantile": {
                quantile: _student_metrics(section)
                for quantile, section in sorted(
                    record["by_quantile"].items(), key=lambda item: float(item[0])
                )
            },
        }

    dual = result["methods"]["dual_view"]["aggregate"]
    cvrd = result["methods"]["cvrd"]["aggregate"]
    result["critical_comparison_cvrd_minus_dual_view"] = {
        key: cvrd[key] - dual[key]
        for key in dual
        if cvrd[key] is not None and dual[key] is not None
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
