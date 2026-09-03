#!/usr/bin/env python3
"""Record numerical evidence for TimesFM-compatible student normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from timesfm_lab.models import (
    TIMESFM_NORMALIZATION_EPSILON,
    denormalize_forecast,
    normalize_context,
)


def _audit(values: torch.Tensor) -> dict[str, float | int | bool]:
    normalized, mean, scale, observed = normalize_context(values)
    restored = denormalize_forecast(normalized, mean, scale)
    errors = torch.where(observed, (restored - values).abs(), torch.zeros_like(values))
    denominator = (
        torch.where(observed, values.abs(), torch.zeros_like(values)).amax().clamp_min(1.0)
    )
    return {
        "elements": values.numel(),
        "observed_elements": int(observed.sum()),
        "maximum_observed_absolute_value": float(denominator),
        "normalized_all_finite": bool(torch.isfinite(normalized).all()),
        "mean_all_finite": bool(torch.isfinite(mean).all()),
        "scale_all_finite": bool(torch.isfinite(scale).all()),
        "inverse_all_finite": bool(torch.isfinite(restored).all()),
        "minimum_safe_scale": float(scale.min()),
        "maximum_safe_scale": float(scale.max()),
        "maximum_absolute_roundtrip_error": float(errors.max()),
        "maximum_relative_to_input_amplitude_error": float(errors.max() / denominator),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitcoin-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        "zero_variance": torch.tensor([[[7.0, 7.0, 7.0, 7.0]]]),
        "std_1e-8": torch.tensor([[[-1e-8, 1e-8, -1e-8, 1e-8]]]),
        "std_1e-3": torch.tensor([[[-1e-3, 1e-3, -1e-3, 1e-3]]]),
        "std_1e-1": torch.tensor([[[-1e-1, 1e-1, -1e-1, 1e-1]]]),
        "order_one": torch.tensor([[[-2.0, -1.0, 1.0, 2.0]]]),
        "extreme_float32": torch.tensor([[[1.0e30, 1.2e30, 0.9e30, 1.1e30]]]),
        "missing": torch.tensor([[[1.0, float("nan"), 3.0, float("inf")]]]),
    }
    audits = {name: _audit(values) for name, values in cases.items()}

    from datasets import load_from_disk

    dataset = load_from_disk(str(args.bitcoin_dataset))
    bitcoin_rows = []
    for row in dataset:
        target = torch.from_numpy(np.atleast_2d(np.asarray(row["target"], dtype=np.float32)))
        bitcoin_rows.append(_audit(target.unsqueeze(0)))
    audits["bitcoin_with_missing"] = {
        "rows": len(bitcoin_rows),
        "observed_elements": sum(int(item["observed_elements"]) for item in bitcoin_rows),
        "all_checks_finite": all(
            bool(item[key])
            for item in bitcoin_rows
            for key in (
                "normalized_all_finite",
                "mean_all_finite",
                "scale_all_finite",
                "inverse_all_finite",
            )
        ),
        "maximum_safe_scale": max(float(item["maximum_safe_scale"]) for item in bitcoin_rows),
        "maximum_observed_absolute_value": max(
            float(item["maximum_observed_absolute_value"]) for item in bitcoin_rows
        ),
        "maximum_absolute_roundtrip_error": max(
            float(item["maximum_absolute_roundtrip_error"]) for item in bitcoin_rows
        ),
        "maximum_relative_to_input_amplitude_error": max(
            float(item["maximum_relative_to_input_amplitude_error"]) for item in bitcoin_rows
        ),
    }
    result = {
        "status": "success",
        "method": {
            "epsilon": TIMESFM_NORMALIZATION_EPSILON,
            "epsilon_reference": (
                "timesfm/src/timesfm3/util.py::_TOLERANCE at pinned TimesFM git "
                "aa480150652811e732d87a3c5344b235234104e3"
            ),
            "safe_scale": "where(population_std < epsilon, 1.0, population_std)",
            "overflow_control": (
                "factor maximum absolute magnitude before mean summation and centered RMS squaring"
            ),
            "missing_values": (
                "exclude non-finite and masked elements from statistics; emit zero normalized "
                "values at missing positions"
            ),
        },
        "cases": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
