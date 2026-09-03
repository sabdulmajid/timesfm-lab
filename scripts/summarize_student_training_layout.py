#!/usr/bin/env python3
"""Select the lowest-wall-clock layout from matched real-corpus measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VARIANTS = ("gt", "kd", "dual_view", "cvrd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.result]
    if any(record["status"] != "succeeded" for record in records):
        raise ValueError("all input benchmarks must have succeeded")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["layout"]), str(record["variant"]))
        if key in indexed:
            raise ValueError(f"duplicate benchmark result for {key}")
        indexed[key] = record
    expected = {(layout, variant) for layout in ("ddp", "single") for variant in VARIANTS}
    missing = expected - indexed.keys()
    if missing:
        raise ValueError(f"missing benchmark results: {sorted(missing)}")
    normalized = {
        key: float(record["elapsed_seconds"]) / int(record["global_windows"]) * 1_000_000
        for key, record in indexed.items()
    }
    ddp_total = sum(normalized[("ddp", variant)] for variant in VARIANTS)
    concurrent_pairs = (("gt", "kd"), ("dual_view", "cvrd"))
    concurrent_total = sum(
        max(normalized[("single", variant)] for variant in pair) for pair in concurrent_pairs
    )
    selected = "concurrent_single_gpu_pairs" if concurrent_total < ddp_total else "ddp_sequential"
    result = {
        "status": "succeeded",
        "methodology": {
            "data": "real production cache and production batch sequence",
            "normalization": "wall seconds per one million windows processed by each variant",
            "ddp_schedule": "four variants sequentially, both GPUs per variant",
            "concurrent_schedule": "GT+KD concurrently, then DualView+CVRD concurrently",
            "precision": "bfloat16 autocast",
            "optimizer": "fused AdamW",
            "selection_rule": "minimum total wall-clock for all four matched variants",
        },
        "seconds_per_million_windows_by_layout_and_variant": {
            layout: {variant: normalized[(layout, variant)] for variant in VARIANTS}
            for layout in ("ddp", "single")
        },
        "estimated_total_wall_seconds_per_million_windows_each": {
            "ddp_sequential": ddp_total,
            "concurrent_single_gpu_pairs": concurrent_total,
        },
        "selected_layout": selected,
        "selected_layout_relative_wall_reduction": (
            abs(ddp_total - concurrent_total) / max(ddp_total, concurrent_total)
        ),
        "input_results": [str(path.resolve()) for path in args.result],
        "raw_results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
