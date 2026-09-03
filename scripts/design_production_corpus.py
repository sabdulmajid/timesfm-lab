#!/usr/bin/env python3
"""Allocate a balanced, distinct production corpus from the pretrain inventory."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from timesfm_lab.config import load_config


def _capacity(item: dict[str, Any], context: int, horizon: int) -> int:
    return int(item["distinct_window_capacity"].get(f"context_{context}_horizon_{horizon}", 0))


def _shape_candidates(
    desired_context: int, fallback_shapes: list[dict[str, int]]
) -> list[tuple[int, int]]:
    candidates = [(desired_context, 64)]
    candidates.extend(
        (int(shape["context"]), int(shape["horizon"])) for shape in fallback_shapes
    )
    result = []
    for shape in candidates:
        if shape not in result and shape[0] <= desired_context:
            result.append(shape)
    return result


def _choose_shapes(
    items: list[dict[str, Any]],
    target: int,
    per_dataset_cap: int,
    context_cycle: list[int],
    fallback_shapes: list[dict[str, int]],
) -> list[dict[str, Any]]:
    fair_target = min(per_dataset_cap, math.ceil(target / len(items)))
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_domain[str(item["domain"])].append(item)
    selected = []
    for domain in sorted(by_domain):
        for index, item in enumerate(sorted(by_domain[domain], key=lambda value: value["dataset"])):
            desired_context = context_cycle[index % len(context_cycle)]
            candidates = _shape_candidates(desired_context, fallback_shapes)
            viable = [
                (context, horizon, _capacity(item, context, horizon))
                for context, horizon in candidates
                if _capacity(item, context, horizon) > 0
            ]
            if not viable:
                continue
            sufficient = [shape for shape in viable if shape[2] >= fair_target]
            context, horizon, capacity = sufficient[0] if sufficient else max(
                viable, key=lambda shape: shape[2]
            )
            selected.append(
                {
                    **item,
                    "context": context,
                    "horizon": horizon,
                    "possible_distinct_windows_at_selected_shape": capacity,
                    "allocation_capacity": min(capacity, per_dataset_cap),
                }
            )
    return selected


def _waterfill(items: list[dict[str, Any]], target: int) -> None:
    remaining = target
    for item in items:
        item["requested_windows"] = 0
    active = list(sorted(items, key=lambda item: item["dataset"]))
    while remaining and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            share = 1
            extra = 0
        allocated = 0
        next_active = []
        for index, item in enumerate(active):
            proposed = share + int(index < extra)
            headroom = int(item["allocation_capacity"]) - int(item["requested_windows"])
            amount = min(proposed, headroom, remaining - allocated)
            item["requested_windows"] += amount
            allocated += amount
            if item["requested_windows"] < item["allocation_capacity"]:
                next_active.append(item)
            if allocated == remaining:
                next_active.extend(active[index + 1 :])
                break
        if allocated == 0:
            break
        remaining -= allocated
        active = next_active
    if remaining:
        raise ValueError(f"corpus allocation is short by {remaining} windows")


def _batch_size(variates: int) -> int:
    if variates == 1:
        return 256
    if variates <= 4:
        return 128
    if variates <= 12:
        return 64
    return 32


def _throughput(variates: int, context: int) -> float:
    base = 80.95 if variates == 1 else 9.16 * 11 / variates
    # Conservative context scaling; overhead keeps short-context throughput from
    # growing without bound, while long contexts are charged linearly.
    return min(base * 2048 / context, base * 4)


def _cache_bytes(variates: int, horizon: int, true_multivariate: bool, width: int) -> int:
    views = 2 if true_multivariate else 1
    return 8 + views * variates * horizon * 9 * width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inventory = json.loads(args.inventory.read_text())
    cache = config["cache"]
    total_target = int(cache["target_windows"])
    mv_target = round(total_target * float(cache["true_multivariate_fraction"]))
    uv_target = total_target - mv_target
    context_cycle = [int(value) for value in cache["long_context_cycle"]]
    fallback_shapes = [
        {"context": int(item["context"]), "horizon": int(item["horizon"])}
        for item in cache["fallback_shapes"]
    ]
    source_items = inventory["datasets"]
    selected_by_class = {
        "univariate": _choose_shapes(
            [item for item in source_items if item["view_class"] == "univariate"],
            uv_target,
            int(cache["maximum_windows_per_univariate_dataset"]),
            context_cycle,
            fallback_shapes,
        ),
        "true_multivariate": _choose_shapes(
            [item for item in source_items if item["view_class"] == "true_multivariate"],
            mv_target,
            int(cache["maximum_windows_per_multivariate_dataset"]),
            context_cycle,
            fallback_shapes,
        ),
    }
    _waterfill(selected_by_class["univariate"], uv_target)
    _waterfill(selected_by_class["true_multivariate"], mv_target)
    selected = selected_by_class["univariate"] + selected_by_class["true_multivariate"]
    for item in selected:
        variates = max(int(value) for value in item["actual_variate_counts"])
        requested = int(item["requested_windows"])
        true_mv = item["view_class"] == "true_multivariate"
        throughput = _throughput(variates, int(item["context"]))
        item["physical_batch_size"] = _batch_size(variates)
        item["estimated_windows_per_second_per_gpu"] = throughput
        item["estimated_gpu_seconds"] = requested / throughput
        item["estimated_cache_bytes_fp16"] = requested * _cache_bytes(
            variates, int(item["horizon"]), true_mv, 2
        )
        item["estimated_cache_bytes_fp32"] = requested * _cache_bytes(
            variates, int(item["horizon"]), true_mv, 4
        )
        item["shard_windows"] = int(cache["shard_windows"])
        item["teacher_views"] = ["ordinary"] if not true_mv else ["multivariate", "univariate"]

    gpu_seconds = [0.0] * int(cache["gpu_count"])
    for item in sorted(selected, key=lambda value: value["estimated_gpu_seconds"], reverse=True):
        gpu = min(range(len(gpu_seconds)), key=gpu_seconds.__getitem__)
        item["assigned_gpu"] = gpu
        gpu_seconds[gpu] += float(item["estimated_gpu_seconds"])
    selected.sort(key=lambda item: item["dataset"])

    domain_windows: Counter[str] = Counter()
    context_windows: Counter[str] = Counter()
    horizon_windows: Counter[str] = Counter()
    for item in selected:
        count = int(item["requested_windows"])
        domain_windows[str(item["domain"])] += count
        context_windows[str(item["context"])] += count
        horizon_windows[str(item["horizon"])] += count
    generated_target = sum(int(item["requested_windows"]) for item in selected)
    result = {
        "status": "planned",
        "run_id": config["run_id"],
        "model_revision": config["model_revision"],
        "dataset_revision": config["dataset_revision"],
        "sampling": cache["sampling"],
        "allocation_policy": {
            "target_windows": total_target,
            "true_multivariate_fraction": float(cache["true_multivariate_fraction"]),
            "per_dataset_caps": {
                "univariate": int(cache["maximum_windows_per_univariate_dataset"]),
                "true_multivariate": int(cache["maximum_windows_per_multivariate_dataset"]),
            },
            "context_cycle_within_each_domain": context_cycle,
            "shape_fallback": fallback_shapes,
            "principle": (
                "equal water-filling over datasets after per-domain context cycling; source "
                "capacity and per-dataset caps prevent giant datasets from dominating"
            ),
        },
        "summary": {
            "requested_windows": total_target,
            "planned_windows": generated_target,
            "possible_capacity_at_selected_shapes": sum(
                int(item["possible_distinct_windows_at_selected_shape"]) for item in selected
            ),
            "shortfall": total_target - generated_target,
            "datasets": len(selected),
            "univariate_datasets": len(selected_by_class["univariate"]),
            "true_multivariate_datasets": len(selected_by_class["true_multivariate"]),
            "univariate_windows": uv_target,
            "true_multivariate_windows": mv_target,
            "domain_windows": dict(sorted(domain_windows.items())),
            "context_windows": dict(sorted(context_windows.items(), key=lambda pair: int(pair[0]))),
            "horizon_windows": dict(sorted(horizon_windows.items(), key=lambda pair: int(pair[0]))),
            "estimated_cache_bytes_fp16": sum(
                int(item["estimated_cache_bytes_fp16"]) for item in selected
            ),
            "estimated_cache_bytes_fp32": sum(
                int(item["estimated_cache_bytes_fp32"]) for item in selected
            ),
            "estimated_gpu_seconds": sum(float(item["estimated_gpu_seconds"]) for item in selected),
            "estimated_two_gpu_wall_seconds": max(gpu_seconds),
            "estimated_gpu_seconds_by_assignment": gpu_seconds,
        },
        "datasets": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
