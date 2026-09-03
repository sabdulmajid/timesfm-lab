#!/usr/bin/env python3
"""Inventory the locally materialized, pinned GiftEvalPretrain selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timesfm_lab.revisions import GIFT_EVAL_PRETRAIN_ID, GIFT_EVAL_PRETRAIN_REVISION


def _domain(name: str) -> str:
    lower = name.lower()
    rules = {
        "Transport": (
            "pems",
            "metro",
            "subway",
            "loop",
            "traffic",
            "taxi",
            "uber",
            "rideshare",
            "pedestrian",
            "vehicle_trip",
        ),
        "Energy": (
            "electric",
            "elf",
            "energy",
            "power",
            "solar",
            "wind",
            "building",
            "bdg-",
            "ideal",
            "lcl",
            "gfc",
        ),
        "Healthcare": ("covid", "cdc_", "tycho", "flu"),
        "Nature": (
            "weather",
            "subseasonal",
            "oikolab",
            "sunspot",
            "borealis",
            "cockatoo",
            "hog",
            "bull",
            "air_quality",
            "kdd2022",
        ),
        "Sales": ("favorita", "m5"),
        "Web/CloudOps": ("cluster", "azure", "borg", "wiki", "web_", "pdb", "smart"),
        "Econ/Fin": (
            "bitcoin",
            "fred",
            "m1_",
            "monash_m3",
            "cif_",
            "tourism",
            "nn5",
            "elecdemand",
            "godaddy",
            "sceaux",
            "spain",
        ),
    }
    for domain, markers in rules.items():
        if any(marker in lower for marker in markers):
            return domain
    return "Unclassified"


def _float_leaf(array: pa.Array) -> pa.Array:
    current = array
    while pa.types.is_list(current.type) or pa.types.is_fixed_size_list(current.type):
        current = current.values
    if not pa.types.is_floating(current.type):
        raise TypeError(f"expected floating target leaf, got {current.type}")
    return current


def _chunk_shapes(array: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Return one source length and actual variate count per Arrow row."""

    if pa.types.is_list(array.type) and pa.types.is_floating(array.type.value_type):
        lengths = np.asarray(pc.list_value_length(array), dtype=np.int64)
        return lengths, np.ones(len(array), dtype=np.int64)
    if pa.types.is_fixed_size_list(array.type) and pa.types.is_list(array.type.value_type):
        variates = int(array.type.list_size)
        inner_lengths = np.asarray(pc.list_value_length(array.values), dtype=np.int64).reshape(
            len(array), variates
        )
        if not np.all(inner_lengths == inner_lengths[:, :1]):
            raise ValueError("variates within one example have unequal source lengths")
        return inner_lengths[:, 0], np.full(len(array), variates, dtype=np.int64)
    raise TypeError(f"unsupported target representation: {array.type}")


def _distribution(values: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        "min": int(values.min()),
        "p25": int(np.percentile(values, 25)),
        "p50": int(np.percentile(values, 50)),
        "p75": int(np.percentile(values, 75)),
        "max": int(values.max()),
        "exact_counts": {
            str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)
        },
    }


def _window_capacity(lengths: np.ndarray, context: int, horizon: int) -> int:
    return int(np.maximum(lengths - context - horizon + 1, 0).sum(dtype=np.int64))


def _cache_bytes_per_window(variates: int, horizon: int, bytes_per_value: int) -> int:
    views = 2 if variates > 1 else 1
    # Two int32 source coordinates plus Q=9 quantiles for each required teacher view.
    return 8 + views * variates * horizon * 9 * bytes_per_value


def _throughput_estimate(variates: int) -> dict[str, Any]:
    if variates == 1:
        return {
            "windows_per_second_lower_bound_per_gpu": 80.95,
            "physical_batch": 256,
            "basis": "measured MV+UV pair at V=1; production stores one view and should be faster",
        }
    if variates <= 12:
        return {
            "windows_per_second_per_gpu": 9.16,
            "physical_batch": 64,
            "basis": "measured paired MV+UV cache unit at V=11",
        }
    if variates <= 32:
        return {
            "single_view_windows_per_second_per_gpu": 8.89,
            "physical_batch": 32,
            "basis": (
                "measured single MV view at V=32; paired production cost is slower and will be "
                "measured from generation manifests"
            ),
        }
    return {
        "series_per_second_single_view_per_gpu": 283.0,
        "physical_batch": 16,
        "basis": "conservative extrapolation from the V=32 single-view plateau",
    }


def _inventory_dataset(path: Path, candidate_shapes: list[tuple[int, int]]) -> dict[str, Any]:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset = load_from_disk(str(path), keep_in_memory=False)
    lengths_parts: list[np.ndarray] = []
    variate_parts: list[np.ndarray] = []
    nonfinite = 0
    points = 0
    for chunk in dataset.data.column("target").chunks:
        lengths, variates = _chunk_shapes(chunk)
        lengths_parts.append(lengths)
        variate_parts.append(variates)
        leaf = _float_leaf(chunk)
        leaf_values = leaf.to_numpy(zero_copy_only=False)
        nonfinite += int(np.count_nonzero(~np.isfinite(leaf_values)))
        points += int(leaf_values.size)
    lengths = np.concatenate(lengths_parts)
    variates = np.concatenate(variate_parts)
    if len(lengths) != len(dataset):
        raise AssertionError("Arrow row count disagrees with Dataset row count")
    actual_variates = sorted(int(value) for value in np.unique(variates))
    maximum_variates = max(actual_variates)
    stat = path.stat()
    del stat  # Path exists; source byte count is taken from dataset_info below.
    info = json.loads((path / "dataset_info.json").read_text())
    first = dataset[0]
    return {
        "dataset": path.name,
        "domain": _domain(path.name),
        "source_rows": len(dataset),
        "frequency": str(first.get("freq", "unknown")),
        "actual_variate_counts": actual_variates,
        "actual_variate_count_distribution": _distribution(variates),
        "view_class": "true_multivariate" if maximum_variates > 1 else "univariate",
        "source_length_distribution": _distribution(lengths),
        "observed_target_values": points - nonfinite,
        "nonfinite_target_values": nonfinite,
        "missing_fraction": nonfinite / points if points else 0.0,
        "source_bytes": int(info.get("dataset_size", info.get("size_in_bytes", 0))),
        "distinct_window_capacity": {
            f"context_{context}_horizon_{horizon}": _window_capacity(lengths, context, horizon)
            for context, horizon in candidate_shapes
        },
        "estimated_cache_bytes_per_window": {
            f"horizon_{horizon}_fp16": _cache_bytes_per_window(
                maximum_variates, horizon, 2
            )
            for horizon in sorted({horizon for _, horizon in candidate_shapes})
        }
        | {
            f"horizon_{horizon}_fp32": _cache_bytes_per_window(
                maximum_variates, horizon, 4
            )
            for horizon in sorted({horizon for _, horizon in candidate_shapes})
        },
        "estimated_teacher_throughput": _throughput_estimate(maximum_variates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [
        (32, 8),
        (64, 16),
        (128, 32),
        (256, 64),
        (512, 64),
        (2048, 64),
        (4096, 64),
        (8192, 64),
        (15360, 64),
        (2048, 256),
        (8192, 256),
    ]
    paths = sorted(
        path
        for path in args.data_root.iterdir()
        if path.is_dir() and (path / "state.json").exists()
    )
    datasets = []
    for index, path in enumerate(paths, start=1):
        item = _inventory_dataset(path, candidates)
        datasets.append(item)
        print(
            f"[{index}/{len(paths)}] {path.name}: {item['source_rows']} rows, "
            f"V={item['actual_variate_counts']}",
            flush=True,
        )
    classes = Counter(item["view_class"] for item in datasets)
    domains = Counter(item["domain"] for item in datasets)
    result = {
        "status": "success",
        "dataset_id": GIFT_EVAL_PRETRAIN_ID,
        "dataset_revision": GIFT_EVAL_PRETRAIN_REVISION,
        "selection": {
            "rule": "all snapshot datasets with repository source bytes <= 1 GiB",
            "materialized_datasets": len(datasets),
            "source_bytes": sum(int(item["source_bytes"]) for item in datasets),
            "excluded_large_families": (
                "datasets above 1 GiB, dominated by repeated ERA5/CMIP6 yearly partitions, "
                "were not downloaded for the production candidate pool"
            ),
        },
        "candidate_context_horizon": [
            {"context": context, "horizon": horizon} for context, horizon in candidates
        ],
        "summary": {
            "view_classes": dict(sorted(classes.items())),
            "domains": dict(sorted(domains.items())),
            "source_rows": sum(int(item["source_rows"]) for item in datasets),
        },
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
