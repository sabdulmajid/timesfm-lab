#!/usr/bin/env python3
"""Audit exact teacher-cache window uniqueness across one or more cache groups."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config


def _audit_dataset(
    dataset: str,
    cache_dir: Path,
    *,
    requested: int,
    context_length: int,
    horizon: int,
) -> tuple[dict[str, Any], set[tuple[str, int, int, int, int]]]:
    shard_paths = sorted(cache_dir.glob("shard-*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no cache shards found in {cache_dir}")
    keys: list[tuple[str, int, int, int, int]] = []
    shard_windows: dict[str, int] = {}
    for path in shard_paths:
        with np.load(path) as shard:
            rows = shard["row_index"]
            ends = shard["context_end"]
        if rows.shape != ends.shape or rows.ndim != 1:
            raise ValueError(f"invalid row/end metadata in {path}")
        shard_windows[path.name] = len(rows)
        keys.extend(
            (dataset, int(row), int(end), context_length, horizon)
            for row, end in zip(rows, ends, strict=True)
        )
    unique_keys = set(keys)
    duplicates = len(keys) - len(unique_keys)
    return {
        "requested_windows": requested,
        "observed_windows": len(keys),
        "unique_windows": len(unique_keys),
        "duplicate_windows": duplicates,
        "duplicate_rate": duplicates / len(keys) if keys else 0.0,
        "request_shortfall": max(0, requested - len(keys)),
        "context_length": context_length,
        "horizon": horizon,
        "shards": shard_windows,
    }, unique_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-cache",
        nargs=2,
        action="append",
        metavar=("CONFIG", "CACHE_ROOT"),
        required=True,
        help="cache config and root containing one subdirectory per configured dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_dataset: dict[str, dict[str, Any]] = {}
    all_keys: set[tuple[str, int, int, int, int]] = set()
    requested_total = 0
    observed_total = 0
    duplicate_total = 0
    inputs: list[dict[str, str]] = []
    for config_path_text, cache_root_text in args.config_cache:
        config_path = Path(config_path_text)
        cache_root = Path(cache_root_text)
        config = load_config(config_path)
        cache = config["cache"]
        datasets = config.get("datasets", [cache.get("dataset")])
        for dataset in datasets:
            if not dataset:
                raise ValueError(f"no dataset configured in {config_path}")
            if dataset in per_dataset:
                raise ValueError(f"dataset {dataset} occurs in more than one input")
            report, keys = _audit_dataset(
                dataset,
                cache_root / dataset,
                requested=int(cache["total_windows"]),
                context_length=int(cache["context_length"]),
                horizon=int(cache["horizon"]),
            )
            per_dataset[dataset] = report
            all_keys.update(keys)
            requested_total += report["requested_windows"]
            observed_total += report["observed_windows"]
            duplicate_total += report["duplicate_windows"]
        inputs.append(
            {"config": str(config_path), "cache_root": str(cache_root.resolve())}
        )

    document = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "key_fields": [
            "dataset",
            "row_index",
            "context_end",
            "context_length",
            "horizon",
        ],
        "inputs": inputs,
        "summary": {
            "requested_windows": requested_total,
            "observed_windows": observed_total,
            "unique_windows": len(all_keys),
            "duplicate_windows": duplicate_total,
            "duplicate_rate": duplicate_total / observed_total if observed_total else 0.0,
            "request_shortfall": max(0, requested_total - observed_total),
        },
        "per_dataset": dict(sorted(per_dataset.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
