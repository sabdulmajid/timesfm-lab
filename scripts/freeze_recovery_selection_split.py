#!/usr/bin/env python3
"""Freeze a target-blind development/confirmation split for recovery screens.

Only cache window identities are read.  Forecast targets and teacher outputs are
never inspected, so creating this manifest does not consume the confirmation
partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.distill.data import split_cache_indices

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_identities(cache_root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    paths = sorted((cache_root / dataset).glob("shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no cache shards for {dataset!r} under {cache_root}")
    for path in paths:
        with np.load(path) as shard:
            rows.append(np.asarray(shard["row_index"], dtype=np.int64))
            ends.append(np.asarray(shard["context_end"], dtype=np.int64))
    return np.concatenate(rows), np.concatenate(ends)


def _identity_hash(
    dataset: str,
    rows: np.ndarray,
    ends: np.ndarray,
    indices: np.ndarray,
    context: int,
    horizon: int,
) -> str:
    """Hash a set of exact identities independently of shard/index order."""
    chosen_rows = rows[indices].astype("<i8", copy=False)
    chosen_ends = ends[indices].astype("<i8", copy=False)
    order = np.lexsort((chosen_ends, chosen_rows))
    digest = hashlib.sha256()
    encoded = dataset.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<qqQ", context, horizon, len(indices)))
    pairs = np.column_stack((chosen_rows[order], chosen_ends[order])).astype("<i8", copy=False)
    digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


def _index_hash(dataset: str, indices: np.ndarray) -> str:
    digest = hashlib.sha256(dataset.encode("utf-8") + b"\0")
    digest.update(np.sort(indices).astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _partition(
    dataset: str,
    rows: np.ndarray,
    ends: np.ndarray,
    indices: np.ndarray,
    context: int,
    horizon: int,
) -> dict[str, Any]:
    return {
        "count": len(indices),
        "cache_index_sha256": _index_hash(dataset, indices),
        "identity_sha256": _identity_hash(dataset, rows, ends, indices, context, horizon),
    }


def _overall_hash(datasets: list[dict[str, Any]], partition: str) -> str:
    digest = hashlib.sha256()
    for item in sorted(datasets, key=lambda value: value["dataset"]):
        value = item["partitions"][partition]
        digest.update(item["dataset"].encode("utf-8") + b"\0")
        digest.update(struct.pack("<Q", value["count"]))
        digest.update(bytes.fromhex(value["identity_sha256"]))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "results/reproduction/distillation/production-1m-corpus-plan.json",
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "teacher_cache/production-1m")
    parser.add_argument("--outer-fraction", type=float, default=0.10)
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--confirmation-fraction", type=float, default=0.50)
    parser.add_argument("--confirmation-seed", type=int, default=420055)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/reproduction/distillation/performance-recovery-selection-split.json",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    if plan.get("dataset_revision") != "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874":
        raise ValueError("unexpected GiftEvalPretrain revision")

    dataset_results: list[dict[str, Any]] = []
    for item in plan["datasets"]:
        dataset = str(item["dataset"])
        context = int(item["context"])
        horizon = int(item["horizon"])
        rows, ends = _load_identities(args.cache_root, dataset)
        if len(rows) != int(item["requested_windows"]):
            raise ValueError(f"{dataset}: cache count does not match frozen corpus plan")
        if len(set(zip(rows.tolist(), ends.tolist(), strict=True))) != len(rows):
            raise ValueError(f"{dataset}: duplicate cache identities are not admissible")

        outer, outer_report = split_cache_indices(
            rows,
            ends,
            context_length=context,
            horizon=horizon,
            validation_fraction=args.outer_fraction,
            seed=args.outer_seed,
            mode="series_or_time",
        )
        outer_training = np.asarray(outer["training"], dtype=np.int64)
        outer_validation = np.asarray(outer["validation"], dtype=np.int64)
        outer_used = np.union1d(outer_training, outer_validation)
        outer_embargo = np.setdiff1d(np.arange(len(rows), dtype=np.int64), outer_used)

        validation_rows = rows[outer_validation]
        validation_ends = ends[outer_validation]
        inner_mode = "held_out_series" if len(np.unique(validation_rows)) > 1 else "blocked_time"
        unsplittable_reason = None
        try:
            inner, inner_report = split_cache_indices(
                validation_rows,
                validation_ends,
                context_length=context,
                horizon=horizon,
                validation_fraction=args.confirmation_fraction,
                seed=args.confirmation_seed,
                mode=inner_mode,
            )
        except ValueError as error:
            if inner_mode != "blocked_time" or "left no training windows" not in str(error):
                raise ValueError(f"{dataset}: nested {inner_mode} split failed: {error}") from error
            # A short, single-series outer holdout can contain no pair of
            # nonoverlapping context+target intervals.  Keep it in development
            # rather than manufacture leakage; the aggregate confirmation set
            # remains source/block-independent on every included dataset.
            inner = {
                "training": np.arange(len(outer_validation), dtype=np.int64),
                "validation": np.asarray([], dtype=np.int64),
            }
            unsplittable_reason = str(error)
            inner_report = {
                "mode": "development_only_no_valid_independent_confirmation",
                "leakage_control": "no confirmation examples emitted for this dataset",
                "excluded_embargo_windows": 0,
                "reason": unsplittable_reason,
            }
        development = outer_validation[np.asarray(inner["training"], dtype=np.int64)]
        confirmation = outer_validation[np.asarray(inner["validation"], dtype=np.int64)]
        inner_used = np.union1d(development, confirmation)
        inner_embargo = np.setdiff1d(outer_validation, inner_used)

        if not len(development):
            raise ValueError(f"{dataset}: nested split produced no development data")
        if not len(confirmation) and unsplittable_reason is None:
            raise ValueError(f"{dataset}: nested split unexpectedly produced no confirmation data")
        if np.intersect1d(development, confirmation).size:
            raise AssertionError(f"{dataset}: development and confirmation overlap")
        if inner_mode == "held_out_series":
            if np.intersect1d(rows[development], rows[confirmation]).size:
                raise AssertionError(f"{dataset}: held-out confirmation shares source rows")
        else:
            for row in np.unique(rows[confirmation]):
                dev_for_row = development[rows[development] == row]
                confirm_for_row = confirmation[rows[confirmation] == row]
                if len(dev_for_row) and len(confirm_for_row):
                    latest_development_end = int(np.max(ends[dev_for_row] + horizon))
                    earliest_confirmation_start = int(np.min(ends[confirm_for_row] - context))
                    if latest_development_end > earliest_confirmation_start:
                        raise AssertionError(f"{dataset}: blocked confirmation intervals overlap")

        partitions = {
            "outer_training": _partition(dataset, rows, ends, outer_training, context, horizon),
            "outer_validation": _partition(dataset, rows, ends, outer_validation, context, horizon),
            "outer_embargo": _partition(dataset, rows, ends, outer_embargo, context, horizon),
            "development": _partition(dataset, rows, ends, development, context, horizon),
            "confirmation": _partition(dataset, rows, ends, confirmation, context, horizon),
            "inner_embargo": _partition(dataset, rows, ends, inner_embargo, context, horizon),
        }
        dataset_results.append(
            {
                "dataset": dataset,
                "context": context,
                "horizon": horizon,
                "cache_windows": len(rows),
                "outer_split_report": outer_report,
                "inner_split_report": inner_report,
                "confirmation_eligible": bool(len(confirmation)),
                "partitions": partitions,
            }
        )

    partition_names = tuple(dataset_results[0]["partitions"])
    totals = {
        name: {
            "count": sum(item["partitions"][name]["count"] for item in dataset_results),
            "identity_sha256": _overall_hash(dataset_results, name),
        }
        for name in partition_names
    }
    payload = {
        "schema_version": 1,
        "protocol_id": "timesfm3-performance-recovery-v1.1",
        "status": "frozen_uninspected",
        "target_accessed": False,
        "teacher_output_accessed": False,
        "identity_tuple": [
            "dataset",
            "row_index",
            "context_end",
            "context_length",
            "horizon",
        ],
        "identity_encoding": (
            "dataset UTF-8 length+bytes, little-endian int64 context/horizon/count, "
            "then lexicographically sorted little-endian int64 (row_index, context_end) pairs"
        ),
        "source": {
            "plan": str(args.plan.relative_to(ROOT)),
            "plan_sha256": _sha256(args.plan),
            "cache_root": str(args.cache_root.relative_to(ROOT)),
            "dataset_revision": plan["dataset_revision"],
            "dataset_count": len(dataset_results),
        },
        "outer_split": {
            "algorithm": "timesfm_lab.distill.data.split_cache_indices",
            "mode": "series_or_time",
            "validation_fraction": args.outer_fraction,
            "seed": args.outer_seed,
        },
        "nested_split": {
            "population": "outer_validation only",
            "confirmation_fraction": args.confirmation_fraction,
            "seed": args.confirmation_seed,
            "preference": (
                "held_out_series when outer validation has >1 row; otherwise blocked_time"
            ),
            "blocked_time_embargo": (
                "development context+target intervals end no later than the earliest "
                "confirmation context interval"
            ),
            "unsplittable_policy": (
                "If a single-row outer holdout contains no two nonoverlapping context+target "
                "blocks, retain it in development and emit no confirmation identities for that "
                "dataset; never manufacture overlap."
            ),
        },
        "screening_contract": {
            "training_partition": "outer_training",
            "checkpoint_selection_partition": "development",
            "finalist_only_partition": "confirmation",
            "confirmation_metrics_must_not_be_computed_before_finalists_are_frozen": True,
            "reconstruction": (
                "recompute the two split_cache_indices calls from cache identities and reject "
                "unless every per-dataset and aggregate identity_sha256 matches"
            ),
        },
        "totals": totals,
        "datasets": dataset_results,
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
