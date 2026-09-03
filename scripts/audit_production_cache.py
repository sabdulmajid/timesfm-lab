#!/usr/bin/env python3
"""Verify a completed production cache against its immutable corpus plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import sample_windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_summary(paths: list[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text()) for path in paths]
    if any(record["status"] != "succeeded" for record in records):
        raise ValueError("at least one cache worker did not succeed")
    datasets = [dataset for record in records for dataset in record["extra"]["datasets"]]
    summaries = [record["extra"]["summary"] for record in records]
    elapsed = [float(summary["elapsed_seconds"]) for summary in summaries]
    windows = sum(int(dataset["generated_windows"]) for dataset in datasets)
    return {
        "worker_result_paths": [str(path.resolve()) for path in paths],
        "workers": len(records),
        "worker_git_commits": sorted({record["git_commit"] for record in records}),
        "worker_model_revisions": sorted({record["model_revision"] for record in records}),
        "worker_dataset_revisions": sorted({record["dataset_revision"] for record in records}),
        "worker_wall_seconds": elapsed,
        "two_replica_wall_seconds": max(elapsed),
        "aggregate_windows_per_second": windows / max(elapsed),
        "mean_gpu_utilization_percent_by_worker": [
            summary["gpu_utilization_mean_percent"] for summary in summaries
        ],
        "max_gpu_utilization_percent_by_worker": [
            summary["gpu_utilization_max_percent"] for summary in summaries
        ],
        "peak_cuda_allocated_bytes_by_worker": [
            summary["peak_cuda_allocated_bytes"] for summary in summaries
        ],
        "max_gpu_smi_memory_mib_by_worker": [
            summary["gpu_smi_memory_max_mib"] for summary in summaries
        ],
        "mean_power_watts_by_worker": [summary["gpu_power_mean_watts"] for summary in summaries],
        "process_user_cpu_seconds_by_worker": [
            summary["process_user_cpu_seconds"] for summary in summaries
        ],
        "process_system_cpu_seconds_by_worker": [
            summary["process_system_cpu_seconds"] for summary in summaries
        ],
        "cache_bytes_per_second_by_worker": [
            summary["cache_bytes_per_second"] for summary in summaries
        ],
        "reported_datasets": sorted(dataset["dataset"] for dataset in datasets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--worker-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    if config["model_revision"] != plan["model_revision"]:
        raise ValueError("config and plan teacher revisions differ")
    if config["dataset_revision"] != plan["dataset_revision"]:
        raise ValueError("config and plan dataset revisions differ")
    seed = int(config["seed"])
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset_results = []
    domain_windows: Counter[str] = Counter()
    view_windows: Counter[str] = Counter()
    dtype_windows: Counter[str] = Counter()
    total_requested = 0
    total_generated = 0
    total_unique = 0
    total_bytes = 0
    for item in plan["datasets"]:
        name = str(item["dataset"])
        requested = int(item["requested_windows"])
        context = int(item["context"])
        horizon = int(item["horizon"])
        true_mv = item["view_class"] == "true_multivariate"
        paths = sorted((args.cache_root / name).glob("shard-*.npz"))
        expected_shards = (requested + int(item["shard_windows"]) - 1) // int(item["shard_windows"])
        if len(paths) != expected_shards:
            raise ValueError(f"{name}: expected {expected_shards} shards, found {len(paths)}")

        rows = []
        ends = []
        shard_manifest = []
        generated = 0
        dataset_bytes = 0
        for path in paths:
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.exists():
                raise FileNotFoundError(f"missing cache sidecar: {sidecar_path}")
            sidecar = json.loads(sidecar_path.read_text())
            actual_sha256 = _sha256(path)
            if sidecar["sha256"] != actual_sha256:
                raise ValueError(f"{name}: checksum mismatch for {path.name}")
            expected_sidecar = {
                "dataset": name,
                "context_length": context,
                "horizon": horizon,
                "view_class": item["view_class"],
            }
            if any(sidecar.get(key) != value for key, value in expected_sidecar.items()):
                raise ValueError(f"{name}: sidecar metadata mismatch for {path.name}")
            with np.load(path) as shard:
                keys = set(shard.files)
                expected_keys = {
                    "row_index",
                    "context_end",
                    "context_length",
                    "horizon",
                }
                expected_keys.update(
                    ("teacher_multivariate", "teacher_univariate")
                    if true_mv
                    else ("teacher_output",)
                )
                if keys != expected_keys:
                    raise ValueError(f"{name}: incorrect keys in {path.name}: {sorted(keys)}")
                shard_rows = shard["row_index"]
                shard_ends = shard["context_end"]
                count = len(shard_rows)
                if shard_ends.shape != shard_rows.shape:
                    raise ValueError(f"{name}: row/end shape mismatch in {path.name}")
                if not np.all(shard["context_length"] == context):
                    raise ValueError(f"{name}: context mismatch in {path.name}")
                if not np.all(shard["horizon"] == horizon):
                    raise ValueError(f"{name}: horizon mismatch in {path.name}")
                prediction_keys = (
                    ("teacher_multivariate", "teacher_univariate")
                    if true_mv
                    else ("teacher_output",)
                )
                prediction_dtypes = set()
                for key in prediction_keys:
                    prediction = shard[key]
                    expected_shape = (
                        count,
                        max(int(value) for value in item["actual_variate_counts"]),
                        horizon,
                        9,
                    )
                    if prediction.shape != expected_shape:
                        raise ValueError(
                            f"{name}: {key} shape {prediction.shape} != {expected_shape}"
                        )
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"{name}: non-finite values in {path.name}:{key}")
                    prediction_dtypes.add(str(prediction.dtype))
                if len(prediction_dtypes) != 1:
                    raise ValueError(f"{name}: teacher views have different dtypes")
                output_dtype = prediction_dtypes.pop()
                if output_dtype != sidecar["output_dtype"]:
                    raise ValueError(f"{name}: sidecar dtype mismatch in {path.name}")
                fp16_valid = (
                    output_dtype == "float16"
                    and sidecar["fp16_safe_and_accurate"] is True
                    and sidecar["fp16_maximum_scaled_error"] <= sidecar["fp16_scaled_error_limit"]
                )
                fp32_valid = (
                    output_dtype == "float32" and sidecar["fp16_safe_and_accurate"] is False
                )
                if not (fp16_valid or fp32_valid):
                    raise ValueError(f"{name}: invalid precision policy in {path.name}")
                rows.append(shard_rows.astype(np.int64))
                ends.append(shard_ends.astype(np.int64))
            generated += count
            dataset_bytes += int(path.stat().st_size)
            dtype_windows[output_dtype] += count
            shard_manifest.append(
                {
                    "path": str(path.resolve()),
                    "sha256": actual_sha256,
                    "bytes": int(path.stat().st_size),
                    "windows": count,
                    "output_dtype": output_dtype,
                    "fp16_maximum_scaled_error": sidecar["fp16_maximum_scaled_error"],
                }
            )

        row_index = np.concatenate(rows)
        context_end = np.concatenate(ends)
        unique = len(set(zip(row_index.tolist(), context_end.tolist(), strict=True)))
        dataset = load_from_disk(str(args.data_root / name), keep_in_memory=False)
        source_shapes = [
            np.atleast_2d(np.asarray(dataset[row]["target"])).shape for row in range(len(dataset))
        ]
        actual_variates = {shape[0] for shape in source_shapes}
        if actual_variates != set(int(value) for value in item["actual_variate_counts"]):
            raise ValueError(f"{name}: source variates disagree with plan")
        source_lengths = [shape[-1] for shape in source_shapes]
        expected_windows, sampling = sample_windows(
            source_lengths,
            context_length=context,
            horizon=horizon,
            total_windows=requested,
            seed=seed,
            mode=str(plan["sampling"]),
        )
        observed_windows = list(zip(row_index.tolist(), context_end.tolist(), strict=True))
        if observed_windows != expected_windows:
            raise ValueError(f"{name}: cached window sequence differs from deterministic plan")
        retrievable = all(
            0 <= row < len(source_lengths)
            and end >= context
            and end + horizon <= source_lengths[row]
            for row, end in observed_windows
        )
        if not retrievable:
            raise ValueError(f"{name}: at least one ground-truth target is not retrievable")
        if generated != requested or unique != requested:
            raise ValueError(
                f"{name}: requested={requested}, generated={generated}, unique={unique}"
            )
        domain_windows[str(item["domain"])] += generated
        view_windows[str(item["view_class"])] += generated
        total_requested += requested
        total_generated += generated
        total_unique += unique
        total_bytes += dataset_bytes
        dataset_results.append(
            {
                "dataset": name,
                "domain": item["domain"],
                "view_class": item["view_class"],
                "actual_variates": sorted(actual_variates),
                "context_length": context,
                "horizon": horizon,
                "requested_windows": requested,
                "possible_distinct_windows": sampling["available_unique_windows"],
                "generated_windows": generated,
                "unique_windows": unique,
                "duplicate_windows": generated - unique,
                "duplicate_rate": (generated - unique) / generated,
                "shortfall": requested - generated,
                "ground_truth_retrievable": retrievable,
                "deterministic_sampler_sequence_match": True,
                "cache_bytes": dataset_bytes,
                "shards": shard_manifest,
            }
        )
        print(f"audited {name}: {generated} distinct windows", flush=True)

    if sorted(path.name for path in args.cache_root.iterdir() if path.is_dir()) != sorted(
        item["dataset"] for item in plan["datasets"]
    ):
        raise ValueError("cache root dataset directories do not exactly match the plan")
    worker_execution = _worker_summary(args.worker_result) if args.worker_result else None
    if worker_execution is not None:
        planned_datasets = sorted(item["dataset"] for item in plan["datasets"])
        if worker_execution["reported_datasets"] != planned_datasets:
            raise ValueError("worker outputs do not report exactly the planned datasets")
        if worker_execution["worker_model_revisions"] != [plan["model_revision"]]:
            raise ValueError("worker teacher revisions disagree with plan")
        if worker_execution["worker_dataset_revisions"] != [plan["dataset_revision"]]:
            raise ValueError("worker dataset revisions disagree with plan")
    result = {
        "status": "succeeded",
        "plan_path": str(args.plan.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "model_revision": plan["model_revision"],
        "dataset_revision": plan["dataset_revision"],
        "sampling_seed": seed,
        "sampling": plan["sampling"],
        "summary": {
            "requested_windows": total_requested,
            "possible_capacity_at_selected_shapes": plan["summary"][
                "possible_capacity_at_selected_shapes"
            ],
            "generated_windows": total_generated,
            "unique_windows": total_unique,
            "duplicate_windows": total_generated - total_unique,
            "duplicate_rate": (total_generated - total_unique) / total_generated,
            "shortfall": total_requested - total_generated,
            "datasets": len(dataset_results),
            "view_class_windows": dict(sorted(view_windows.items())),
            "domain_windows": dict(sorted(domain_windows.items())),
            "output_dtype_windows": dict(sorted(dtype_windows.items())),
            "cache_bytes": total_bytes,
            "all_ground_truth_retrievable": True,
            "all_shard_checksums_verified": True,
            "all_teacher_outputs_finite": True,
            "all_deterministic_sampler_sequences_match": True,
        },
        "worker_execution": worker_execution,
        "datasets": dataset_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
