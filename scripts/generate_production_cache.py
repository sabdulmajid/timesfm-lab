#!/usr/bin/env python3
"""Generate a resumable production teacher cache with one replica per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import sample_windows
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]
FP16_MAX_SCALED_ERROR = 5e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _GpuSampler:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.physical_gpu}",
                    "--query-gpu=utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                try:
                    utilization, power, memory = completed.stdout.strip().split(",")
                    self.samples.append(
                        {
                            "utilization_percent": float(utilization),
                            "power_watts": float(power),
                            "memory_used_mib": float(memory),
                        }
                    )
                except ValueError:
                    pass
            self._stop.wait(1.0)

    def __enter__(self) -> _GpuSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _fp16_candidate(arrays: list[npt.NDArray[np.float32]]) -> tuple[list[np.ndarray], bool, float]:
    maximum_scaled_error = 0.0
    halves = []
    with np.errstate(over="ignore", invalid="ignore"):
        for values in arrays:
            half = values.astype(np.float16)
            halves.append(half)
            if not np.isfinite(half).all():
                return arrays, False, math.inf
            denominator = np.maximum(np.abs(values), 1.0)
            error = np.abs(values - half.astype(np.float32)) / denominator
            maximum_scaled_error = max(maximum_scaled_error, float(error.max(initial=0.0)))
    if maximum_scaled_error > FP16_MAX_SCALED_ERROR:
        return arrays, False, maximum_scaled_error
    return halves, True, maximum_scaled_error


def _write_shard(
    output: Path,
    arrays: dict[str, npt.NDArray[Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(output)
    result = {
        **metadata,
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "write_and_checksum_seconds": time.perf_counter() - started,
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _load_valid_sidecar(output: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    sidecar = output.with_suffix(".json")
    if not output.exists() or not sidecar.exists():
        return None
    try:
        metadata = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None
    if metadata.get("sha256") != _sha256(output):
        return None
    return metadata


def _teacher_outputs(
    teacher: Any,
    contexts: list[np.ndarray],
    horizon: int,
    true_multivariate: bool,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    ordinary = list(
        teacher.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            sort_quantiles=True,
            univariate=False,
        )
    )
    multivariate = [np.asarray(output.quantiles, dtype=np.float32) for output in ordinary]
    if not true_multivariate:
        return multivariate, None
    univariate = list(
        teacher.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            sort_quantiles=True,
            univariate=True,
        )
    )
    return multivariate, [np.asarray(output.quantiles, dtype=np.float32) for output in univariate]


def _generate_dataset(
    teacher: Any,
    item: dict[str, Any],
    *,
    data_root: Path,
    cache_root: Path,
    seed: int,
    sampling_mode: str,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset_name = str(item["dataset"])
    context_length = int(item["context"])
    horizon = int(item["horizon"])
    requested = int(item["requested_windows"])
    batch_size = int(item["physical_batch_size"])
    shard_windows = int(item["shard_windows"])
    true_multivariate = item["view_class"] == "true_multivariate"
    dataset_output = cache_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    decode_started = time.perf_counter()
    dataset = load_from_disk(str(data_root / dataset_name), keep_in_memory=False)
    source = [
        np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
        for row in range(len(dataset))
    ]
    decode_seconds = time.perf_counter() - decode_started
    variates = {values.shape[0] for values in source}
    if len(variates) != 1:
        raise ValueError(f"{dataset_name} has mixed actual variate counts: {variates}")
    actual_variates = variates.pop()
    if true_multivariate != (actual_variates > 1):
        raise ValueError(f"inventory view class disagrees with actual data for {dataset_name}")

    windows, sampling = sample_windows(
        [values.shape[-1] for values in source],
        context_length=context_length,
        horizon=horizon,
        total_windows=requested,
        seed=seed,
        mode=sampling_mode,
    )
    if len(windows) != requested or len(set(windows)) != requested:
        raise RuntimeError(f"production sampler failed exact allocation for {dataset_name}")

    futures: list[Future[dict[str, Any]]] = []
    cached_windows = 0
    inference_seconds = 0.0
    materialize_seconds = 0.0
    shard_count = math.ceil(len(windows) / shard_windows)
    for shard_index, shard_start in enumerate(range(0, len(windows), shard_windows)):
        shard_meta = windows[shard_start : shard_start + shard_windows]
        output = dataset_output / f"shard-{shard_index:05d}-of-{shard_count:05d}.npz"
        expected = {
            "dataset": dataset_name,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "windows": len(shard_meta),
            "context_length": context_length,
            "horizon": horizon,
            "actual_variates": actual_variates,
            "view_class": item["view_class"],
        }
        existing = _load_valid_sidecar(output, expected)
        if existing is not None:
            futures.append(executor.submit(lambda value=existing: value))
            cached_windows += len(shard_meta)
            print(f"{dataset_name}: reused shard {shard_index + 1}/{shard_count}", flush=True)
            continue

        row_indices: list[int] = []
        end_indices: list[int] = []
        teacher_primary: list[np.ndarray] = []
        teacher_uv: list[np.ndarray] = []
        for batch_start in range(0, len(shard_meta), batch_size):
            batch_meta = shard_meta[batch_start : batch_start + batch_size]
            materialize_started = time.perf_counter()
            contexts = [
                np.ascontiguousarray(source[row][:, end - context_length : end])
                for row, end in batch_meta
            ]
            materialize_seconds += time.perf_counter() - materialize_started
            inference_started = time.perf_counter()
            primary, uv = _teacher_outputs(teacher, contexts, horizon, true_multivariate)
            inference_seconds += time.perf_counter() - inference_started
            if uv is not None and len(uv) != len(primary):
                raise RuntimeError("teacher MV/UV batch lengths disagree")
            for index, ((row, end), prediction) in enumerate(
                zip(batch_meta, primary, strict=True)
            ):
                expected_shape = (actual_variates, horizon, 9)
                if prediction.shape != expected_shape or not np.isfinite(prediction).all():
                    raise RuntimeError(
                        f"invalid teacher output for {dataset_name}: {prediction.shape}"
                    )
                row_indices.append(row)
                end_indices.append(end)
                teacher_primary.append(prediction)
                if uv is not None:
                    uv_prediction = uv[index]
                    if uv_prediction.shape != expected_shape or not np.isfinite(
                        uv_prediction
                    ).all():
                        raise RuntimeError(
                            f"invalid teacher UV output for {dataset_name}: {uv_prediction.shape}"
                        )
                    teacher_uv.append(uv_prediction)

        primary_array = np.stack(teacher_primary)
        output_arrays, fp16_safe, maximum_scaled_error = _fp16_candidate(
            [primary_array]
            + ([np.stack(teacher_uv)] if true_multivariate else [])
        )
        arrays: dict[str, npt.NDArray[Any]] = {
            "row_index": np.asarray(row_indices, dtype=np.int32),
            "context_end": np.asarray(end_indices, dtype=np.int32),
            "context_length": np.full(len(row_indices), context_length, dtype=np.int32),
            "horizon": np.full(len(row_indices), horizon, dtype=np.int32),
        }
        if true_multivariate:
            arrays["teacher_multivariate"] = output_arrays[0]
            arrays["teacher_univariate"] = output_arrays[1]
        else:
            arrays["teacher_output"] = output_arrays[0]
        metadata = {
            **expected,
            "row_index_min": min(row_indices),
            "row_index_max": max(row_indices),
            "context_end_min": min(end_indices),
            "context_end_max": max(end_indices),
            "teacher_views": item["teacher_views"],
            "output_dtype": str(output_arrays[0].dtype),
            "fp16_safe_and_accurate": fp16_safe,
            "fp16_maximum_scaled_error": maximum_scaled_error,
            "fp16_scaled_error_limit": FP16_MAX_SCALED_ERROR,
        }
        futures.append(executor.submit(_write_shard, output, arrays, metadata))
        cached_windows += len(shard_meta)
        print(
            f"{dataset_name}: generated shard {shard_index + 1}/{shard_count} "
            f"({cached_windows}/{requested})",
            flush=True,
        )
    shard_results = [future.result() for future in futures]
    del source, dataset
    return {
        "dataset": dataset_name,
        "domain": item["domain"],
        "view_class": item["view_class"],
        "actual_variates": actual_variates,
        "context_length": context_length,
        "horizon": horizon,
        "requested_windows": requested,
        "generated_windows": sum(int(shard["windows"]) for shard in shard_results),
        "unique_windows": len(
            {(dataset_name, row, end, context_length, horizon) for row, end in windows}
        ),
        "shortfall": requested - len(windows),
        "sampling": sampling,
        "physical_batch_size": batch_size,
        "source_decode_seconds": decode_seconds,
        "window_materialization_seconds": materialize_seconds,
        "teacher_inference_seconds": inference_seconds,
        "teacher_windows_per_second": (
            requested / inference_seconds if inference_seconds else None
        ),
        "cache_bytes": sum(int(shard["bytes"]) for shard in shard_results),
        "shards": shard_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--assigned-gpu", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    if plan["model_revision"] != config["model_revision"]:
        raise ValueError("plan/config model revisions disagree")
    if plan["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("plan/config dataset revisions disagree")
    assigned = [
        item for item in plan["datasets"] if int(item["assigned_gpu"]) == args.assigned_gpu
    ]
    if not assigned:
        raise ValueError(f"plan assigns no datasets to GPU {args.assigned_gpu}")

    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator  # type: ignore[import-untyped]

    if torch.cuda.device_count() != 1:
        raise RuntimeError("launch each cache worker with exactly one CUDA_VISIBLE_DEVICES GPU")
    record = RunRecord.start(
        run_id=f"{config['run_id']}-cache-gpu{args.assigned_gpu}",
        config_path=f"{args.config};{args.plan}",
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    teacher = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(config["model"]["id"]),
            revision=str(config["model_revision"]),
            device="cuda:0",
            per_core_batch_size=max(int(item["physical_batch_size"]) for item in assigned),
        )
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    with _GpuSampler(args.physical_gpu) as gpu_sampler, ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cache-writer"
    ) as executor:
        datasets = [
            _generate_dataset(
                teacher,
                item,
                data_root=args.data_root,
                cache_root=args.cache_root,
                seed=int(config["seed"]),
                sampling_mode=str(plan["sampling"]),
                executor=executor,
            )
            for item in assigned
        ]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall_started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    total_windows = sum(int(item["generated_windows"]) for item in datasets)
    utilization = [sample["utilization_percent"] for sample in gpu_sampler.samples]
    power = [sample["power_watts"] for sample in gpu_sampler.samples]
    memory = [sample["memory_used_mib"] for sample in gpu_sampler.samples]
    metrics = {
        "windows": float(total_windows),
        "windows_per_second": total_windows / elapsed,
        "cache_bytes": float(sum(int(item["cache_bytes"]) for item in datasets)),
        "unique_windows": float(sum(int(item["unique_windows"]) for item in datasets)),
    }
    record.extra.update(
        {
            "assigned_gpu": args.assigned_gpu,
            "physical_gpu": args.physical_gpu,
            "datasets": datasets,
            "summary": {
                "elapsed_seconds": elapsed,
                "process_user_cpu_seconds": usage.ru_utime - usage_started.ru_utime,
                "process_system_cpu_seconds": usage.ru_stime - usage_started.ru_stime,
                "windows_per_second": total_windows / elapsed,
                "cache_bytes_per_second": (
                    sum(int(item["cache_bytes"]) for item in datasets) / elapsed
                ),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_samples": len(gpu_sampler.samples),
                "gpu_utilization_mean_percent": (
                    float(np.mean(utilization)) if utilization else None
                ),
                "gpu_utilization_max_percent": max(utilization, default=None),
                "gpu_power_mean_watts": float(np.mean(power)) if power else None,
                "gpu_power_max_watts": max(power, default=None),
                "gpu_smi_memory_max_mib": max(memory, default=None),
            },
            "cache_semantics": {
                "univariate": "one ordinary teacher view; no duplicated MV/UV supervision",
                "true_multivariate": (
                    "separate native multivariate and target-only univariate views"
                ),
                "ground_truth": (
                    "retrievable exactly by dataset,row_index,context_end,context_length,horizon"
                ),
                "storage": "uncompressed NPZ with per-shard SHA-256 and atomic completion",
                "fp16_policy": (
                    "store FP16 only when finite and max abs(error)/max(abs(value),1) <= "
                    f"{FP16_MAX_SCALED_ERROR}; otherwise retain FP32"
                ),
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
    )
    record.succeed(metrics)
    record.write(args.output)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
