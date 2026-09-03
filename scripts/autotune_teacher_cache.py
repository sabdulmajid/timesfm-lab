#!/usr/bin/env python3
"""Autotune TimesFM-3 teacher-cache batch size on one GPU.

The timed unit is one cache batch: an MV prediction followed by its UV view.
Inputs are sliced from predecoded GiftEvalPretrain rows before every repetition.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


class _GpuSampler:
    def __init__(self, physical_index: int) -> None:
        self.physical_index = physical_index
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.physical_index}",
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
            self._stop.wait(0.05)

    def __enter__(self) -> _GpuSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class _DecodeTimer:
    """CUDA-event timing around the official model.decode calls."""

    def __init__(self, torch: Any, decode: Any) -> None:
        self.torch = torch
        self.decode = decode
        self.events: list[tuple[Any, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.decode(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return output

    def milliseconds(self) -> float:
        return math.fsum(float(start.elapsed_time(end)) for start, end in self.events)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _prepare_contexts(
    source: list[np.ndarray], batch_size: int, context: int, horizon: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    eligible = [row for row in source if row.shape[-1] >= context + horizon]
    if not eligible:
        raise ValueError("no source rows are long enough for the benchmark shape")
    contexts: list[np.ndarray] = []
    for index in range(batch_size):
        values = eligible[index % len(eligible)]
        end = int(rng.integers(context, values.shape[-1] - horizon + 1))
        contexts.append(np.ascontiguousarray(values[:, end - context : end]))
    return contexts


def _predict_pair(teacher: Any, contexts: list[np.ndarray], horizon: int) -> None:
    list(
        teacher.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            sort_quantiles=True,
            univariate=False,
        )
    )
    list(
        teacher.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            sort_quantiles=True,
            univariate=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from datasets import load_from_disk
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    tuning = config["autotune"]
    candidates = [int(value) for value in tuning["batch_sizes"]]
    plateau_tolerance = float(tuning["plateau_tolerance_fraction"])
    repeats = int(tuning["repeats"])
    warmup = int(tuning["warmup"])
    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(tuning["physical_gpu_index"])
    )
    model_cfg = config["model"]
    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    teacher = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(model_cfg["id"]),
            revision=str(config["model_revision"]),
            device=str(model_cfg["device"]),
            per_core_batch_size=max(candidates),
        )
    )
    torch.cuda.synchronize()
    results: list[dict[str, Any]] = []

    for shape_index, shape in enumerate(tuning["shapes"]):
        dataset_name = str(shape["dataset"])
        dataset = load_from_disk(str(args.data_root / dataset_name))
        source = [
            np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
            for row in range(len(dataset))
        ]
        context = int(shape["context"])
        horizon = int(shape["horizon"])
        shape_results: list[dict[str, Any]] = []
        stopped_after: str | None = None

        for candidate_index, batch_size in enumerate(candidates):
            try:
                for warmup_index in range(warmup):
                    contexts = _prepare_contexts(
                        source,
                        batch_size,
                        context,
                        horizon,
                        int(config["seed"]) + shape_index * 10_000 + warmup_index,
                    )
                    _predict_pair(teacher, contexts, horizon)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                prep_ms: list[float] = []
                inference_ms: list[float] = []
                model_ms: list[float] = []
                residual_ms: list[float] = []
                with _GpuSampler(physical_gpu) as sampler:
                    for repetition in range(repeats):
                        prep_started = time.perf_counter()
                        contexts = _prepare_contexts(
                            source,
                            batch_size,
                            context,
                            horizon,
                            int(config["seed"])
                            + shape_index * 10_000
                            + candidate_index * 100
                            + repetition,
                        )
                        prep_ms.append((time.perf_counter() - prep_started) * 1000.0)
                        decode_timer = _DecodeTimer(torch, teacher.model.decode)
                        started = time.perf_counter()
                        with patch.object(teacher.model, "decode", side_effect=decode_timer):
                            _predict_pair(teacher, contexts, horizon)
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        gpu_model_ms = decode_timer.milliseconds()
                        inference_ms.append(elapsed_ms)
                        model_ms.append(gpu_model_ms)
                        residual_ms.append(max(0.0, elapsed_ms - gpu_model_ms))
                utilization = [item["utilization_percent"] for item in sampler.samples]
                power = [item["power_watts"] for item in sampler.samples]
                smi_memory = [item["memory_used_mib"] for item in sampler.samples]
                elapsed_seconds = math.fsum(inference_ms) / 1000.0
                total_seconds = elapsed_seconds + math.fsum(prep_ms) / 1000.0
                item = {
                    "batch_size": batch_size,
                    "status": "succeeded",
                    "host_preprocessing_ms": _summary(prep_ms),
                    "inference_ms": _summary(inference_ms),
                    "model_decode_gpu_ms": _summary(model_ms),
                    "adapter_transfers_and_postprocess_ms": _summary(residual_ms),
                    "windows_per_second": repeats * batch_size / total_seconds,
                    "inference_only_windows_per_second": (
                        repeats * batch_size / elapsed_seconds
                    ),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "gpu_samples": {
                        "count": len(sampler.samples),
                        "utilization_mean_percent": (
                            float(np.mean(utilization)) if utilization else None
                        ),
                        "utilization_max_percent": max(utilization, default=None),
                        "power_mean_watts": float(np.mean(power)) if power else None,
                        "power_max_watts": max(power, default=None),
                        "smi_memory_max_mib": max(smi_memory, default=None),
                    },
                }
                shape_results.append(item)
                print(f"{shape['name']} batch={batch_size}: {item}", flush=True)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                shape_results.append(
                    {"batch_size": batch_size, "status": "oom", "error": str(error)}
                )
                stopped_after = f"OOM at batch {batch_size}"
                print(f"{shape['name']} batch={batch_size}: OOM", flush=True)
                break

        succeeded = [item for item in shape_results if item["status"] == "succeeded"]
        peak_throughput = max(float(item["windows_per_second"]) for item in succeeded)
        # Pick the first point statistically/practically on the plateau instead
        # of automatically picking the largest allocation after a tiny gain.
        best = next(
            item
            for item in succeeded
            if float(item["windows_per_second"])
            >= peak_throughput * (1.0 - plateau_tolerance)
        )
        results.append(
            {
                "name": str(shape["name"]),
                "dataset": dataset_name,
                "source_rows": len(source),
                "variates": int(source[0].shape[0]),
                "context": context,
                "horizon": horizon,
                "view_pair": "MV then UV (the teacher-cache unit)",
                "measurements": shape_results,
                "selected_batch_size": int(best["batch_size"]),
                "selected_windows_per_second": float(best["windows_per_second"]),
                "peak_measured_windows_per_second": peak_throughput,
                "plateau_tolerance_fraction": plateau_tolerance,
                "stopped_after": stopped_after,
            }
        )

    record.extra.update(
        {
            "results": results,
            "timing_semantics": {
                "host_preprocessing": "predecoded-row selection and contiguous window slicing",
                "model_decode_gpu": "sum of CUDA events around official model.decode calls",
                "adapter_transfers_and_postprocess": (
                    "end-to-end minus model CUDA time; includes official formatting, H2D, D2H, "
                    "quantile sorting, and output construction"
                ),
                "h2d_separately_available": False,
                "reason": "official predict_batch does not expose transfer boundaries",
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
        }
    )
    record.succeed(
        {
            f"{item['name']}/selected_batch_size": float(item["selected_batch_size"])
            for item in results
        }
        | {
            f"{item['name']}/windows_per_second": float(item["selected_windows_per_second"])
            for item in results
        }
    )
    output = args.output or Path(str(config["output"]))
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
