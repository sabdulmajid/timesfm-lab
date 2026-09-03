#!/usr/bin/env python3
"""Benchmark unmodified official TimesFM 3 inference with synchronized CUDA timing."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _contexts(batch: int, variates: int, length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    time_axis = np.arange(length, dtype=np.float32)
    values: list[np.ndarray] = []
    for item in range(batch):
        rows = []
        for channel in range(variates):
            signal = (
                0.002 * time_axis
                + np.sin(time_axis / (17.0 + channel))
                + 0.2 * np.cos(time_axis / (61.0 + item))
            )
            rows.append(signal + rng.normal(0.0, 0.01, length).astype(np.float32))
        values.append(np.stack(rows).astype(np.float32, copy=False))
    return values


class _GpuSampler:
    def __init__(self, physical_index: int) -> None:
        self.physical_index = physical_index
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.physical_index}",
                    "--query-gpu=utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                try:
                    utilization, power = result.stdout.strip().split(",")
                    self.samples.append(
                        {"utilization_percent": float(utilization), "power_watts": float(power)}
                    )
                except ValueError:
                    pass
            self._stop.wait(0.1)

    def __enter__(self) -> _GpuSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    model_cfg = config["model"]
    record = RunRecord.start(
        run_id=config["run_id"],
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=config["model_revision"],
        dataset_revision=config["dataset_revision"],
        hardware_snapshot=config["hardware_snapshot"],
        repository=ROOT,
    )
    load_started = time.perf_counter()
    evaluator = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=model_cfg["id"],
            revision=config["model_revision"],
            device=model_cfg["device"],
            per_core_batch_size=int(model_cfg["per_core_batch_size"]),
        )
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    results: list[dict[str, Any]] = []
    warmup = int(config["benchmark"]["warmup"])
    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(config["benchmark"]["physical_gpu_index"])
    )

    for shape_index, shape in enumerate(config["benchmark"]["shapes"]):
        batch = int(shape["batch"])
        variates = int(shape["variates"])
        context_length = int(shape["context"])
        horizon = int(shape["horizon"])
        repeats = int(shape["repeats"])
        contexts = _contexts(batch, variates, context_length, int(config["seed"]) + shape_index)

        def predict() -> list[Any]:
            return list(
                evaluator.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                    make_positive=False,
                    sort_quantiles=True,
                    univariate=False,
                )
            )

        for _ in range(warmup):
            outputs = predict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies_ms: list[float] = []
        with _GpuSampler(physical_gpu) as sampler:
            for _ in range(repeats):
                started = time.perf_counter()
                outputs = predict()
                torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

        quantiles = np.asarray(outputs[0].quantiles)
        if len(outputs) != batch or quantiles.shape != (variates, horizon, 9):
            raise RuntimeError(
                f"unexpected output for {shape['name']}: count={len(outputs)}, shape={quantiles.shape}"
            )
        if not np.isfinite(quantiles).all() or not (np.diff(quantiles, axis=-1) >= 0).all():
            raise RuntimeError(f"invalid quantiles for {shape['name']}")
        total_seconds = math.fsum(latencies_ms) / 1000.0
        sample_utilization = [sample["utilization_percent"] for sample in sampler.samples]
        sample_power = [sample["power_watts"] for sample in sampler.samples]
        results.append(
            {
                "name": shape["name"],
                "batch": batch,
                "variates": variates,
                "context": context_length,
                "horizon": horizon,
                "warmup": warmup,
                "repeats": repeats,
                "latency_ms": {
                    "p50": _percentile(latencies_ms, 50),
                    "p95": _percentile(latencies_ms, 95),
                    "mean": float(np.mean(latencies_ms)),
                    "samples": latencies_ms,
                },
                "throughput": {
                    "requests_per_second": repeats * batch / total_seconds,
                    "series_per_second": repeats * batch * variates / total_seconds,
                    "forecast_points_per_second": repeats * batch * variates * horizon / total_seconds,
                },
                "memory_bytes": {
                    "peak_allocated": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved": int(torch.cuda.max_memory_reserved()),
                },
                "gpu_samples": {
                    "count": len(sampler.samples),
                    "utilization_mean_percent": (
                        float(np.mean(sample_utilization)) if sample_utilization else None
                    ),
                    "utilization_max_percent": max(sample_utilization, default=None),
                    "power_mean_watts": float(np.mean(sample_power)) if sample_power else None,
                    "power_max_watts": max(sample_power, default=None),
                },
                "output_sha256": hashlib.sha256(quantiles.tobytes()).hexdigest(),
            }
        )
        print(results[-1], flush=True)

    record.extra.update(
        {
            "model_load_seconds": model_load_seconds,
            "model_parameter_count": sum(parameter.numel() for parameter in evaluator.model.parameters()),
            "method": "official predict_batch; wall-clock end-to-end timing with CUDA synchronization",
            "precision": str(next(evaluator.model.parameters()).dtype),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "results": results,
        }
    )
    record.succeed({f"{item['name']}/latency_p50_ms": item["latency_ms"]["p50"] for item in results})
    output = args.output or Path(config["output"])
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
