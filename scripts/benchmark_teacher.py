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
    from timesfm3.timesfm3_forecaster import _Query

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

        global_horizon = (
            math.ceil(horizon / evaluator.config.output_patch_length)
            * evaluator.config.output_patch_length
        )
        queries = [
            _Query(
                horizon=global_horizon,
                targets=context,
                past_only_covariates=None,
                past_future_covariates=None,
            )
            for context in contexts
        ]
        batch_context = min(
            math.ceil(
                max(query.context_length for query in queries)
                / evaluator.config.input_patch_length
            )
            * evaluator.config.input_patch_length,
            evaluator.global_context,
        )
        formatted = [query.format(batch_context) for query in queries]
        batched_horizon, batched_target, batched_mask, _, _ = tuple(
            list(values) for values in zip(*formatted, strict=True)
        )
        resident_target = torch.from_numpy(np.stack(batched_target)).to(
            evaluator.device, dtype=torch.float32
        )
        resident_mask = torch.from_numpy(np.stack(batched_mask)).to(
            evaluator.device, dtype=torch.bool
        )

        def model_only_predict(
            target: Any = resident_target,
            decode_horizon: int = batched_horizon[0],
            mask: Any = resident_mask,
        ) -> Any:
            with torch.inference_mode():
                return evaluator.model.decode(
                    target=target,
                    horizon=decode_horizon,
                    mask=mask,
                )

        def end_to_end_predict(
            input_contexts: list[np.ndarray] = contexts,
            prediction_horizon: int = horizon,
        ) -> list[Any]:
            return list(
                evaluator.predict_batch(
                    contexts=input_contexts,
                    horizon=prediction_horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                    make_positive=False,
                    sort_quantiles=True,
                    univariate=False,
                )
            )

        for _ in range(warmup):
            model_only_output = model_only_predict()
            outputs = end_to_end_predict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model_only_latencies_ms: list[float] = []
        with _GpuSampler(physical_gpu) as model_only_sampler:
            for _ in range(repeats):
                started = time.perf_counter()
                model_only_output = model_only_predict()
                torch.cuda.synchronize()
                model_only_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        model_only_memory = {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        }
        torch.cuda.reset_peak_memory_stats()
        end_to_end_latencies_ms: list[float] = []
        with _GpuSampler(physical_gpu) as end_to_end_sampler:
            for _ in range(repeats):
                started = time.perf_counter()
                outputs = end_to_end_predict()
                torch.cuda.synchronize()
                end_to_end_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        end_to_end_memory = {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        }

        quantiles = np.asarray(outputs[0].quantiles)
        if len(outputs) != batch or quantiles.shape != (variates, horizon, 9):
            raise RuntimeError(
                f"unexpected output for {shape['name']}: "
                f"count={len(outputs)}, shape={quantiles.shape}"
            )
        if not np.isfinite(quantiles).all() or not (np.diff(quantiles, axis=-1) >= 0).all():
            raise RuntimeError(f"invalid quantiles for {shape['name']}")
        model_only_quantiles = np.sort(
            model_only_output.cpu().numpy()[:, :variates, :horizon, :], axis=-1
        )
        end_to_end_quantiles = np.stack([output.quantiles for output in outputs])
        if not np.array_equal(model_only_quantiles, end_to_end_quantiles):
            raise RuntimeError("model-only and end-to-end forecast paths disagree")

        def latency(samples: list[float]) -> dict[str, Any]:
            return {
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "mean": float(np.mean(samples)),
                "samples": samples,
            }

        def throughput(
            samples: list[float],
            repeat_count: int = repeats,
            batch_size: int = batch,
            variate_count: int = variates,
            prediction_horizon: int = horizon,
        ) -> dict[str, float]:
            total_seconds = math.fsum(samples) / 1000.0
            return {
                "requests_per_second": repeat_count * batch_size / total_seconds,
                "series_per_second": repeat_count * batch_size * variate_count / total_seconds,
                "forecast_points_per_second": (
                    repeat_count
                    * batch_size
                    * variate_count
                    * prediction_horizon
                    / total_seconds
                ),
            }

        def gpu_samples(sampler: _GpuSampler) -> dict[str, float | int | None]:
            sample_utilization = [sample["utilization_percent"] for sample in sampler.samples]
            sample_power = [sample["power_watts"] for sample in sampler.samples]
            return {
                "count": len(sampler.samples),
                "utilization_mean_percent": (
                    float(np.mean(sample_utilization)) if sample_utilization else None
                ),
                "utilization_max_percent": max(sample_utilization, default=None),
                "power_mean_watts": float(np.mean(sample_power)) if sample_power else None,
                "power_max_watts": max(sample_power, default=None),
            }

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
                    "model_only": latency(model_only_latencies_ms),
                    "end_to_end": latency(end_to_end_latencies_ms),
                },
                "throughput": {
                    "model_only": throughput(model_only_latencies_ms),
                    "end_to_end": throughput(end_to_end_latencies_ms),
                },
                "memory_bytes": {
                    "model_only": model_only_memory,
                    "end_to_end": end_to_end_memory,
                },
                "gpu_samples": {
                    "model_only": gpu_samples(model_only_sampler),
                    "end_to_end": gpu_samples(end_to_end_sampler),
                },
                "output_sha256": hashlib.sha256(quantiles.tobytes()).hexdigest(),
            }
        )
        print(results[-1], flush=True)

    record.extra.update(
        {
            "model_load_seconds": model_load_seconds,
            "model_parameter_count": sum(
                parameter.numel() for parameter in evaluator.model.parameters()
            ),
            "method": (
                "separate synchronized wall-clock timing of the pinned model.decode over "
                "resident, officially formatted GPU tensors and official predict_batch end-to-end"
            ),
            "precision": str(next(evaluator.model.parameters()).dtype),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "results": results,
        }
    )
    record.succeed(
        {
            f"{item['name']}/{scope}_latency_p50_ms": item["latency_ms"][scope]["p50"]
            for item in results
            for scope in ("model_only", "end_to_end")
        }
    )
    output = args.output or Path(config["output"])
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
