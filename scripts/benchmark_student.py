#!/usr/bin/env python3
"""Benchmark a trained student on the same shapes as the reference teacher."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from timesfm_lab.config import load_config
from timesfm_lab.models import StudentConfig, TimesFMStudent
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _contexts(batch: int, variates: int, length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    axis = np.arange(length, dtype=np.float32)
    result = []
    for item in range(batch):
        rows = [
            0.002 * axis
            + np.sin(axis / (17.0 + channel))
            + 0.2 * np.cos(axis / (61.0 + item))
            + rng.normal(0.0, 0.01, length).astype(np.float32)
            for channel in range(variates)
        ]
        result.append(np.stack(rows).astype(np.float32, copy=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_config", type=Path)
    parser.add_argument("systems_config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("gt", "kd", "dual_view", "cvrd", "relkd"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    student_cfg = load_config(args.student_config)
    systems_cfg = load_config(args.systems_config)
    device = torch.device("cuda:0")
    load_started = time.perf_counter()
    model = TimesFMStudent(StudentConfig(**student_cfg["student"]))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    record = RunRecord.start(
        run_id=f"{student_cfg['run_id']}-{args.variant}-blackwell-inference",
        config_path=f"{args.student_config};{args.systems_config}",
        seed=int(student_cfg["seed"]),
        model_revision=student_cfg["model_revision"],
        dataset_revision=student_cfg["dataset_revision"],
        hardware_snapshot=student_cfg["hardware_snapshot"],
        repository=ROOT,
    )
    results = []
    warmup = int(systems_cfg["benchmark"]["warmup"])
    for shape_index, shape in enumerate(systems_cfg["benchmark"]["shapes"]):
        batch = int(shape["batch"])
        variates = int(shape["variates"])
        context_length = int(shape["context"])
        horizon = int(shape["horizon"])
        repeats = int(shape["repeats"])
        arrays = _contexts(batch, variates, context_length, int(student_cfg["seed"]) + shape_index)
        host_context = np.stack(arrays)
        device_context = torch.from_numpy(host_context).to(device)

        def model_only_predict(
            input_context: torch.Tensor = device_context,
            prediction_horizon: int = horizon,
        ) -> torch.Tensor:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return model(input_context, prediction_horizon)

        def end_to_end_predict(
            input_array: np.ndarray = host_context,
            prediction_horizon: int = horizon,
        ) -> np.ndarray:
            context = torch.from_numpy(input_array).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(context, prediction_horizon)
            return output.float().cpu().numpy()

        for _ in range(warmup):
            model_only_output = model_only_predict()
            output = end_to_end_predict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model_only_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model_only_output = model_only_predict()
            torch.cuda.synchronize()
            model_only_samples.append((time.perf_counter() - started) * 1000.0)
        end_to_end_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = end_to_end_predict()
            torch.cuda.synchronize()
            end_to_end_samples.append((time.perf_counter() - started) * 1000.0)
        model_only_output = model_only_output.float().cpu().numpy()
        if output.shape != (batch, variates, horizon, 9):
            raise RuntimeError(f"unexpected output shape {output.shape}")
        if not np.array_equal(output, model_only_output):
            raise RuntimeError("model-only and end-to-end forecast paths disagree")
        if not np.isfinite(output).all() or not (np.diff(output, axis=-1) >= 0).all():
            raise RuntimeError("invalid forecast quantiles")
        model_only_seconds = math.fsum(model_only_samples) / 1000.0
        end_to_end_seconds = math.fsum(end_to_end_samples) / 1000.0

        def latency(samples: list[float]) -> dict[str, object]:
            return {
                "p50": float(np.percentile(samples, 50)),
                "p95": float(np.percentile(samples, 95)),
                "mean": float(np.mean(samples)),
                "samples": samples,
            }

        def throughput(
            total_seconds: float,
            repeat_count: int = repeats,
            batch_size: int = batch,
            variate_count: int = variates,
            prediction_horizon: int = horizon,
        ) -> dict[str, float]:
            return {
                "requests_per_second": repeat_count * batch_size / total_seconds,
                "series_per_second": (repeat_count * batch_size * variate_count / total_seconds),
                "forecast_points_per_second": (
                    repeat_count * batch_size * variate_count * prediction_horizon / total_seconds
                ),
            }

        result = {
            "name": shape["name"],
            "batch": batch,
            "variates": variates,
            "context": context_length,
            "horizon": horizon,
            "warmup": warmup,
            "repeats": repeats,
            "latency_ms": {
                "model_only": latency(model_only_samples),
                "end_to_end": latency(end_to_end_samples),
            },
            "throughput": {
                "model_only": throughput(model_only_seconds),
                "end_to_end": throughput(end_to_end_seconds),
            },
            "memory_bytes": {
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
            },
            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        }
        results.append(result)
        print(result, flush=True)
    record.extra.update(
        {
            "variant": args.variant,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_distributed": False,
            "model_load_seconds": load_seconds,
            "model_parameter_count": model.parameter_count,
            "precision": "bfloat16 autocast",
            "method": (
                "separate synchronized wall-clock model-only timing over a resident GPU tensor "
                "and end-to-end tensor construction, H2D, inference, and CPU output timing"
            ),
            "results": results,
            "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        }
    )
    record.succeed(
        {
            f"{item['name']}/{scope}_latency_p50_ms": item["latency_ms"][scope]["p50"]
            for item in results
            for scope in ("model_only", "end_to_end")
        }
    )
    record.write(args.output)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
