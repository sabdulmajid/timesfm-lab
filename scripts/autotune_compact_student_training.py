#!/usr/bin/env python3
"""Autotune compact-student physical batches on real production-cache windows.

Each timed iteration performs the complete single-microbatch training path:
source/cache materialization, pinned H2D copies, BF16-autocast forward and loss,
backward, gradient clipping, and a fused AdamW update.  The run intentionally
uses the largest-token production shapes rather than synthetic tensors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from timesfm_lab.config import load_config
from timesfm_lab.distill.losses import DistillationLoss, LossWeights
from timesfm_lab.models import build_student, masked_mean_and_scale
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class _Shape:
    name: str
    dataset: str
    context: int
    horizon: int
    variates: int
    production_windows: int
    shard_path: Path


@dataclass(slots=True)
class _ShapeData:
    shape: _Shape
    row_index: npt.NDArray[np.int32]
    context_end: npt.NDArray[np.int32]
    teacher_primary: npt.NDArray[Any]
    source: dict[int, npt.NDArray[np.float32]]


class _NonFiniteStep(RuntimeError):
    """A measured training step produced a non-finite value."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _validate_and_resolve(config: dict[str, Any]) -> tuple[dict[str, Any], list[_Shape]]:
    candidate_path = _root_path(config["candidate_config"])
    plan_path = _root_path(config["production_plan"])
    data_root = _root_path(config["data_root"])
    cache_root = _root_path(config["cache_root"])
    candidate = load_config(candidate_path)
    plan = json.loads(plan_path.read_text())
    probe = config["probe"]

    if config["model_revision"] != candidate["model_revision"]:
        raise ValueError("probe/candidate model revision mismatch")
    if config["dataset_revision"] != candidate["dataset_revision"]:
        raise ValueError("probe/candidate dataset revision mismatch")
    if config["dataset_revision"] != plan["dataset_revision"]:
        raise ValueError("probe/production-plan dataset revision mismatch")
    if candidate["student"].get("architecture") != "compact_timesfm3":
        raise ValueError("candidate config is not the compact TimesFM-3 student")

    batch_sizes = [int(value) for value in probe["batch_sizes"]]
    if (
        not batch_sizes
        or any(value <= 0 for value in batch_sizes)
        or batch_sizes != sorted(set(batch_sizes))
    ):
        raise ValueError("probe batch sizes must be unique, positive, and increasing")
    if int(probe["warmup_steps"]) < 1 or int(probe["measured_steps"]) < 1:
        raise ValueError("at least one warmup and measured step are required")
    tolerance = float(probe["plateau_tolerance_fraction"])
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("plateau tolerance must be in [0, 1)")

    variants = [str(value) for value in probe["variants"]]
    if len(variants) != len(set(variants)) or not variants:
        raise ValueError("probe variants must be nonempty and unique")
    weights = candidate["training"]["loss_weights"]
    if any(variant not in weights for variant in variants):
        raise ValueError("every probe variant must exist in candidate loss weights")

    plan_items = {str(item["dataset"]): item for item in plan["datasets"]}
    shapes: list[_Shape] = []
    for item in probe["shapes"]:
        dataset = str(item["dataset"])
        planned = plan_items[dataset]
        context = int(item["context"])
        horizon = int(item["horizon"])
        variates = int(item["variates"])
        if planned["view_class"] != "true_multivariate":
            raise ValueError(f"{dataset} is not a true-multivariate production example")
        planned_variates = int(planned["actual_variate_count_distribution"]["max"])
        if (context, horizon, variates) != (
            int(planned["context"]),
            int(planned["horizon"]),
            planned_variates,
        ):
            raise ValueError(f"{dataset} shape differs from the production plan")
        shard_index = int(item.get("shard_index", 0))
        matches = sorted((cache_root / dataset).glob(f"shard-{shard_index:05d}-of-*.npz"))
        if len(matches) != 1:
            raise ValueError(f"expected one cache shard for {dataset} index {shard_index}")
        if not (data_root / dataset).is_dir():
            raise FileNotFoundError(f"missing source dataset {data_root / dataset}")
        sidecar = matches[0].with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        if metadata["sha256"] != _sha256(matches[0]):
            raise ValueError(f"cache checksum failed: {matches[0]}")
        if (
            int(metadata["context_length"]) != context
            or int(metadata["horizon"]) != horizon
            or int(metadata["actual_variates"]) != variates
        ):
            raise ValueError(f"cache sidecar shape mismatch: {matches[0]}")
        with np.load(matches[0]) as shard:
            required = {
                "row_index",
                "context_end",
                "context_length",
                "horizon",
                "teacher_multivariate",
            }
            if not required.issubset(shard.files):
                raise ValueError(f"cache shard schema mismatch: {matches[0]}")
            if len(shard["row_index"]) < max(batch_sizes):
                raise ValueError(f"cache shard too small for maximum batch: {matches[0]}")
            if shard["teacher_multivariate"].shape[1:] != (variates, horizon, 9):
                raise ValueError(f"teacher target shape mismatch: {matches[0]}")
        shapes.append(
            _Shape(
                name=str(item["name"]),
                dataset=dataset,
                context=context,
                horizon=horizon,
                variates=variates,
                production_windows=int(planned["requested_windows"]),
                shard_path=matches[0],
            )
        )

    return candidate, shapes


def _load_shape_data(shape: _Shape, data_root: Path) -> _ShapeData:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    with np.load(shape.shard_path) as shard:
        rows = np.asarray(shard["row_index"], dtype=np.int32).copy()
        ends = np.asarray(shard["context_end"], dtype=np.int32).copy()
        teacher = np.asarray(shard["teacher_multivariate"]).copy()
        if not np.all(shard["context_length"] == shape.context):
            raise ValueError(f"{shape.dataset}: nonuniform cache context")
        if not np.all(shard["horizon"] == shape.horizon):
            raise ValueError(f"{shape.dataset}: nonuniform cache horizon")

    dataset = load_from_disk(str(data_root / shape.dataset), keep_in_memory=False)
    source: dict[int, npt.NDArray[np.float32]] = {}
    for row in np.unique(rows):
        values = np.atleast_2d(np.asarray(dataset[int(row)]["target"], dtype=np.float32))
        if values.shape[0] != shape.variates:
            raise ValueError(f"{shape.dataset}: source/cache variate mismatch")
        source[int(row)] = values
    return _ShapeData(shape, rows, ends, teacher, source)


def _interpolate_context(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Mirror the pinned TimesFM-3 leading-trim and linear-interpolation policy."""

    target = np.asarray(values, dtype=np.float32)
    all_missing = np.isnan(target).all(axis=0)
    first_valid = target.shape[-1] if all_missing.all() else int(np.argmax(~all_missing))
    if first_valid == target.shape[-1]:
        return np.zeros_like(target)
    output = np.full_like(target, np.nan)
    trimmed = target[:, first_valid:].copy()
    for row in trimmed:
        missing = np.isnan(row)
        if missing.any():
            valid = np.flatnonzero(~missing)
            if valid.size:
                row[missing] = np.interp(np.flatnonzero(missing), valid, row[valid])
            else:
                row[missing] = 0.0
    output[:, first_valid:] = trimmed
    return output


def _materialize_numpy(
    data: _ShapeData, batch_size: int, iteration: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(data.row_index)
    # A coprime stride prevents every candidate from timing only the same first rows.
    start = (iteration * batch_size * 104729) % count
    indices = (start + np.arange(batch_size, dtype=np.int64)) % count
    contexts = []
    targets = []
    for index in indices:
        row = int(data.row_index[index])
        end = int(data.context_end[index])
        values = data.source[row]
        contexts.append(_interpolate_context(values[:, end - data.shape.context : end]))
        targets.append(values[:, end : end + data.shape.horizon])
    return (
        np.ascontiguousarray(np.stack(contexts), dtype=np.float32),
        np.ascontiguousarray(np.stack(targets), dtype=np.float32),
        np.ascontiguousarray(data.teacher_primary[indices], dtype=np.float32),
    )


def _pinned_to_device(values: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(values).pin_memory().to(device, non_blocking=True)


def _training_loss(
    model: nn.Module,
    objective: DistillationLoss,
    context: Tensor,
    target: Tensor,
    teacher: Tensor,
    horizon: int,
    epsilon: float,
) -> Tensor:
    prediction = model(context, horizon)
    mean, scale, _ = masked_mean_and_scale(context, epsilon=epsilon)
    forecast_mean = mean.unsqueeze(-1)
    forecast_scale = scale.unsqueeze(-1)
    target_mask = torch.isfinite(target)
    safe_target = torch.where(target_mask, target, mean)
    normalized_target = (safe_target - mean) / scale
    normalized_prediction = (prediction - forecast_mean) / forecast_scale
    normalized_teacher = (teacher - forecast_mean) / forecast_scale
    include_teacher = objective.weights.multivariate_kd > 0 or objective.weights.cvrd > 0
    return objective(
        normalized_prediction,
        normalized_target,
        mask=target_mask,
        teacher_multivariate=normalized_teacher if include_teacher else None,
    )["loss"]


def _full_gradients_finite(model: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)


def _parameters_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _one_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: DistillationLoss,
    data: _ShapeData,
    batch_size: int,
    iteration: int,
    device: torch.device,
    gradient_clip: float,
    epsilon: float,
) -> dict[str, float | bool]:
    wall_start = time.perf_counter()
    prep_start = wall_start
    context_np, target_np, teacher_np = _materialize_numpy(data, batch_size, iteration)
    prep_seconds = time.perf_counter() - prep_start

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    train_start = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    context = _pinned_to_device(context_np, device)
    target = _pinned_to_device(target_np, device)
    teacher = _pinned_to_device(teacher_np, device)
    h2d_end.record()
    train_start.record()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _training_loss(
            model,
            objective,
            context,
            target,
            teacher,
            data.shape.horizon,
            epsilon,
        )
    if not bool(torch.isfinite(loss)):
        raise _NonFiniteStep("non-finite training loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise _NonFiniteStep("non-finite gradient norm")
    optimizer.step()
    train_end.record()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start

    # Full tensor scans are outside the throughput timing.  The clipped norm also
    # guards the optimizer update before it happens.
    gradients_finite = _full_gradients_finite(model)
    parameters_finite = _parameters_finite(model)
    if not gradients_finite or not parameters_finite:
        raise _NonFiniteStep("non-finite gradient tensor or updated parameter")
    return {
        "loss": float(loss.detach()),
        "gradient_norm_before_clip": float(gradient_norm.detach()),
        "gradient_clipped": bool(float(gradient_norm.detach()) > gradient_clip),
        "gradients_finite": gradients_finite,
        "parameters_finite_after_step": parameters_finite,
        "host_materialization_seconds": prep_seconds,
        "h2d_seconds": float(h2d_start.elapsed_time(h2d_end)) / 1000.0,
        "gpu_train_step_seconds": float(train_start.elapsed_time(train_end)) / 1000.0,
        "end_to_end_seconds": wall_seconds,
    }


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _measure_batch(
    *,
    student_config: dict[str, Any],
    weights: dict[str, float],
    data: _ShapeData,
    batch_size: int,
    seed: int,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = build_student(student_config).to(device).train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        objective = DistillationLoss(LossWeights.from_mapping(weights))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
        )
        for iteration in range(warmup_steps):
            _one_step(
                model=model,
                optimizer=optimizer,
                objective=objective,
                data=data,
                batch_size=batch_size,
                iteration=iteration,
                device=device,
                gradient_clip=gradient_clip,
                epsilon=float(student_config["normalization_epsilon"]),
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        measurements = [
            _one_step(
                model=model,
                optimizer=optimizer,
                objective=objective,
                data=data,
                batch_size=batch_size,
                iteration=warmup_steps + iteration,
                device=device,
                gradient_clip=gradient_clip,
                epsilon=float(student_config["normalization_epsilon"]),
            )
            for iteration in range(measured_steps)
        ]
        end_to_end = [float(item["end_to_end_seconds"]) for item in measurements]
        elapsed = math.fsum(end_to_end)
        return {
            "batch_size": batch_size,
            "status": "succeeded",
            "parameter_count": parameter_count,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "measured_windows": batch_size * measured_steps,
            "windows_per_second": batch_size * measured_steps / elapsed,
            "end_to_end_seconds": _summary(end_to_end),
            "host_materialization_seconds": _summary(
                [float(item["host_materialization_seconds"]) for item in measurements]
            ),
            "h2d_seconds": _summary([float(item["h2d_seconds"]) for item in measurements]),
            "gpu_train_step_seconds": _summary(
                [float(item["gpu_train_step_seconds"]) for item in measurements]
            ),
            "loss": _summary([float(item["loss"]) for item in measurements]),
            "gradient_norm_before_clip": _summary(
                [float(item["gradient_norm_before_clip"]) for item in measurements]
            ),
            "gradient_clip_count": sum(bool(item["gradient_clipped"]) for item in measurements),
            "finite_gradients_every_step": all(
                bool(item["gradients_finite"]) for item in measurements
            ),
            "finite_parameters_every_step": all(
                bool(item["parameters_finite_after_step"]) for item in measurements
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except BaseException as error:
        if not _is_oom(error):
            raise
        return {
            "batch_size": batch_size,
            "status": "oom",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def _select_measurement(
    measurements: list[dict[str, Any]], tolerance: float, tested_maximum: int
) -> dict[str, Any]:
    succeeded = [item for item in measurements if item["status"] == "succeeded"]
    if not succeeded:
        raise RuntimeError("every physical batch OOMed")
    peak = max(float(item["windows_per_second"]) for item in succeeded)
    selected = next(
        item
        for item in succeeded
        if float(item["windows_per_second"]) >= peak * (1.0 - tolerance)
    )
    last = succeeded[-1]
    previous = succeeded[-2] if len(succeeded) > 1 else None
    materially_increasing = (
        int(last["batch_size"]) == tested_maximum
        and previous is not None
        and float(last["windows_per_second"])
        > float(previous["windows_per_second"]) * (1.0 + tolerance)
    )
    return {
        "selected_batch_size": int(selected["batch_size"]),
        "selected_windows_per_second": float(selected["windows_per_second"]),
        "peak_batch_size": int(
            max(succeeded, key=lambda item: float(item["windows_per_second"]))["batch_size"]
        ),
        "peak_windows_per_second": peak,
        "throughput_materially_increasing_at_tested_maximum": materially_increasing,
        "maximum_successful_batch_size": int(last["batch_size"]),
        "first_oom_batch_size": next(
            (int(item["batch_size"]) for item in measurements if item["status"] == "oom"),
            None,
        ),
    }


def _gpu_processes(physical_gpu: int) -> list[dict[str, Any]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    mapping = {
        int(index.strip()): uuid.strip()
        for index, uuid in (row.split(",", 1) for row in gpu_rows)
    }
    if physical_gpu not in mapping:
        raise ValueError(f"physical GPU {physical_gpu} does not exist")
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    matches = []
    for row in output.splitlines():
        fields = [field.strip() for field in row.split(",", 3)]
        if len(fields) >= 3 and fields[0] == mapping[physical_gpu]:
            matches.append(
                {
                    "pid": int(fields[1]),
                    "process": fields[2],
                    "used_gpu_memory_mib": fields[3] if len(fields) == 4 else None,
                }
            )
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify pinned config/cache inputs without initializing CUDA or writing evidence",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    candidate, shapes = _validate_and_resolve(config)
    probe = config["probe"]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config": str(args.config),
                    "candidate_config": str(config["candidate_config"]),
                    "variants": probe["variants"],
                    "batch_sizes": probe["batch_sizes"],
                    "shapes": [
                        {
                            "name": shape.name,
                            "dataset": shape.dataset,
                            "context": shape.context,
                            "horizon": shape.horizon,
                            "variates": shape.variates,
                            "cache_shard_sha256": _sha256(shape.shard_path),
                        }
                        for shape in shapes
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = args.output or _root_path(config["output"])
    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(probe["physical_gpu_index"])
    )
    occupied = _gpu_processes(physical_gpu)
    if occupied:
        raise RuntimeError(f"physical GPU {physical_gpu} is occupied: {occupied}")

    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    try:
        device = torch.device(str(probe["device"]))
        torch.cuda.set_device(device)
        if bool(probe.get("require_bfloat16", True)) and not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected GPU does not support BF16")
        data_root = _root_path(config["data_root"])
        batch_sizes = [int(value) for value in probe["batch_sizes"]]
        warmup_steps = int(probe["warmup_steps"])
        measured_steps = int(probe["measured_steps"])
        tolerance = float(probe["plateau_tolerance_fraction"])
        results: list[dict[str, Any]] = []

        for shape in shapes:
            data = _load_shape_data(shape, data_root)
            for variant in probe["variants"]:
                measurements = []
                for batch_size in batch_sizes:
                    item = _measure_batch(
                        student_config=candidate["student"],
                        weights=candidate["training"]["loss_weights"][variant],
                        data=data,
                        batch_size=batch_size,
                        # Both objectives see the identical initialized weights;
                        # shape/batch sweeps also restart from that frozen seed.
                        seed=int(config["seed"]),
                        warmup_steps=warmup_steps,
                        measured_steps=measured_steps,
                        device=device,
                        learning_rate=float(probe["learning_rate"]),
                        weight_decay=float(probe["weight_decay"]),
                        gradient_clip=float(probe["gradient_clip"]),
                    )
                    measurements.append(item)
                    print(
                        f"{shape.name} {variant} batch={batch_size}: "
                        f"{item['status']} {item.get('windows_per_second', '')}",
                        flush=True,
                    )
                    if item["status"] == "oom" and bool(probe.get("stop_after_oom", True)):
                        break
                selection = _select_measurement(measurements, tolerance, max(batch_sizes))
                results.append(
                    {
                        "shape": shape.name,
                        "dataset": shape.dataset,
                        "context": shape.context,
                        "horizon": shape.horizon,
                        "variates": shape.variates,
                        "tokens_per_window": shape.variates
                        * math.ceil(
                            shape.context
                            / int(candidate["student"]["input_patch_length"])
                        ),
                        "production_windows": shape.production_windows,
                        "variant": variant,
                        "measurements": measurements,
                        **selection,
                    }
                )

        metrics: dict[str, float] = {}
        variant_summaries: dict[str, Any] = {}
        for variant in probe["variants"]:
            selected = [item for item in results if item["variant"] == variant]
            conservative = min(float(item["selected_windows_per_second"]) for item in selected)
            weighted_harmonic = math.fsum(
                float(item["production_windows"]) for item in selected
            ) / math.fsum(
                float(item["production_windows"]) / float(item["selected_windows_per_second"])
                for item in selected
            )
            key = str(variant)
            metrics[f"{key}_conservative_windows_per_second"] = conservative
            metrics[f"{key}_representative_weighted_windows_per_second"] = weighted_harmonic
            variant_summaries[key] = {
                "conservative_selected_windows_per_second": conservative,
                "representative_shape_weighted_windows_per_second": weighted_harmonic,
                "stable_manage_json_pointer": (
                    f"/metrics/{key}_conservative_windows_per_second"
                ),
                "all_gradients_finite": all(
                    bool(measurement["finite_gradients_every_step"])
                    for item in selected
                    for measurement in item["measurements"]
                    if measurement["status"] == "succeeded"
                ),
            }
        record.extra.update(
            {
                "schema_version": 1,
                "candidate_config": str(config["candidate_config"]),
                "candidate_config_sha256": _sha256(_root_path(config["candidate_config"])),
                "production_plan": str(config["production_plan"]),
                "production_plan_sha256": _sha256(_root_path(config["production_plan"])),
                "physical_gpu_index": physical_gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                },
                "training_semantics": {
                    "precision": "BF16 autocast with FP32 parameters and loss reductions",
                    "optimizer": "fused AdamW",
                    "timed_unit": (
                        "real source/cache materialization + pinned H2D + forward + loss + "
                        "backward + clip + optimizer + CUDA synchronization"
                    ),
                    "gradient_tensor_scan": "performed after timing on every measured step",
                    "input_preprocessing": "TimesFM-3 linear interpolation",
                    "logical_accumulation": (
                        "none; conservative one optimizer update per probe batch"
                    ),
                },
                "plateau_tolerance_fraction": tolerance,
                "results": results,
                "variant_summaries": variant_summaries,
            }
        )
        record.succeed(metrics)
    except BaseException as error:
        record.fail(f"{type(error).__name__}: {error}")
        record.write(output)
        raise
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
