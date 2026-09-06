#!/usr/bin/env python3
"""Bounded eager-versus-compile probe for the compact production student.

The probe deliberately leaves the production trainer and candidate config
untouched.  It uses the same real production-cache examples, compact model,
view wrapper, BF16 policy, loss, fused optimizer, clipping, and full training
step as the screen.  Compilation wraps only the model callable; checkpoints,
optimizer state, validation, and parameter names remain attached to the raw
student exactly as they would in a promoted implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import autotune_compact_student_training as eager_probe
import numpy as np
import torch
from torch import Tensor, nn
from train_production_student import _StudentViews

from timesfm_lab.config import load_config
from timesfm_lab.distill.losses import DistillationLoss, LossWeights
from timesfm_lab.models import build_student, masked_mean_and_scale
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EAGER_HELPER_PATH = ROOT / "scripts/autotune_compact_student_training.py"
TRAINER_PATH = ROOT / "scripts/train_production_student.py"


@dataclass(slots=True)
class _Snapshot:
    output: Tensor
    loss: float
    gradients: dict[str, Tensor]
    parameters: dict[str, Tensor]
    gradient_norm: float


class _CompileTimeout(TimeoutError):
    """The first Inductor training step exceeded the declared bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dynamo_counters() -> dict[str, dict[str, int | float | str]]:
    from torch._dynamo.utils import counters  # pylint: disable=import-outside-toplevel

    output: dict[str, dict[str, int | float | str]] = {}
    for category, values in counters.items():
        serialized: dict[str, int | float | str] = {}
        for key, value in values.items():
            if isinstance(value, (int, float, str)):
                serialized[str(key)] = value
            else:
                serialized[str(key)] = str(value)
        if serialized:
            output[str(category)] = serialized
    return output


def _unique_graphs(counters: dict[str, dict[str, int | float | str]]) -> int:
    value = counters.get("stats", {}).get("unique_graphs", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


@contextlib.contextmanager
def _alarm_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _CompileTimeout(f"first compiled step exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _loss_from_prediction(
    prediction: Tensor,
    objective: DistillationLoss,
    context: Tensor,
    target: Tensor,
    teacher: Tensor,
    epsilon: float,
) -> Tensor:
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


def _step(
    *,
    callable_model: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: DistillationLoss,
    data: eager_probe._ShapeData,
    batch_size: int,
    iteration: int,
    device: torch.device,
    gradient_clip: float,
    epsilon: float,
    capture_snapshot: bool = False,
) -> tuple[dict[str, Any], _Snapshot | None]:
    wall_start = time.perf_counter()
    prep_start = wall_start
    context_np, target_np, teacher_np = eager_probe._materialize_numpy(data, batch_size, iteration)
    prep_seconds = time.perf_counter() - prep_start

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    train_start = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    context = eager_probe._pinned_to_device(context_np, device)
    target = eager_probe._pinned_to_device(target_np, device)
    teacher = eager_probe._pinned_to_device(teacher_np, device)
    h2d_end.record()
    train_start.record()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction, unused_univariate = callable_model(context, data.shape.horizon, False)
        if unused_univariate is not None:
            raise RuntimeError("KD4 compile probe unexpectedly requested a UV student view")
        loss = _loss_from_prediction(
            prediction,
            objective,
            context,
            target,
            teacher,
            epsilon,
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        student.parameters(), gradient_clip, error_if_nonfinite=True
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite gradient norm")

    captured_output = prediction.detach().float().cpu() if capture_snapshot else None
    captured_gradients = (
        {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in student.named_parameters()
            if parameter.grad is not None
        }
        if capture_snapshot
        else None
    )
    optimizer.step()
    train_end.record()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start

    if not eager_probe._full_gradients_finite(student):
        raise FloatingPointError("non-finite gradient tensor")
    if not eager_probe._parameters_finite(student):
        raise FloatingPointError("non-finite updated parameter")

    snapshot = None
    if capture_snapshot:
        assert captured_output is not None and captured_gradients is not None
        snapshot = _Snapshot(
            output=captured_output,
            loss=float(loss.detach()),
            gradients=captured_gradients,
            parameters={
                name: parameter.detach().float().cpu().clone()
                for name, parameter in student.named_parameters()
            },
            gradient_norm=float(gradient_norm.detach()),
        )

    return (
        {
            "batch_size": batch_size,
            "iteration": iteration,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm.detach()),
            "host_materialization_seconds": prep_seconds,
            "h2d_seconds": float(h2d_start.elapsed_time(h2d_end)) / 1000.0,
            "gpu_train_step_seconds": float(train_start.elapsed_time(train_end)) / 1000.0,
            "end_to_end_seconds": wall_seconds,
            "input_sha256": _array_sha256(context_np, target_np, teacher_np),
        },
        snapshot,
    )


def _tensor_collection_difference(
    reference: dict[str, Tensor], candidate: dict[str, Tensor]
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        return {
            "names_match": False,
            "missing": sorted(set(reference) - set(candidate)),
            "unexpected": sorted(set(candidate) - set(reference)),
        }
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    maximum_absolute = 0.0
    worst_tensor = None
    for name in sorted(reference):
        left = reference[name].double()
        right = candidate[name].double()
        difference = left - right
        local_max = float(difference.abs().max())
        if local_max > maximum_absolute:
            maximum_absolute = local_max
            worst_tensor = name
        difference_square_sum += float(difference.square().sum())
        reference_square_sum += float(left.square().sum())
    return {
        "names_match": True,
        "tensor_count": len(reference),
        "maximum_absolute_difference": maximum_absolute,
        "worst_tensor": worst_tensor,
        "relative_l2_difference": math.sqrt(difference_square_sum)
        / max(math.sqrt(reference_square_sum), 1e-30),
    }


def _correctness(
    eager: _Snapshot, compiled: _Snapshot, tolerance: dict[str, Any]
) -> dict[str, Any]:
    output_difference = (eager.output.double() - compiled.output.double()).abs()
    output_reference = eager.output.double().abs()
    gradients = _tensor_collection_difference(eager.gradients, compiled.gradients)
    parameters = _tensor_collection_difference(eager.parameters, compiled.parameters)
    loss_close = math.isclose(
        eager.loss,
        compiled.loss,
        abs_tol=float(tolerance["loss_atol"]),
        rel_tol=float(tolerance["loss_rtol"]),
    )
    output_close = bool(
        torch.allclose(
            eager.output,
            compiled.output,
            atol=float(tolerance["output_atol"]),
            rtol=float(tolerance["output_rtol"]),
        )
    )
    gradient_close = bool(
        gradients.get("names_match")
        and float(gradients["maximum_absolute_difference"]) <= float(tolerance["gradient_max_abs"])
        and float(gradients["relative_l2_difference"]) <= float(tolerance["gradient_relative_l2"])
    )
    parameter_close = bool(
        parameters.get("names_match")
        and float(parameters["maximum_absolute_difference"])
        <= float(tolerance["parameter_max_abs"])
        and float(parameters["relative_l2_difference"]) <= float(tolerance["parameter_relative_l2"])
    )
    return {
        "passed": loss_close and output_close and gradient_close and parameter_close,
        "loss": {
            "eager": eager.loss,
            "compiled": compiled.loss,
            "absolute_difference": abs(eager.loss - compiled.loss),
            "close": loss_close,
        },
        "output": {
            "maximum_absolute_difference": float(output_difference.max()),
            "maximum_reference_magnitude": float(output_reference.max()),
            "close": output_close,
        },
        "gradient_norm": {
            "eager": eager.gradient_norm,
            "compiled": compiled.gradient_norm,
            "absolute_difference": abs(eager.gradient_norm - compiled.gradient_norm),
        },
        "gradients": gradients | {"close": gradient_close},
        "updated_parameters": parameters | {"close": parameter_close},
        "tolerances": tolerance,
    }


def _measurement_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = math.fsum(float(item["end_to_end_seconds"]) for item in steps)
    windows = sum(int(item["batch_size"]) for item in steps)
    return {
        "steps": len(steps),
        "windows": windows,
        "windows_per_second": windows / elapsed,
        "end_to_end_seconds": eager_probe._summary(
            [float(item["end_to_end_seconds"]) for item in steps]
        ),
        "gpu_train_step_seconds": eager_probe._summary(
            [float(item["gpu_train_step_seconds"]) for item in steps]
        ),
        "host_materialization_seconds": eager_probe._summary(
            [float(item["host_materialization_seconds"]) for item in steps]
        ),
        "h2d_seconds": eager_probe._summary([float(item["h2d_seconds"]) for item in steps]),
        "loss": eager_probe._summary([float(item["loss"]) for item in steps]),
        "input_sha256": [str(item["input_sha256"]) for item in steps],
    }


def _run_engine(
    *,
    engine: str,
    candidate: dict[str, Any],
    config: dict[str, Any],
    data: eager_probe._ShapeData,
    primary_batch: int,
    secondary_batch: int,
    device: torch.device,
) -> tuple[dict[str, Any], _Snapshot]:
    probe = config["probe"]
    torch.manual_seed(int(config["seed"]))
    student = build_student(candidate["student"]).to(device).train()
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    initialization_sha256 = _state_sha256(student)
    if parameter_count != int(probe["expected_parameter_count"]):
        raise ValueError(f"parameter count changed: {parameter_count}")
    if initialization_sha256 != str(probe["expected_initialization_sha256"]):
        raise ValueError(f"initialization changed: {initialization_sha256}")

    raw_views = _StudentViews(student).to(device).train()
    if engine == "compiled":
        compile_config = probe["compile"]
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        callable_model = torch.compile(
            raw_views,
            backend=str(compile_config["backend"]),
            mode=str(compile_config["mode"]),
            fullgraph=bool(compile_config["fullgraph"]),
            dynamic=bool(compile_config["dynamic"]),
        )
    elif engine == "eager":
        callable_model = raw_views
    else:
        raise ValueError(f"unknown engine {engine!r}")

    weights = candidate["training"]["loss_weights"][str(probe["variant"])]
    objective = DistillationLoss(LossWeights.from_mapping(weights))
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(probe["learning_rate"]),
        weight_decay=float(probe["weight_decay"]),
        fused=True,
    )
    torch.cuda.reset_peak_memory_stats(device)

    first_timeout = int(probe["first_compile_timeout_seconds"]) if engine == "compiled" else 0
    with _alarm_timeout(first_timeout):
        correctness_step, snapshot = _step(
            callable_model=callable_model,
            student=student,
            optimizer=optimizer,
            objective=objective,
            data=data,
            batch_size=primary_batch,
            iteration=0,
            device=device,
            gradient_clip=float(probe["gradient_clip"]),
            epsilon=float(candidate["student"]["normalization_epsilon"]),
            capture_snapshot=True,
        )
    assert snapshot is not None
    counters_after_first = _dynamo_counters() if engine == "compiled" else {}

    for iteration in range(1, 1 + int(probe["warmup_steps"])):
        _step(
            callable_model=callable_model,
            student=student,
            optimizer=optimizer,
            objective=objective,
            data=data,
            batch_size=primary_batch,
            iteration=iteration,
            device=device,
            gradient_clip=float(probe["gradient_clip"]),
            epsilon=float(candidate["student"]["normalization_epsilon"]),
        )
    primary_start = 1 + int(probe["warmup_steps"])
    primary_steps = [
        _step(
            callable_model=callable_model,
            student=student,
            optimizer=optimizer,
            objective=objective,
            data=data,
            batch_size=primary_batch,
            iteration=primary_start + index,
            device=device,
            gradient_clip=float(probe["gradient_clip"]),
            epsilon=float(candidate["student"]["normalization_epsilon"]),
        )[0]
        for index in range(int(probe["measured_steps"]))
    ]

    secondary_transition, _ = _step(
        callable_model=callable_model,
        student=student,
        optimizer=optimizer,
        objective=objective,
        data=data,
        batch_size=secondary_batch,
        iteration=primary_start + int(probe["measured_steps"]),
        device=device,
        gradient_clip=float(probe["gradient_clip"]),
        epsilon=float(candidate["student"]["normalization_epsilon"]),
    )
    secondary_start = primary_start + int(probe["measured_steps"]) + 1
    secondary_steps = [
        _step(
            callable_model=callable_model,
            student=student,
            optimizer=optimizer,
            objective=objective,
            data=data,
            batch_size=secondary_batch,
            iteration=secondary_start + index,
            device=device,
            gradient_clip=float(probe["gradient_clip"]),
            epsilon=float(candidate["student"]["normalization_epsilon"]),
        )[0]
        for index in range(int(probe["secondary_measured_steps"]))
    ]
    final_counters = _dynamo_counters() if engine == "compiled" else {}

    result = {
        "engine": engine,
        "parameter_count": parameter_count,
        "initialization_sha256": initialization_sha256,
        "correctness_step": correctness_step,
        "primary": _measurement_summary(primary_steps),
        "secondary_batch_transition": secondary_transition,
        "secondary": _measurement_summary(secondary_steps),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "compiler_counters_after_first_step": counters_after_first,
        "compiler_counters_final": final_counters,
    }
    del optimizer, callable_model, raw_views, student
    gc.collect()
    torch.cuda.empty_cache()
    return result, snapshot


def _validate_config(
    config: dict[str, Any], candidate: dict[str, Any], shapes: list[eager_probe._Shape]
) -> dict[str, Any]:
    probe = config["probe"]
    if str(probe["variant"]) not in candidate["training"]["loss_weights"]:
        raise ValueError("compile probe variant is absent from candidate config")
    if list(probe["variants"]) != [probe["variant"]]:
        raise ValueError("shared validator variants must contain only the compile-probe variant")
    if int(probe["first_compile_timeout_seconds"]) <= 0:
        raise ValueError("first compile timeout must be positive")
    if int(probe["measured_steps"]) < 5:
        raise ValueError("at least five steady-state primary steps are required")
    if int(probe["secondary_measured_steps"]) < 1:
        raise ValueError("at least one secondary-batch measurement is required")
    if float(probe["minimum_conservative_speedup"]) <= 1.0:
        raise ValueError("compile promotion must require a speedup above one")
    if not bool(probe["compile"]["fullgraph"]):
        raise ValueError("bounded probe requires fullgraph=True")
    if not bool(probe["compile"]["dynamic"]):
        raise ValueError("bounded probe requires dynamic=True to test batch-size stability")
    configured_shapes = {str(item["name"]): item for item in probe["shapes"]}
    allowed_batches = {int(value) for value in probe["batch_sizes"]}
    resolved = []
    for shape in shapes:
        item = configured_shapes[shape.name]
        primary = int(item["primary_batch_size"])
        secondary = int(item["secondary_batch_size"])
        if primary not in allowed_batches or secondary not in allowed_batches:
            raise ValueError(f"{shape.name}: per-shape batch absent from validator batch_sizes")
        if primary <= secondary:
            raise ValueError(f"{shape.name}: secondary batch must be smaller than primary")
        resolved.append(
            {
                "name": shape.name,
                "dataset": shape.dataset,
                "context": shape.context,
                "horizon": shape.horizon,
                "variates": shape.variates,
                "primary_batch_size": primary,
                "secondary_batch_size": secondary,
                "cache_shard": str(shape.shard_path.relative_to(ROOT)),
                "cache_shard_sha256": _sha256(shape.shard_path),
            }
        )
    return {
        "parameter_count": int(probe["expected_parameter_count"]),
        "expected_initialization_sha256": str(probe["expected_initialization_sha256"]),
        "shapes": resolved,
    }


def _cpu_initialization_audit(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    student = build_student(candidate["student"])
    result = {
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "initialization_sha256": _state_sha256(student),
    }
    del student
    if result["parameter_count"] != int(config["probe"]["expected_parameter_count"]):
        raise ValueError(f"CPU preflight parameter count mismatch: {result}")
    if result["initialization_sha256"] != str(config["probe"]["expected_initialization_sha256"]):
        raise ValueError(f"CPU preflight initialization mismatch: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    candidate, shapes = eager_probe._validate_and_resolve(config)
    validation = _validate_config(config, candidate, shapes)
    cpu_audit = _cpu_initialization_audit(config, candidate)
    fingerprints = {
        "probe_script_sha256": _sha256(SCRIPT_PATH),
        "eager_helper_script_sha256": _sha256(EAGER_HELPER_PATH),
        "production_trainer_script_sha256": _sha256(TRAINER_PATH),
        "config_sha256": _sha256(args.config.resolve()),
        "candidate_config_sha256": _sha256(eager_probe._root_path(config["candidate_config"])),
        "production_plan_sha256": _sha256(eager_probe._root_path(config["production_plan"])),
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated_cpu_only",
                    "config": str(args.config),
                    "validation": validation,
                    "cpu_initialization_audit": cpu_audit,
                    "fingerprints": fingerprints,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = args.output or eager_probe._root_path(config["output"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(config["probe"]["physical_gpu_index"])
    )
    occupied = eager_probe._gpu_processes(physical_gpu)
    if occupied:
        raise RuntimeError(f"physical GPU {physical_gpu} is occupied: {occupied}")
    if bool(config["probe"].get("require_isolated_visible_device", True)):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(physical_gpu):
            raise RuntimeError(
                "compile probe requires one isolated visible GPU: "
                f"CUDA_VISIBLE_DEVICES={physical_gpu}, found {visible!r}"
            )

    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    temporary_cache: str | None = None
    try:
        if bool(config["probe"]["compile"].get("fresh_temporary_cache", True)):
            temporary_cache = tempfile.mkdtemp(prefix="timesfm-compile-probe-")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = temporary_cache
            os.environ["TRITON_CACHE_DIR"] = str(Path(temporary_cache) / "triton")

        device = torch.device(str(config["probe"]["device"]))
        torch.cuda.set_device(device)
        require_bfloat16 = bool(config["probe"].get("require_bfloat16", True))
        if require_bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected GPU does not support BF16")

        data_root = eager_probe._root_path(config["data_root"])
        configured_shapes = {str(item["name"]): item for item in config["probe"]["shapes"]}
        results = []
        for shape in shapes:
            shape_config = configured_shapes[shape.name]
            data = eager_probe._load_shape_data(shape, data_root)
            primary = int(shape_config["primary_batch_size"])
            secondary = int(shape_config["secondary_batch_size"])
            eager_result, eager_snapshot = _run_engine(
                engine="eager",
                candidate=candidate,
                config=config,
                data=data,
                primary_batch=primary,
                secondary_batch=secondary,
                device=device,
            )
            compiled_result, compiled_snapshot = _run_engine(
                engine="compiled",
                candidate=candidate,
                config=config,
                data=data,
                primary_batch=primary,
                secondary_batch=secondary,
                device=device,
            )
            correctness = _correctness(
                eager_snapshot,
                compiled_snapshot,
                config["probe"]["correctness"],
            )
            if (
                eager_result["correctness_step"]["input_sha256"]
                != compiled_result["correctness_step"]["input_sha256"]
            ):
                raise RuntimeError("eager/compiled correctness batches differ")
            unique_graphs = _unique_graphs(compiled_result["compiler_counters_final"])
            maximum_graphs = int(config["probe"]["compile"]["maximum_unique_graphs_per_shape"])
            graph_stable = unique_graphs <= maximum_graphs
            primary_speedup = float(compiled_result["primary"]["windows_per_second"]) / float(
                eager_result["primary"]["windows_per_second"]
            )
            secondary_speedup = float(compiled_result["secondary"]["windows_per_second"]) / float(
                eager_result["secondary"]["windows_per_second"]
            )
            result = {
                "shape": shape.name,
                "dataset": shape.dataset,
                "context": shape.context,
                "horizon": shape.horizon,
                "variates": shape.variates,
                "primary_batch_size": primary,
                "secondary_batch_size": secondary,
                "cache_shard_sha256": _sha256(shape.shard_path),
                "eager": eager_result,
                "compiled": compiled_result,
                "correctness": correctness,
                "unique_graphs": unique_graphs,
                "maximum_unique_graphs": maximum_graphs,
                "dynamic_batch_graph_stable": graph_stable,
                "primary_speedup": primary_speedup,
                "secondary_speedup": secondary_speedup,
                "conservative_speedup": min(primary_speedup, secondary_speedup),
            }
            results.append(result)
            print(
                f"{shape.name}: correct={correctness['passed']} graphs={unique_graphs} "
                f"primary={primary_speedup:.3f}x secondary={secondary_speedup:.3f}x",
                flush=True,
            )

        conservative_speedup = min(float(item["conservative_speedup"]) for item in results)
        correctness_passed = all(bool(item["correctness"]["passed"]) for item in results)
        graph_stable = all(bool(item["dynamic_batch_graph_stable"]) for item in results)
        promotion_threshold = float(config["probe"]["minimum_conservative_speedup"])
        promoted = (
            correctness_passed and graph_stable and conservative_speedup >= promotion_threshold
        )
        record.extra.update(
            {
                "schema_version": 1,
                "fingerprints": fingerprints,
                "validation": validation,
                "cpu_initialization_audit": cpu_audit,
                "candidate_config": str(config["candidate_config"]),
                "production_plan": str(config["production_plan"]),
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
                "compile": config["probe"]["compile"],
                "fresh_temporary_compile_cache": temporary_cache is not None,
                "training_semantics": {
                    "precision": "BF16 autocast with FP32 parameters and loss reductions",
                    "optimizer": "fused AdamW",
                    "compiled_boundary": "_StudentViews forward only",
                    "optimizer_clip_checkpoint_owner": "uncompiled CompactTimesFM3Student",
                    "objective": str(config["probe"]["variant"]),
                    "input_preprocessing": "TimesFM-3 linear interpolation",
                    "timed_unit": (
                        "real source/cache materialization + pinned H2D + forward + loss + "
                        "backward + clip + optimizer + CUDA synchronization"
                    ),
                },
                "results": results,
                "promotion_rule": {
                    "minimum_conservative_speedup": promotion_threshold,
                    "requires_correctness": True,
                    "requires_dynamic_batch_graph_stability": True,
                    "passed": promoted,
                },
            }
        )
        record.succeed(
            {
                "conservative_speedup": conservative_speedup,
                "correctness_passed": float(correctness_passed),
                "dynamic_batch_graph_stable": float(graph_stable),
                "promote_compile": float(promoted),
            }
        )
    except BaseException as error:
        record.extra.update(
            {
                "schema_version": 1,
                "fingerprints": fingerprints,
                "validation": validation,
                "cpu_initialization_audit": cpu_audit,
                "physical_gpu_index": physical_gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        record.fail(f"{type(error).__name__}: {error}")
        record.write(output)
        raise
    finally:
        if temporary_cache is not None:
            shutil.rmtree(temporary_cache, ignore_errors=True)

    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
