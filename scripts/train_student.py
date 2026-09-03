#!/usr/bin/env python3
"""Train matched GT, KD, Dual-View KD, or CVRD students with torch DDP."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import split_cache_indices
from timesfm_lab.distill.losses import DistillationLoss, LossWeights
from timesfm_lab.models import StudentConfig, TimesFMStudent, masked_mean_and_scale
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _load_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(cache_dir.glob("shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no cache shards found in {cache_dir}")
    pieces: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as shard:
            for key in shard.files:
                pieces.setdefault(key, []).append(shard[key])
    return {key: np.concatenate(values) for key, values in pieces.items()}


def _load_source(data_root: Path, name: str) -> list[np.ndarray]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_root / name))
    return [np.atleast_2d(np.asarray(row["target"], dtype=np.float32)) for row in dataset]


def _materialize(
    cache: dict[str, np.ndarray],
    source: list[np.ndarray],
    indices: np.ndarray,
    context_length: int,
    horizon: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    contexts = []
    targets = []
    for index in indices:
        row = int(cache["row_index"][index])
        end = int(cache["context_end"][index])
        values = source[row]
        contexts.append(values[:, end - context_length : end])
        targets.append(values[:, end : end + horizon])
    context = torch.from_numpy(np.stack(contexts)).to(device=device, non_blocking=True)
    target = torch.from_numpy(np.stack(targets)).to(device=device, non_blocking=True)
    teacher_mv = torch.from_numpy(cache["teacher_multivariate"][indices]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher_uv = torch.from_numpy(cache["teacher_univariate"][indices]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return context, target, teacher_mv, teacher_uv


def _normalization(context: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    mean, scale, _ = masked_mean_and_scale(context, epsilon=epsilon)
    return mean.unsqueeze(-1), scale.unsqueeze(-1)


def _canonical_variant(variant: str) -> str:
    # Internal alias retained solely so historical pilot invocations remain identifiable.
    return "cvrd" if variant == "relkd" else variant


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _student_univariate(model: Any, context: Tensor, horizon: int) -> Tensor:
    batch, variates, length = context.shape
    outputs = model(context.reshape(batch * variates, 1, length), horizon)
    return outputs.reshape(batch, variates, horizon, 9)


def _losses(
    model: Any,
    objective: DistillationLoss,
    variant: str,
    context: Tensor,
    target: Tensor,
    teacher_mv: Tensor,
    teacher_uv: Tensor,
    horizon: int,
    normalization_epsilon: float,
) -> tuple[dict[str, Tensor], Tensor, Tensor | None]:
    variant = _canonical_variant(variant)
    prediction = model(context, horizon)
    has_cross_variate_context = context.shape[1] > 1
    prediction_uv = (
        _student_univariate(model, context, horizon)
        if variant in {"dual_view", "cvrd"} and has_cross_variate_context
        else None
    )
    mean, scale = _normalization(context, normalization_epsilon)
    target_mask = torch.isfinite(target)
    safe_target = torch.where(target_mask, target, mean.squeeze(-1))
    normalized_target = (safe_target - mean.squeeze(-1)) / scale.squeeze(-1)
    normalized_prediction = (prediction - mean) / scale
    normalized_teacher_mv = (teacher_mv - mean) / scale
    normalized_prediction_uv = (
        (prediction_uv - mean) / scale if prediction_uv is not None else None
    )
    normalized_teacher_uv = (teacher_uv - mean) / scale
    values = objective(
        normalized_prediction,
        normalized_target,
        mask=target_mask,
        teacher_multivariate=normalized_teacher_mv if variant != "gt" else None,
        student_univariate=normalized_prediction_uv,
        teacher_univariate=normalized_teacher_uv if prediction_uv is not None else None,
    )
    return values, normalized_prediction, normalized_prediction_uv


def _validation(
    model: Any,
    cache: dict[str, np.ndarray],
    source: list[np.ndarray],
    indices: np.ndarray,
    context_length: int,
    horizon: int,
    device: torch.device,
) -> dict[str, float]:
    batch_indices = indices[: min(32, len(indices))]
    context, target, teacher_mv, teacher_uv = _materialize(
        cache, source, batch_indices, context_length, horizon, device
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(context, horizon)
        prediction_uv = _student_univariate(model, context, horizon)
    mean, scale = _normalization(context, model.config.normalization_epsilon)
    prediction = (prediction.float() - mean) / scale
    prediction_uv = (prediction_uv.float() - mean) / scale
    teacher_mv = (teacher_mv - mean) / scale
    teacher_uv = (teacher_uv - mean) / scale
    student_response = (prediction - prediction_uv).flatten()
    teacher_response = (teacher_mv - teacher_uv).flatten()
    centered_student = student_response - student_response.mean()
    centered_teacher = teacher_response - teacher_response.mean()
    correlation = (centered_student * centered_teacher).mean() / (
        centered_student.square().mean().sqrt() * centered_teacher.square().mean().sqrt()
    ).clamp_min(1e-12)
    response_nmae = (student_response - teacher_response).abs().mean() / (
        teacher_response.abs().mean().clamp_min(1e-12)
    )
    target_mask = torch.isfinite(target)
    safe_target = torch.where(target_mask, target, mean.squeeze(-1))
    normalized_target = (safe_target - mean.squeeze(-1)) / scale.squeeze(-1)
    levels = prediction.new_tensor((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    error = normalized_target.unsqueeze(-1) - prediction
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    expanded_mask = target_mask.unsqueeze(-1).expand_as(pinball)
    gt_loss = pinball.masked_select(expanded_mask).mean()
    return {
        "ground_truth_pinball": float(gt_loss),
        "response_correlation": float(correlation),
        "response_normalized_mae": float(response_nmae),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--variant",
        choices=("gt", "kd", "dual_view", "cvrd", "relkd"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("retired"):
        raise ValueError(f"refusing to run retired config: {config['retired']}")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) + rank)

    dataset_names = list(config.get("datasets", [config.get("dataset")]))
    if not dataset_names or dataset_names == [None]:
        raise ValueError("training config must define dataset or datasets")
    diverse = len(dataset_names) > 1
    cache_paths: dict[str, Path] = {}
    for name in dataset_names:
        candidates = [root / name if diverse else root for root in args.cache_dir]
        try:
            cache_paths[name] = next(
                candidate for candidate in candidates if any(candidate.glob("shard-*.npz"))
            )
        except StopIteration as error:
            raise FileNotFoundError(f"no cache root contains dataset {name}") from error
    caches = {name: _load_cache(path) for name, path in cache_paths.items()}
    sources = {name: _load_source(args.data_root, name) for name in dataset_names}
    cache_context = int(config.get("cache_context_length", 512))
    horizons = {int(cache["teacher_multivariate"].shape[-2]) for cache in caches.values()}
    if len(horizons) != 1:
        raise ValueError(f"all caches must use one horizon, received {sorted(horizons)}")
    cache_horizon = horizons.pop()

    splits: dict[str, dict[str, np.ndarray]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    split_mode = config["training"].get("validation_split", "random_windows")
    for name in dataset_names:
        splits[name], split_reports[name] = split_cache_indices(
            caches[name]["row_index"],
            caches[name]["context_end"],
            context_length=cache_context,
            horizon=cache_horizon,
            validation_fraction=float(config["training"]["validation_fraction"]),
            seed=int(config["seed"]),
            mode=split_mode,
        )

    bare_model = TimesFMStudent(StudentConfig(**config["student"]))
    initialization_sha256 = _state_sha256(bare_model)
    bare_model.to(device)
    parameter_count = bare_model.parameter_count
    model: Any = (
        DistributedDataParallel(bare_model, device_ids=[local_rank])
        if world_size > 1
        else bare_model
    )
    variant = _canonical_variant(args.variant)
    weight_configs = config["training"]["loss_weights"]
    weight_key = args.variant if args.variant in weight_configs else variant
    weights = LossWeights.from_mapping(weight_configs[weight_key])
    objective = DistillationLoss(weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = int(config["training"]["steps"])
    min_lr = float(config["training"]["min_learning_rate"])
    max_lr = float(config["training"]["learning_rate"])
    per_gpu_batch = int(config["training"]["per_gpu_batch"])
    generator = np.random.default_rng(int(config["seed"]) * 100 + rank)
    training_sequence_sha256 = hashlib.sha256()
    record = (
        RunRecord.start(
            run_id=f"{config['run_id']}-{variant}",
            config_path=str(args.config),
            seed=int(config["seed"]),
            model_revision=config["model_revision"],
            dataset_revision=config["dataset_revision"],
            hardware_snapshot=config["hardware_snapshot"],
            repository=ROOT,
        )
        if rank == 0
        else None
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metric_sums: dict[str, Tensor] = {}
    model.train()
    for step in range(steps):
        dataset_name = dataset_names[step % len(dataset_names)]
        training_indices = splits[dataset_name]["training"]
        selected = generator.choice(training_indices, size=per_gpu_batch, replace=False)
        training_sequence_sha256.update(dataset_name.encode())
        training_sequence_sha256.update(np.asarray(selected, dtype="<i8").tobytes())
        context, target, teacher_mv, teacher_uv = _materialize(
            caches[dataset_name],
            sources[dataset_name],
            selected,
            cache_context,
            cache_horizon,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values, _, _ = _losses(
                model,
                objective,
                variant,
                context,
                target,
                teacher_mv,
                teacher_uv,
                cache_horizon,
                bare_model.config.normalization_epsilon,
            )
        if not all(torch.isfinite(value) for value in values.values()):
            raise FloatingPointError(
                f"non-finite loss at step {step + 1} on dataset {dataset_name}"
            )
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip"])
        )
        optimizer.step()
        progress = (step + 1) / steps
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        for key, value in values.items():
            metric_sums[key] = (
                metric_sums.get(key, torch.zeros_like(value.detach())) + value.detach()
            )
        if rank == 0 and ((step + 1) % 10 == 0 or step == 0):
            print(
                f"variant={variant} dataset={dataset_name} step={step + 1}/{steps} "
                f"loss={float(values['loss'].detach()):.6f} "
                f"gt={float(values['ground_truth'].detach()):.6f} "
                f"mv_kd={float(values['multivariate_kd'].detach()):.6f} "
                f"uv_kd={float(values['univariate_kd'].detach()):.6f} "
                f"cvrd={float(values['cvrd'].detach()):.6f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if world_size > 1:
        for value in metric_sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.barrier()

    if rank == 0:
        model.eval()
        validation_by_dataset = {
            name: _validation(
                bare_model,
                caches[name],
                sources[name],
                splits[name]["validation"],
                cache_context,
                cache_horizon,
                device,
            )
            for name in dataset_names
        }
        validation = {
            key: float(np.mean([values[key] for values in validation_by_dataset.values()]))
            for key in next(iter(validation_by_dataset.values()))
        }
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.checkpoint_dir / f"student-{variant}.pt"
        torch.save(bare_model.state_dict(), checkpoint)
        assert record is not None
        denominator = steps * world_size
        train_metrics = {key: float(value / denominator) for key, value in metric_sums.items()}
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in validation.items()},
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            record.fail("non-finite training or validation metric")
            output = Path("results/reproduction/distillation") / (
                f"{config['run_id']}-{variant}.json"
            )
            record.write(output)
            raise FloatingPointError("non-finite training or validation metric")
        record.extra.update(
            {
                "variant": variant,
                "legacy_variant_alias": args.variant if args.variant != variant else None,
                "datasets": dataset_names,
                "validation_by_dataset": validation_by_dataset,
                "parameter_count": parameter_count,
                "initialization_sha256": initialization_sha256,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_distributed": False,
                "training": {
                    "steps": steps,
                    "world_size": world_size,
                    "per_gpu_batch": per_gpu_batch,
                    "global_batch": per_gpu_batch * world_size,
                    "windows_processed": steps * per_gpu_batch * world_size,
                    "elapsed_seconds": elapsed,
                    "windows_per_second": steps * per_gpu_batch * world_size / elapsed,
                    "precision": config["training"]["precision"],
                    "optimizer": "AdamW",
                    "schedule": "cosine",
                    "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(device),
                    "loss_weights": {
                        "ground_truth": weights.ground_truth,
                        "multivariate_kd": weights.multivariate_kd,
                        "univariate_kd": weights.univariate_kd,
                        "cvrd": weights.cvrd,
                    },
                    "normalization_epsilon": bare_model.config.normalization_epsilon,
                    "rank0_training_sequence_sha256": training_sequence_sha256.hexdigest(),
                },
                "split": {
                    name: {
                        "training": len(indices["training"]),
                        "validation": len(indices["validation"]),
                        "method": split_reports[name],
                    }
                    for name, indices in splits.items()
                },
            }
        )
        record.succeed(metrics)
        output = Path("results/reproduction/distillation") / (
            f"{config['run_id']}-{variant}.json"
        )
        record.write(output)
        print(output, flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
