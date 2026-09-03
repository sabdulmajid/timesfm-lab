#!/usr/bin/env python3
"""Train one matched production student variant to validation convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import split_cache_indices
from timesfm_lab.distill.losses import DistillationLoss, LossWeights, pinball_loss
from timesfm_lab.models import StudentConfig, TimesFMStudent, masked_mean_and_scale
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _Corpus:
    name: str
    domain: str
    view_class: str
    context_length: int
    horizon: int
    batch_size: int
    row_index: npt.NDArray[np.int32]
    context_end: npt.NDArray[np.int32]
    teacher_primary: npt.NDArray[Any]
    teacher_univariate: npt.NDArray[Any] | None
    source: list[npt.NDArray[np.float32]]
    training_indices: npt.NDArray[Any]
    validation_indices: npt.NDArray[Any]
    split_report: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_size(config: dict[str, Any], context: int) -> int:
    mapping = {int(key): int(value) for key, value in config.items()}
    if context not in mapping:
        raise KeyError(f"no student batch size configured for context={context}")
    return mapping[context]


def _load_corpus(
    item: dict[str, Any],
    *,
    data_root: Path,
    cache_root: Path,
    validation_fraction: float,
    validation_mode: str,
    seed: int,
    batch_sizes: dict[str, Any],
) -> _Corpus:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    name = str(item["dataset"])
    cache_paths = sorted((cache_root / name).glob("shard-*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"no production cache shards for {name}")
    pieces: dict[str, list[np.ndarray]] = {}
    for path in cache_paths:
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        if metadata["sha256"] != _sha256(path):
            raise ValueError(f"cache checksum failed: {path}")
        with np.load(path) as shard:
            for key in shard.files:
                pieces.setdefault(key, []).append(shard[key])
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    expected_windows = int(item["requested_windows"])
    if len(arrays["row_index"]) != expected_windows:
        raise ValueError(f"cache window count disagrees with plan for {name}")
    context_length = int(item["context"])
    horizon = int(item["horizon"])
    if not np.all(arrays["context_length"] == context_length) or not np.all(
        arrays["horizon"] == horizon
    ):
        raise ValueError(f"cache shape metadata disagrees with plan for {name}")
    true_multivariate = item["view_class"] == "true_multivariate"
    if true_multivariate:
        teacher_primary = arrays["teacher_multivariate"]
        teacher_univariate = arrays["teacher_univariate"]
        if "teacher_output" in arrays:
            raise ValueError(f"true-MV cache contains UV-only schema for {name}")
    else:
        teacher_primary = arrays["teacher_output"]
        teacher_univariate = None
        if "teacher_multivariate" in arrays or "teacher_univariate" in arrays:
            raise ValueError(f"V=1 cache contains manufactured MV/UV supervision for {name}")

    dataset = load_from_disk(str(data_root / name), keep_in_memory=False)
    source = [
        np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
        for row in range(len(dataset))
    ]
    actual_variates = {values.shape[0] for values in source}
    if len(actual_variates) != 1 or (max(actual_variates) > 1) != true_multivariate:
        raise ValueError(f"actual target shape disagrees with view class for {name}")
    splits, split_report = split_cache_indices(
        arrays["row_index"],
        arrays["context_end"],
        context_length=context_length,
        horizon=horizon,
        validation_fraction=validation_fraction,
        seed=seed,
        mode=validation_mode,
    )
    return _Corpus(
        name=name,
        domain=str(item["domain"]),
        view_class=str(item["view_class"]),
        context_length=context_length,
        horizon=horizon,
        batch_size=_batch_size(batch_sizes, context_length),
        row_index=arrays["row_index"],
        context_end=arrays["context_end"],
        teacher_primary=teacher_primary,
        teacher_univariate=teacher_univariate,
        source=source,
        training_indices=splits["training"],
        validation_indices=splits["validation"],
        split_report=split_report,
    )


def _epoch_batches(
    corpora: list[_Corpus], seed: int, epoch: int
) -> list[tuple[int, npt.NDArray[Any]]]:
    rng = np.random.default_rng(seed * 100_000 + epoch)
    batches = []
    for corpus_index, corpus in enumerate(corpora):
        order = rng.permutation(corpus.training_indices)
        for start in range(0, len(order), corpus.batch_size):
            batches.append((corpus_index, order[start : start + corpus.batch_size]))
    rng.shuffle(batches)
    return batches


def _to_device(
    values: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values)).pin_memory()
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _materialize(
    corpus: _Corpus,
    indices: npt.NDArray[Any],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    contexts = []
    targets = []
    for index in indices:
        row = int(corpus.row_index[index])
        end = int(corpus.context_end[index])
        values = corpus.source[row]
        contexts.append(values[:, end - corpus.context_length : end])
        targets.append(values[:, end : end + corpus.horizon])
    context = _to_device(np.stack(contexts), device)
    target = _to_device(np.stack(targets), device)
    teacher_primary = _to_device(corpus.teacher_primary[indices], device, torch.float32)
    teacher_uv = (
        _to_device(corpus.teacher_univariate[indices], device, torch.float32)
        if corpus.teacher_univariate is not None
        else None
    )
    return context, target, teacher_primary, teacher_uv


def _student_univariate(model: Any, context: Tensor, horizon: int) -> Tensor:
    batch, variates, length = context.shape
    output: Tensor = model(context.reshape(batch * variates, 1, length), horizon)
    return output.reshape(batch, variates, horizon, 9)


def _normalized(
    context: Tensor,
    target: Tensor,
    student_mv: Tensor,
    teacher_primary: Tensor,
    student_uv: Tensor | None,
    teacher_uv: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
    mean, scale, _ = masked_mean_and_scale(context, epsilon=epsilon)
    forecast_mean = mean.unsqueeze(-1)
    forecast_scale = scale.unsqueeze(-1)
    target_mask = torch.isfinite(target)
    safe_target = torch.where(target_mask, target, mean)
    normalized_target = (safe_target - mean) / scale
    return (
        normalized_target,
        (student_mv - forecast_mean) / forecast_scale,
        (teacher_primary - forecast_mean) / forecast_scale,
        (student_uv - forecast_mean) / forecast_scale if student_uv is not None else None,
        (teacher_uv - forecast_mean) / forecast_scale if teacher_uv is not None else None,
        target_mask,
    )


def _loss(
    model: Any,
    objective: DistillationLoss,
    variant: str,
    corpus: _Corpus,
    context: Tensor,
    target: Tensor,
    teacher_primary: Tensor,
    teacher_uv: Tensor | None,
    epsilon: float,
) -> dict[str, Tensor]:
    student_mv = model(context, corpus.horizon)
    student_uv = (
        _student_univariate(model, context, corpus.horizon)
        if variant in {"dual_view", "cvrd"} and corpus.view_class == "true_multivariate"
        else None
    )
    normalized_target, student_mv, teacher_primary, student_uv, teacher_uv, mask = _normalized(
        context,
        target,
        student_mv,
        teacher_primary,
        student_uv,
        teacher_uv,
        epsilon,
    )
    return objective(
        student_mv,
        normalized_target,
        mask=mask,
        teacher_multivariate=teacher_primary if variant != "gt" else None,
        student_univariate=student_uv,
        teacher_univariate=teacher_uv if student_uv is not None else None,
    )


def _response_summary(
    student_response: Tensor, teacher_response: Tensor, mask: Tensor
) -> dict[str, float]:
    expanded = mask.unsqueeze(-1).expand_as(student_response)
    student = student_response.masked_select(expanded).double()
    teacher = teacher_response.masked_select(expanded).double()
    count = max(student.numel(), 1)
    student_centered = student - student.mean()
    teacher_centered = teacher - teacher.mean()
    denominator = (
        student_centered.square().sum().sqrt() * teacher_centered.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "count": float(count),
        "pearson": float((student_centered * teacher_centered).sum() / denominator),
        "nmae": float(
            (student - teacher).abs().sum() / teacher.abs().sum().clamp_min(1e-12)
        ),
        "sign_agreement": float((torch.sign(student) == torch.sign(teacher)).double().mean()),
        "cosine": float(
            (student * teacher).sum()
            / (student.square().sum().sqrt() * teacher.square().sum().sqrt()).clamp_min(1e-12)
        ),
        "magnitude_mae": float((student.abs() - teacher.abs()).abs().mean()),
    }


@torch.inference_mode()
def _validate(
    model: TimesFMStudent,
    corpora: list[_Corpus],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_student = 0.0
    total_teacher = 0.0
    total_weight = 0
    by_dataset = {}
    for corpus in corpora:
        student_sum = 0.0
        teacher_sum = 0.0
        weight_sum = 0
        response_parts: list[dict[str, float]] = []
        for start in range(0, len(corpus.validation_indices), corpus.batch_size):
            indices = corpus.validation_indices[start : start + corpus.batch_size]
            context, target, teacher_primary, teacher_uv = _materialize(corpus, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_mv = model(context, corpus.horizon)
                student_uv = (
                    _student_univariate(model, context, corpus.horizon)
                    if corpus.view_class == "true_multivariate"
                    else None
                )
            normalized_target, student_mv, teacher_primary, student_uv, teacher_uv, mask = (
                _normalized(
                    context,
                    target,
                    student_mv.float(),
                    teacher_primary,
                    student_uv.float() if student_uv is not None else None,
                    teacher_uv,
                    model.config.normalization_epsilon,
                )
            )
            weight = int(mask.sum())
            student_value = float(pinball_loss(student_mv, normalized_target, mask))
            teacher_value = float(pinball_loss(teacher_primary, normalized_target, mask))
            student_sum += student_value * weight
            teacher_sum += teacher_value * weight
            weight_sum += weight
            if student_uv is not None and teacher_uv is not None:
                response_parts.append(
                    _response_summary(student_mv - student_uv, teacher_primary - teacher_uv, mask)
                )
        dataset_result: dict[str, Any] = {
            "student_pinball": student_sum / weight_sum,
            "teacher_pinball": teacher_sum / weight_sum,
            "observed_targets": weight_sum,
            "windows": len(corpus.validation_indices),
        }
        if response_parts:
            response_weight = sum(item["count"] for item in response_parts)
            dataset_result["response"] = {
                key: sum(item[key] * item["count"] for item in response_parts) / response_weight
                for key in ("pearson", "nmae", "sign_agreement", "cosine", "magnitude_mae")
            }
        by_dataset[corpus.name] = dataset_result
        total_student += student_sum
        total_teacher += teacher_sum
        total_weight += weight_sum
    model.train()
    return {
        "student_pinball": total_student / total_weight,
        "teacher_pinball": total_teacher / total_weight,
        "observed_targets": total_weight,
        "by_dataset": by_dataset,
    }


def _save_checkpoint(
    path: Path,
    model: TimesFMStudent,
    optimizer: torch.optim.Optimizer,
    **state: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), **state}, temporary
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--variant", choices=("gt", "kd", "dual_view", "cvrd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    training = config["training"]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    load_started = time.perf_counter()
    corpora = [
        _load_corpus(
            item,
            data_root=args.data_root,
            cache_root=args.cache_root,
            validation_fraction=float(training["validation_fraction"]),
            validation_mode=str(training["validation_split"]),
            seed=seed,
            batch_sizes=training["batch_size_by_context"],
        )
        for item in plan["datasets"]
    ]
    corpus_load_seconds = time.perf_counter() - load_started
    model = TimesFMStudent(StudentConfig(**config["student"]))
    initialization_sha256 = _state_sha256(model)
    model.to(device)
    weights = LossWeights.from_mapping(training["loss_weights"][args.variant])
    objective = DistillationLoss(weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    max_steps = args.max_steps or int(training["max_steps"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    batch_offset = 0
    best_score = math.inf
    stale_evaluations = 0
    learning_curve: list[dict[str, Any]] = []
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        epoch = int(state["epoch"])
        batch_offset = int(state["batch_offset"])
        best_score = float(state["best_score"])
        stale_evaluations = int(state["stale_evaluations"])
        learning_curve = list(state["learning_curve"])

    record = RunRecord.start(
        run_id=f"{config['run_id']}-{args.variant}",
        config_path=f"{args.config};{args.plan}",
        seed=seed,
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    sequence_digest = hashlib.sha256()
    train_sums = {
        key: 0.0
        for key in (
            "loss",
            "ground_truth",
            "multivariate_kd",
            "univariate_kd",
            "cvrd",
        )
    }
    trained_windows = 0
    stopped_for_plateau = False
    started = time.perf_counter()
    model.train()
    while step < max_steps and not stopped_for_plateau:
        batches = _epoch_batches(corpora, seed, epoch)
        for offset in range(batch_offset, len(batches)):
            corpus_index, indices = batches[offset]
            corpus = corpora[corpus_index]
            sequence_digest.update(corpus.name.encode())
            sequence_digest.update(np.asarray(indices, dtype="<i8").tobytes())
            context, target, teacher_primary, teacher_uv = _materialize(corpus, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = _loss(
                    model,
                    objective,
                    args.variant,
                    corpus,
                    context,
                    target,
                    teacher_primary,
                    teacher_uv,
                    model.config.normalization_epsilon,
                )
            if not all(torch.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"non-finite loss at step {step + 1} on {corpus.name}")
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            step += 1
            trained_windows += len(indices)
            progress = step / max_steps
            lr = float(training["min_learning_rate"]) + 0.5 * (
                float(training["learning_rate"]) - float(training["min_learning_rate"])
            ) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            for key in train_sums:
                train_sums[key] += float(values[key].detach()) * len(indices)
            if step == 1 or step % 100 == 0:
                print(
                    f"variant={args.variant} step={step}/{max_steps} epoch={epoch} "
                    f"dataset={corpus.name} batch={len(indices)} loss={float(values['loss']):.6f} "
                    f"windows_per_second={trained_windows / (time.perf_counter() - started):.1f}",
                    flush=True,
                )

            next_epoch = epoch
            next_offset = offset + 1
            if next_offset == len(batches):
                next_epoch += 1
                next_offset = 0
            validation_every = int(training["validation_every_steps"])
            should_validate = step % validation_every == 0 or step == max_steps
            if should_validate:
                validation_started = time.perf_counter()
                validation = _validate(model, corpora, device)
                validation_seconds = time.perf_counter() - validation_started
                score = float(validation["student_pinball"])
                relative_improvement = (
                    (best_score - score) / best_score if math.isfinite(best_score) else math.inf
                )
                improved = score < best_score
                materially_improved = relative_improvement >= float(
                    training["plateau_min_relative_improvement"]
                )
                if improved:
                    best_score = score
                    torch.save(
                        model.state_dict(),
                        args.checkpoint_dir / f"student-{args.variant}-best.pt",
                    )
                stale_evaluations = 0 if materially_improved else stale_evaluations + 1
                learning_curve.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "windows_processed": trained_windows,
                        "learning_rate": lr,
                        "validation_seconds": validation_seconds,
                        "relative_improvement_from_best": relative_improvement,
                        "validation": validation,
                    }
                )
                print(
                    f"validation variant={args.variant} step={step} pinball={score:.6f} "
                    f"teacher={validation['teacher_pinball']:.6f} stale={stale_evaluations}",
                    flush=True,
                )
                stopped_for_plateau = (
                    step >= int(training["plateau_min_steps"])
                    and stale_evaluations >= int(training["plateau_patience_evaluations"])
                )

            if step % int(training["checkpoint_every_steps"]) == 0 or stopped_for_plateau:
                _save_checkpoint(
                    args.checkpoint_dir / f"student-{args.variant}-resume.pt",
                    model,
                    optimizer,
                    step=step,
                    epoch=next_epoch,
                    batch_offset=next_offset,
                    best_score=best_score,
                    stale_evaluations=stale_evaluations,
                    learning_curve=learning_curve,
                )
                torch.save(
                    model.state_dict(),
                    args.checkpoint_dir / f"student-{args.variant}-step{step}.pt",
                )
            if step >= max_steps or stopped_for_plateau:
                break
        epoch += 1
        batch_offset = 0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_checkpoint = args.checkpoint_dir / f"student-{args.variant}-final.pt"
    torch.save(model.state_dict(), final_checkpoint)
    final_validation = learning_curve[-1]["validation"]
    metrics = {
        "validation/student_pinball": float(final_validation["student_pinball"]),
        "validation/teacher_pinball": float(final_validation["teacher_pinball"]),
        "training/windows_per_second": trained_windows / elapsed,
        **{
            f"training/{key}": value / trained_windows for key, value in train_sums.items()
        },
    }
    record.extra.update(
        {
            "variant": args.variant,
            "parameter_count": model.parameter_count,
            "initialization_sha256": initialization_sha256,
            "rank0_training_sequence_sha256": sequence_digest.hexdigest(),
            "corpus_load_seconds": corpus_load_seconds,
            "datasets": [
                {
                    "dataset": corpus.name,
                    "domain": corpus.domain,
                    "view_class": corpus.view_class,
                    "context": corpus.context_length,
                    "horizon": corpus.horizon,
                    "batch_size": corpus.batch_size,
                    "training_windows": len(corpus.training_indices),
                    "validation_windows": len(corpus.validation_indices),
                    "split": corpus.split_report,
                }
                for corpus in corpora
            ],
            "training": {
                "steps": step,
                "epochs_completed": epoch,
                "windows_processed": trained_windows,
                "elapsed_seconds": elapsed,
                "windows_per_second": trained_windows / elapsed,
                "precision": "bfloat16 autocast",
                "optimizer": "fused AdamW",
                "schedule": "cosine",
                "stopped_for_plateau": stopped_for_plateau,
                "maximum_steps": max_steps,
                "loss_weights": training["loss_weights"][args.variant],
            },
            "learning_curve": learning_curve,
            "final_checkpoint": str(final_checkpoint.resolve()),
            "best_checkpoint": str(
                (args.checkpoint_dir / f"student-{args.variant}-best.pt").resolve()
            ),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
    )
    record.succeed(metrics)
    record.write(args.output)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
