#!/usr/bin/env python3
"""Train one matched production student variant to validation convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import split_cache_indices
from timesfm_lab.distill.losses import DistillationLoss, LossWeights, pinball_loss
from timesfm_lab.models import build_student, masked_mean_and_scale
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
        corpus_batches = [
            order[start : start + corpus.batch_size]
            for start in range(0, len(order), corpus.batch_size)
        ]
        if len(corpus_batches) > 1 and len(corpus_batches[-1]) == 1:
            corpus_batches[-2] = np.concatenate((corpus_batches[-2], corpus_batches.pop()))
        batches.extend((corpus_index, indices) for indices in corpus_batches)
    rng.shuffle(batches)
    return batches


def _pack_logical_batches(
    physical_batches: list[tuple[int, npt.NDArray[Any]]],
    logical_batch_size_windows: int,
) -> list[list[tuple[int, npt.NDArray[Any]]]]:
    """Pack the existing deterministic microbatch stream into optimizer batches.

    A logical batch may contain microbatches from different corpora. This keeps
    every corpus's configured physical memory limit, consumes the exact same
    ordered examples as ``_epoch_batches``, and makes every optimizer batch but
    the final epoch tail contain the requested number of windows.
    """

    if logical_batch_size_windows <= 0:
        raise ValueError("logical_batch_size_windows must be positive")
    logical_batches: list[list[tuple[int, npt.NDArray[Any]]]] = []
    current: list[tuple[int, npt.NDArray[Any]]] = []
    remaining = logical_batch_size_windows
    for corpus_index, indices in physical_batches:
        start = 0
        while start < len(indices):
            take = min(remaining, len(indices) - start)
            current.append((corpus_index, indices[start : start + take]))
            start += take
            remaining -= take
            if remaining == 0:
                logical_batches.append(current)
                current = []
                remaining = logical_batch_size_windows
    if current:
        logical_batches.append(current)
    return logical_batches


def _observed_target_count(corpus: _Corpus, indices: npt.NDArray[Any]) -> int:
    """Count finite target positions without materializing context on the GPU."""

    count = 0
    for index in indices:
        row = int(corpus.row_index[index])
        end = int(corpus.context_end[index])
        target = corpus.source[row][:, end : end + corpus.horizon]
        count += int(np.count_nonzero(np.isfinite(target)))
    return count


def _to_device(
    values: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values)).pin_memory()
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _timesfm_interpolate_context(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Apply the pinned forecaster's trim/interpolate semantics at fixed width."""

    target = np.asarray(values, dtype=np.float32)
    all_missing = np.isnan(target).all(axis=0)
    first_valid = target.shape[-1] if all_missing.all() else int(np.argmax(~all_missing))
    if first_valid == target.shape[-1]:
        return np.zeros_like(target)
    output = np.full_like(target, np.nan)
    trimmed = target[:, first_valid:].copy()
    for row in trimmed:
        missing = np.isnan(row)
        if not missing.any():
            continue
        valid_indices = np.flatnonzero(~missing)
        if valid_indices.size:
            row[missing] = np.interp(
                np.flatnonzero(missing), valid_indices, row[valid_indices]
            )
        else:
            row[missing] = 0.0
    output[:, first_valid:] = trimmed
    return output


def _materialize(
    corpus: _Corpus,
    indices: npt.NDArray[Any],
    device: torch.device,
    input_preprocessing: str = "masked_raw",
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    contexts = []
    targets = []
    for index in indices:
        row = int(corpus.row_index[index])
        end = int(corpus.context_end[index])
        values = corpus.source[row]
        context_values = values[:, end - corpus.context_length : end]
        if input_preprocessing == "timesfm3_linear_interpolation":
            context_values = _timesfm_interpolate_context(context_values)
        elif input_preprocessing != "masked_raw":
            raise ValueError(f"unsupported input_preprocessing={input_preprocessing!r}")
        contexts.append(context_values)
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


def _deployment_forecast(
    model: nn.Module,
    context: Tensor,
    horizon: int,
    inference: dict[str, Any],
) -> Tensor:
    observed = torch.isfinite(context)
    positive = model(context, horizon, observed_mask=observed)
    if bool(inference.get("sort_quantiles", True)):
        positive = torch.sort(positive, dim=-1).values
    if bool(inference.get("use_symmetric_averaging", False)):
        negative = model(-context, horizon, observed_mask=observed)
        if bool(inference.get("sort_quantiles", True)):
            negative = torch.sort(negative, dim=-1).values
        output = (positive - negative.flip(-1)) / 2
    else:
        output = positive
    if bool(inference.get("make_positive", False)):
        nonnegative = observed.any(dim=-1) & torch.where(
            observed, context >= 0, torch.ones_like(observed)
        ).all(dim=-1)
        output = torch.where(
            nonnegative[..., None, None], output.clamp_min(0), output
        )
    return output


class _StudentViews(nn.Module):
    """Expose MV and optional UV forecasts in one DDP-safe forward graph."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(
        self, context: Tensor, horizon: int, include_univariate: bool
    ) -> tuple[Tensor, Tensor | None]:
        multivariate = self.student(context, horizon)
        univariate = (
            _student_univariate(self.student, context, horizon) if include_univariate else None
        )
        return multivariate, univariate


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
    weights = objective.weights
    include_univariate = (
        (weights.univariate_kd > 0 or weights.cvrd > 0)
        and corpus.view_class == "true_multivariate"
    )
    student_mv, student_uv = model(context, corpus.horizon, include_univariate)
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
        teacher_multivariate=(
            teacher_primary if weights.multivariate_kd > 0 or weights.cvrd > 0 else None
        ),
        student_univariate=student_uv,
        teacher_univariate=teacher_uv if student_uv is not None else None,
    )


def _response_sufficient_statistics(
    student_response: Tensor, teacher_response: Tensor, mask: Tensor
) -> dict[str, float]:
    expanded = mask.unsqueeze(-1).expand_as(student_response)
    student = student_response.masked_select(expanded).double()
    teacher = teacher_response.masked_select(expanded).double()
    return {
        "count": float(student.numel()),
        "student_sum": float(student.sum()),
        "teacher_sum": float(teacher.sum()),
        "student_square_sum": float(student.square().sum()),
        "teacher_square_sum": float(teacher.square().sum()),
        "cross_sum": float((student * teacher).sum()),
        "absolute_error_sum": float((student - teacher).abs().sum()),
        "absolute_teacher_sum": float(teacher.abs().sum()),
        "sign_agreement_sum": float(
            (
                torch.where(student > 1e-3, 1, torch.where(student < -1e-3, -1, 0))
                == torch.where(teacher > 1e-3, 1, torch.where(teacher < -1e-3, -1, 0))
            )
            .double()
            .sum()
        ),
        "magnitude_error_sum": float((student.abs() - teacher.abs()).abs().sum()),
    }


def _response_from_sufficient_statistics(parts: list[dict[str, float]]) -> dict[str, float]:
    totals = {key: sum(part[key] for part in parts) for key in parts[0]}
    count = max(totals["count"], 1.0)
    covariance = totals["cross_sum"] - (totals["student_sum"] * totals["teacher_sum"] / count)
    student_variance = totals["student_square_sum"] - totals["student_sum"] ** 2 / count
    teacher_variance = totals["teacher_square_sum"] - totals["teacher_sum"] ** 2 / count
    pearson_denominator = math.sqrt(max(student_variance * teacher_variance, 0.0))
    cosine_denominator = math.sqrt(totals["student_square_sum"] * totals["teacher_square_sum"])
    return {
        "pearson": covariance / max(pearson_denominator, 1e-12),
        "nmae": totals["absolute_error_sum"] / max(totals["absolute_teacher_sum"], 1e-12),
        "sign_agreement": totals["sign_agreement_sum"] / count,
        "cosine": totals["cross_sum"] / max(cosine_denominator, 1e-12),
        "magnitude_mae": totals["magnitude_error_sum"] / count,
    }


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("balanced validation requires finite positive component metrics")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _validation_score(validation: dict[str, Any], training: dict[str, Any]) -> float:
    selection_metric = str(training.get("validation_selection_metric", "student_pinball"))
    if selection_metric == "student_pinball":
        return float(validation["student_pinball"])
    if selection_metric == "balanced_forecast_ratio":
        return float(validation["balanced"]["forecast_ratio"])
    raise ValueError(f"unsupported validation_selection_metric={selection_metric!r}")


@torch.inference_mode()
def _validate(
    model: nn.Module,
    corpora: list[_Corpus],
    device: torch.device,
    input_preprocessing: str,
    inference: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    total_student = 0.0
    total_teacher = 0.0
    total_student_median_absolute_error = 0.0
    total_teacher_median_absolute_error = 0.0
    total_student_coverage = np.zeros(9, dtype=np.float64)
    total_weight = 0
    true_mv_student = 0.0
    true_mv_student_univariate = 0.0
    true_mv_teacher = 0.0
    true_mv_teacher_univariate = 0.0
    true_mv_weight = 0
    by_dataset = {}
    for corpus in corpora:
        student_sum = 0.0
        teacher_sum = 0.0
        dataset_student_median_absolute_error = 0.0
        dataset_teacher_median_absolute_error = 0.0
        weight_sum = 0
        response_parts: list[dict[str, float]] = []
        for start in range(0, len(corpus.validation_indices), corpus.batch_size):
            indices = corpus.validation_indices[start : start + corpus.batch_size]
            context, target, teacher_primary, teacher_uv = _materialize(
                corpus, indices, device, input_preprocessing
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_mv = _deployment_forecast(
                    model, context, corpus.horizon, inference
                )
                student_uv = (
                    _deployment_forecast(
                        model,
                        context.reshape(
                            context.shape[0] * context.shape[1], 1, context.shape[2]
                        ),
                        corpus.horizon,
                        inference,
                    ).reshape(
                        context.shape[0], context.shape[1], corpus.horizon, 9
                    )
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
            expanded_mask = mask.unsqueeze(-1)
            student_median_sum = float(
                torch.where(
                    mask,
                    (student_mv[..., 4] - normalized_target).abs(),
                    torch.zeros_like(normalized_target),
                ).sum()
            )
            teacher_median_sum = float(
                torch.where(
                    mask,
                    (teacher_primary[..., 4] - normalized_target).abs(),
                    torch.zeros_like(normalized_target),
                ).sum()
            )
            coverage = (
                ((normalized_target.unsqueeze(-1) <= student_mv) & expanded_mask)
                .sum(dim=(0, 1, 2))
                .double()
                .cpu()
                .numpy()
            )
            student_sum += student_value * weight
            teacher_sum += teacher_value * weight
            total_student_median_absolute_error += student_median_sum
            total_teacher_median_absolute_error += teacher_median_sum
            dataset_student_median_absolute_error += student_median_sum
            dataset_teacher_median_absolute_error += teacher_median_sum
            total_student_coverage += coverage
            weight_sum += weight
            if student_uv is not None and teacher_uv is not None:
                student_uv_value = float(pinball_loss(student_uv, normalized_target, mask))
                teacher_uv_value = float(pinball_loss(teacher_uv, normalized_target, mask))
                true_mv_student += student_value * weight
                true_mv_student_univariate += student_uv_value * weight
                true_mv_teacher += teacher_value * weight
                true_mv_teacher_univariate += teacher_uv_value * weight
                true_mv_weight += weight
                response_parts.append(
                    _response_sufficient_statistics(
                        student_mv - student_uv, teacher_primary - teacher_uv, mask
                    )
                )
        dataset_result: dict[str, Any] = {
            "student_pinball": student_sum / weight_sum,
            "teacher_pinball": teacher_sum / weight_sum,
            "student_normalized_median_mae": dataset_student_median_absolute_error
            / max(weight_sum, 1),
            "teacher_normalized_median_mae": dataset_teacher_median_absolute_error
            / max(weight_sum, 1),
            "observed_targets": weight_sum,
            "windows": len(corpus.validation_indices),
        }
        if response_parts:
            dataset_result["response"] = _response_from_sufficient_statistics(response_parts)
        by_dataset[corpus.name] = dataset_result
        total_student += student_sum
        total_teacher += teacher_sum
        total_weight += weight_sum
    model.train()
    result = {
        "student_pinball": total_student / total_weight,
        "teacher_pinball": total_teacher / total_weight,
        "student_normalized_median_mae": total_student_median_absolute_error / total_weight,
        "teacher_normalized_median_mae": total_teacher_median_absolute_error / total_weight,
        "student_empirical_quantile_coverage": {
            f"{level / 10:.1f}": float(total_student_coverage[level - 1] / total_weight)
            for level in range(1, 10)
        },
        "observed_targets": total_weight,
        "by_dataset": by_dataset,
    }
    if true_mv_weight:
        result["true_multivariate"] = {
            "student_multivariate_pinball": true_mv_student / true_mv_weight,
            "student_univariate_pinball": true_mv_student_univariate / true_mv_weight,
            "student_mv_minus_uv_fraction": (
                true_mv_student / true_mv_student_univariate - 1.0
            ),
            "teacher_multivariate_pinball": true_mv_teacher / true_mv_weight,
            "teacher_univariate_pinball": true_mv_teacher_univariate / true_mv_weight,
            "teacher_mv_minus_uv_fraction": (
                true_mv_teacher / true_mv_teacher_univariate - 1.0
            ),
            "observed_targets": true_mv_weight,
        }
    dataset_values = list(by_dataset.values())
    true_mv_names = {corpus.name for corpus in corpora if corpus.view_class == "true_multivariate"}
    true_mv_values = [by_dataset[name] for name in sorted(true_mv_names)]
    balanced = {
        "student_pinball": _geometric_mean(
            [float(value["student_pinball"]) for value in dataset_values]
        ),
        "teacher_pinball": _geometric_mean(
            [float(value["teacher_pinball"]) for value in dataset_values]
        ),
        "student_normalized_median_mae": _geometric_mean(
            [float(value["student_normalized_median_mae"]) for value in dataset_values]
        ),
        "teacher_normalized_median_mae": _geometric_mean(
            [float(value["teacher_normalized_median_mae"]) for value in dataset_values]
        ),
        "true_mv_student_pinball": _geometric_mean(
            [float(value["student_pinball"]) for value in true_mv_values]
        ),
        "true_mv_teacher_pinball": _geometric_mean(
            [float(value["teacher_pinball"]) for value in true_mv_values]
        ),
        "true_mv_student_normalized_median_mae": _geometric_mean(
            [float(value["student_normalized_median_mae"]) for value in true_mv_values]
        ),
        "true_mv_teacher_normalized_median_mae": _geometric_mean(
            [float(value["teacher_normalized_median_mae"]) for value in true_mv_values]
        ),
        "dataset_count": len(dataset_values),
        "true_mv_dataset_count": len(true_mv_values),
        "aggregation": "unweighted geometric mean across datasets",
    }
    # Frozen 30/30/15/15 quality weights, renormalized after excluding the
    # separately measured 10% latency term from checkpoint selection.
    balanced["forecast_ratio"] = (
        (balanced["student_normalized_median_mae"] / balanced["teacher_normalized_median_mae"])
        ** (1 / 3)
        * (balanced["student_pinball"] / balanced["teacher_pinball"]) ** (1 / 3)
        * (
            balanced["true_mv_student_normalized_median_mae"]
            / balanced["true_mv_teacher_normalized_median_mae"]
        )
        ** (1 / 6)
        * (balanced["true_mv_student_pinball"] / balanced["true_mv_teacher_pinball"])
        ** (1 / 6)
    )
    result["balanced"] = balanced
    return result


def _save_checkpoint(
    path: Path,
    model: nn.Module,
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
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="load model weights only and begin a new optimizer/schedule at step zero",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--training-seed",
        type=int,
        help="override model initialization and epoch-order seed",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="override the validation split seed independently of training",
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="run the full declared step budget while retaining all validation checkpoints",
    )
    args = parser.parse_args()
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    training = config["training"]
    inference = dict(config.get("inference", {}))
    input_preprocessing = str(training.get("input_preprocessing", "masked_raw"))
    if args.variant not in training["loss_weights"]:
        raise ValueError(
            f"variant {args.variant!r} is absent from training.loss_weights; "
            f"available={tuple(training['loss_weights'])}"
        )
    configured_logical_batch = training.get("logical_batch_size_windows")
    logical_batch_size_windows = (
        int(configured_logical_batch) if configured_logical_batch is not None else None
    )
    if logical_batch_size_windows is not None and logical_batch_size_windows <= 0:
        raise ValueError("training.logical_batch_size_windows must be positive")
    if logical_batch_size_windows is not None and args.distributed:
        raise ValueError(
            "fixed logical-batch accumulation currently supports single-GPU training only; "
            "run one independent variant per GPU instead of --distributed"
        )
    configured_seed = int(config["seed"])
    training_seed = args.training_seed if args.training_seed is not None else configured_seed
    split_seed = args.split_seed if args.split_seed is not None else configured_seed
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    if args.distributed:
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size != 2:
            raise ValueError("production DDP is defined for exactly two ranks")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    load_started = time.perf_counter()
    corpora = [
        _load_corpus(
            item,
            data_root=args.data_root,
            cache_root=args.cache_root,
            validation_fraction=float(training["validation_fraction"]),
            validation_mode=str(training["validation_split"]),
            seed=split_seed,
            batch_sizes=training["batch_size_by_context"],
        )
        for item in plan["datasets"]
    ]
    corpus_load_seconds = time.perf_counter() - load_started
    student = build_student(config["student"])
    random_initialization_sha256 = _state_sha256(student)
    initialization_checkpoint_sha256 = None
    if args.initialize_from is not None:
        initialization_state = torch.load(
            args.initialize_from, map_location="cpu", weights_only=True
        )
        if isinstance(initialization_state, dict) and "model" in initialization_state:
            initialization_state = initialization_state["model"]
        student.load_state_dict(initialization_state)
        initialization_checkpoint_sha256 = _sha256(args.initialize_from)
    initialization_sha256 = _state_sha256(student)
    student.to(device)
    training_model: Any = _StudentViews(student).to(device)
    if args.distributed:
        training_model = torch.nn.parallel.DistributedDataParallel(
            training_model, device_ids=[local_rank]
        )
    weights = LossWeights.from_mapping(training["loss_weights"][args.variant])
    objective = DistillationLoss(weights)
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
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
    trained_windows = 0
    observed_targets_processed = 0
    physical_microbatches_processed = 0
    physical_microbatch_windows_sum = 0
    physical_microbatch_windows_min = math.inf
    physical_microbatch_windows_max = 0
    optimizer_batch_windows_sum = 0
    optimizer_batch_windows_min = math.inf
    optimizer_batch_windows_max = 0
    previous_elapsed = 0.0
    sequence_chain = bytes(32)
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
    train_weight_sum = 0.0
    gradient_norm_sum = 0.0
    gradient_clip_count = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_training_seed = int(state.get("training_seed", configured_seed))
        checkpoint_split_seed = int(state.get("split_seed", configured_seed))
        if checkpoint_training_seed != training_seed or checkpoint_split_seed != split_seed:
            raise ValueError(
                "resume seed mismatch: "
                f"checkpoint training/split={checkpoint_training_seed}/{checkpoint_split_seed}, "
                f"requested={training_seed}/{split_seed}"
            )
        checkpoint_logical_batch = state.get("logical_batch_size_windows")
        if checkpoint_logical_batch != logical_batch_size_windows:
            raise ValueError(
                "resume logical-batch mismatch: "
                f"checkpoint={checkpoint_logical_batch}, "
                f"requested={logical_batch_size_windows}"
            )
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        epoch = int(state["epoch"])
        batch_offset = int(state["batch_offset"])
        best_score = float(state["best_score"])
        stale_evaluations = int(state["stale_evaluations"])
        learning_curve = list(state["learning_curve"])
        trained_windows = int(state.get("trained_windows", 0))
        observed_targets_processed = int(state.get("observed_targets_processed", 0))
        physical_microbatches_processed = int(
            state.get("physical_microbatches_processed", step)
        )
        physical_microbatch_windows_sum = int(
            state.get("physical_microbatch_windows_sum", trained_windows)
        )
        physical_microbatch_windows_min = float(
            state.get("physical_microbatch_windows_min", math.inf)
        )
        physical_microbatch_windows_max = int(
            state.get("physical_microbatch_windows_max", 0)
        )
        optimizer_batch_windows_sum = int(
            state.get("optimizer_batch_windows_sum", trained_windows)
        )
        optimizer_batch_windows_min = float(
            state.get("optimizer_batch_windows_min", math.inf)
        )
        optimizer_batch_windows_max = int(state.get("optimizer_batch_windows_max", 0))
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        sequence_chain = bytes.fromhex(state.get("training_sequence_sha256", bytes(32).hex()))
        train_sums.update(state.get("train_sums", {}))
        train_weight_sum = float(state.get("train_weight_sum", trained_windows))
        gradient_norm_sum = float(state.get("gradient_norm_sum", 0.0))
        gradient_clip_count = int(state.get("gradient_clip_count", 0))

    record = (
        RunRecord.start(
            run_id=f"{config['run_id']}-{args.variant}",
            config_path=f"{args.config};{args.plan}",
            seed=training_seed,
            model_revision=str(config["model_revision"]),
            dataset_revision=str(config["dataset_revision"]),
            hardware_snapshot=str(config["hardware_snapshot"]),
            repository=ROOT,
        )
        if is_main
        else None
    )
    stopped_for_plateau = False
    started = time.perf_counter()
    training_model.train()
    validate_at_start = bool(training.get("validate_at_start", False))
    if validate_at_start and step == 0 and not learning_curve:
        validation_started = time.perf_counter()
        initial_validation = (
            _validate(student, corpora, device, input_preprocessing, inference)
            if is_main
            else None
        )
        if args.distributed:
            shared_validation = [initial_validation]
            torch.distributed.broadcast_object_list(shared_validation, src=0)
            initial_validation = shared_validation[0]
        assert initial_validation is not None
        initial_validation_seconds = time.perf_counter() - validation_started
        best_score = _validation_score(initial_validation, training)
        learning_curve.append(
            {
                "step": 0,
                "epoch": epoch,
                "windows_processed": trained_windows,
                "observed_targets_processed": observed_targets_processed,
                "optimizer_batches_processed": step,
                "physical_microbatches_processed": physical_microbatches_processed,
                "learning_rate": float(training["learning_rate"]),
                "validation_seconds": initial_validation_seconds,
                "relative_improvement_from_best": None,
                "validation": initial_validation,
            }
        )
        if is_main:
            torch.save(
                student.state_dict(),
                args.checkpoint_dir / f"student-{args.variant}-best.pt",
            )
            print(
                f"validation variant={args.variant} step=0 score={best_score:.6f} "
                f"pinball={initial_validation['student_pinball']:.6f} "
                f"teacher={initial_validation['teacher_pinball']:.6f} stale=0",
                flush=True,
            )
    while step < max_steps and not stopped_for_plateau:
        physical_batches = _epoch_batches(corpora, training_seed, epoch)
        batches = (
            _pack_logical_batches(physical_batches, logical_batch_size_windows)
            if logical_batch_size_windows is not None
            else [[physical_batch] for physical_batch in physical_batches]
        )
        for offset in range(batch_offset, len(batches)):
            optimizer_batch = batches[offset]
            optimizer_batch_windows = sum(len(indices) for _, indices in optimizer_batch)
            if optimizer_batch_windows <= 0:
                raise ValueError("empty optimizer batch")
            for corpus_index, global_indices in optimizer_batch:
                corpus = corpora[corpus_index]
                sequence_chain = hashlib.sha256(
                    sequence_chain
                    + corpus.name.encode()
                    + np.asarray(global_indices, dtype="<i8").tobytes()
                ).digest()

            # The opt-in path scans only the small target slices first so each
            # reduced microbatch loss can be weighted by its exact number of
            # finite targets. Historical one-microbatch behavior remains the
            # default, including its established two-rank weighting.
            logical_observed_targets = (
                sum(
                    _observed_target_count(corpora[corpus_index], global_indices)
                    for corpus_index, global_indices in optimizer_batch
                )
                if logical_batch_size_windows is not None
                else None
            )
            if logical_observed_targets is not None and logical_observed_targets <= 0:
                raise ValueError(f"optimizer batch at epoch={epoch} offset={offset} has no targets")

            optimizer.zero_grad(set_to_none=True)
            statistics = torch.zeros(len(train_sums) + 1, dtype=torch.float64, device=device)
            datasets_in_batch: list[str] = []
            realized_observed_targets = 0
            for corpus_index, global_indices in optimizer_batch:
                corpus = corpora[corpus_index]
                datasets_in_batch.append(corpus.name)
                indices = global_indices[rank::world_size]
                context, target, teacher_primary, teacher_uv = _materialize(
                    corpus, indices, device, input_preprocessing
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = _loss(
                        training_model,
                        objective,
                        args.variant,
                        corpus,
                        context,
                        target,
                        teacher_primary,
                        teacher_uv,
                        student.config.normalization_epsilon,
                    )
                finite = torch.tensor(
                    float(all(torch.isfinite(value) for value in values.values())), device=device
                )
                if args.distributed:
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError(
                        f"non-finite loss at step {step + 1} on {corpus.name}"
                    )
                local_observed_targets = torch.isfinite(target).sum().to(torch.float64)
                local_weight = local_observed_targets * 9
                global_weight = local_weight.clone()
                if args.distributed:
                    torch.distributed.all_reduce(global_weight, op=torch.distributed.ReduceOp.SUM)
                if logical_observed_targets is None:
                    loss_scale = world_size * local_weight / global_weight.clamp_min(1)
                else:
                    loss_scale = local_observed_targets / logical_observed_targets
                (values["loss"] * loss_scale).backward()
                microbatch_statistics = torch.stack(
                    [values[key].detach().to(torch.float64) * local_weight for key in train_sums]
                    + [local_weight]
                )
                if args.distributed:
                    torch.distributed.all_reduce(
                        microbatch_statistics, op=torch.distributed.ReduceOp.SUM
                    )
                statistics += microbatch_statistics
                realized_observed_targets += int(local_observed_targets.item())
                observed_targets_processed += int(
                    global_weight.item() / 9
                    if logical_observed_targets is None
                    else local_observed_targets.item()
                )
                physical_microbatches_processed += 1
                physical_microbatch_windows_sum += len(global_indices)
                physical_microbatch_windows_min = min(
                    physical_microbatch_windows_min, len(global_indices)
                )
                physical_microbatch_windows_max = max(
                    physical_microbatch_windows_max, len(global_indices)
                )
            if (
                logical_observed_targets is not None
                and realized_observed_targets != logical_observed_targets
            ):
                raise RuntimeError(
                    "CPU/GPU observed-target count mismatch: "
                    f"expected={logical_observed_targets}, "
                    f"realized={realized_observed_targets}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(), float(training["gradient_clip"])
            )
            gradient_norm_value = float(gradient_norm.detach())
            gradient_norm_sum += gradient_norm_value
            gradient_clip_count += int(gradient_norm_value > float(training["gradient_clip"]))
            optimizer.step()
            step += 1
            trained_windows += optimizer_batch_windows
            optimizer_batch_windows_sum += optimizer_batch_windows
            optimizer_batch_windows_min = min(
                optimizer_batch_windows_min, optimizer_batch_windows
            )
            optimizer_batch_windows_max = max(
                optimizer_batch_windows_max, optimizer_batch_windows
            )
            progress = step / max_steps
            lr = float(training["min_learning_rate"]) + 0.5 * (
                float(training["learning_rate"]) - float(training["min_learning_rate"])
            ) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_values = {
                key: float(statistics[index] / statistics[-1].clamp_min(1))
                for index, key in enumerate(train_sums)
            }
            metric_weight = (
                float(logical_observed_targets)
                if logical_observed_targets is not None
                else float(optimizer_batch_windows)
            )
            for key, value in batch_values.items():
                train_sums[key] += value * metric_weight
            train_weight_sum += metric_weight
            if is_main and (step == 1 or step % 100 == 0):
                throughput_elapsed = previous_elapsed + time.perf_counter() - started
                dataset_label = "+".join(dict.fromkeys(datasets_in_batch))
                print(
                    f"variant={args.variant} step={step}/{max_steps} epoch={epoch} "
                    f"datasets={dataset_label} logical_batch={optimizer_batch_windows} "
                    f"microbatches={len(optimizer_batch)} "
                    f"loss={batch_values['loss']:.6f} "
                    f"windows_per_second={trained_windows / throughput_elapsed:.1f}",
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
                validation = (
                    _validate(student, corpora, device, input_preprocessing, inference)
                    if is_main
                    else None
                )
                if args.distributed:
                    shared_validation = [validation]
                    torch.distributed.broadcast_object_list(shared_validation, src=0)
                    validation = shared_validation[0]
                assert validation is not None
                validation_seconds = time.perf_counter() - validation_started
                score = _validation_score(validation, training)
                relative_improvement = (
                    (best_score - score) / best_score if math.isfinite(best_score) else math.inf
                )
                improved = score < best_score
                materially_improved = relative_improvement >= float(
                    training["plateau_min_relative_improvement"]
                )
                if improved and is_main:
                    best_score = score
                    torch.save(
                        student.state_dict(),
                        args.checkpoint_dir / f"student-{args.variant}-best.pt",
                    )
                elif improved:
                    best_score = score
                stale_evaluations = 0 if materially_improved else stale_evaluations + 1
                learning_curve.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "windows_processed": trained_windows,
                        "observed_targets_processed": observed_targets_processed,
                        "optimizer_batches_processed": step,
                        "physical_microbatches_processed": physical_microbatches_processed,
                        "learning_rate": lr,
                        "validation_seconds": validation_seconds,
                        "relative_improvement_from_best": relative_improvement,
                        "validation": validation,
                    }
                )
                if is_main:
                    print(
                        f"validation variant={args.variant} step={step} score={score:.6f} "
                        f"pinball={validation['student_pinball']:.6f} "
                        f"teacher={validation['teacher_pinball']:.6f} "
                        f"stale={stale_evaluations}",
                        flush=True,
                    )
                stopped_for_plateau = (
                    not args.disable_early_stopping
                    and step >= int(training["plateau_min_steps"])
                    and stale_evaluations >= int(training["plateau_patience_evaluations"])
                )

            if is_main and (
                step % int(training["checkpoint_every_steps"]) == 0 or stopped_for_plateau
            ):
                _save_checkpoint(
                    args.checkpoint_dir / f"student-{args.variant}-resume.pt",
                    student,
                    optimizer,
                    step=step,
                    epoch=next_epoch,
                    batch_offset=next_offset,
                    best_score=best_score,
                    stale_evaluations=stale_evaluations,
                    learning_curve=learning_curve,
                    trained_windows=trained_windows,
                    observed_targets_processed=observed_targets_processed,
                    physical_microbatches_processed=physical_microbatches_processed,
                    physical_microbatch_windows_sum=physical_microbatch_windows_sum,
                    physical_microbatch_windows_min=physical_microbatch_windows_min,
                    physical_microbatch_windows_max=physical_microbatch_windows_max,
                    optimizer_batch_windows_sum=optimizer_batch_windows_sum,
                    optimizer_batch_windows_min=optimizer_batch_windows_min,
                    optimizer_batch_windows_max=optimizer_batch_windows_max,
                    logical_batch_size_windows=logical_batch_size_windows,
                    elapsed_seconds=previous_elapsed + time.perf_counter() - started,
                    training_sequence_sha256=sequence_chain.hex(),
                    training_seed=training_seed,
                    split_seed=split_seed,
                    train_sums=train_sums,
                    train_weight_sum=train_weight_sum,
                    gradient_norm_sum=gradient_norm_sum,
                    gradient_clip_count=gradient_clip_count,
                )
                torch.save(
                    student.state_dict(),
                    args.checkpoint_dir / f"student-{args.variant}-step{step}.pt",
                )
            if step >= max_steps or stopped_for_plateau:
                break
        epoch += 1
        batch_offset = 0
    torch.cuda.synchronize()
    elapsed = previous_elapsed + time.perf_counter() - started
    if args.distributed:
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
        elapsed = float(elapsed_tensor)
    final_checkpoint = args.checkpoint_dir / f"student-{args.variant}-final.pt"
    if is_main:
        torch.save(student.state_dict(), final_checkpoint)
    final_validation = learning_curve[-1]["validation"]
    metrics = {
        "validation/student_pinball": float(final_validation["student_pinball"]),
        "validation/teacher_pinball": float(final_validation["teacher_pinball"]),
        "training/windows_per_second": trained_windows / elapsed,
        **{
            f"training/{key}": value / max(train_weight_sum, 1.0)
            for key, value in train_sums.items()
        },
    }
    if not is_main:
        torch.distributed.destroy_process_group()
        return 0
    assert record is not None
    record.extra.update(
        {
            "variant": args.variant,
            "training_seed": training_seed,
            "validation_split_seed": split_seed,
            "resume_checkpoint": str(args.resume.resolve()) if args.resume is not None else None,
            "initialization_checkpoint": (
                str(args.initialize_from.resolve())
                if args.initialize_from is not None
                else None
            ),
            "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
            "parameter_count": student.parameter_count,
            "random_initialization_sha256": random_initialization_sha256,
            "initialization_sha256": initialization_sha256,
            "rank0_training_sequence_sha256": sequence_chain.hex(),
            "corpus_load_seconds": corpus_load_seconds,
            "datasets": [
                {
                    "dataset": corpus.name,
                    "domain": corpus.domain,
                    "view_class": corpus.view_class,
                    "context": corpus.context_length,
                    "horizon": corpus.horizon,
                    "batch_size": corpus.batch_size,
                    "physical_batch_size_windows": corpus.batch_size,
                    "training_windows": len(corpus.training_indices),
                    "validation_windows": len(corpus.validation_indices),
                    "split": corpus.split_report,
                }
                for corpus in corpora
            ],
            "training": {
                "steps": step,
                "optimizer_batches_processed": step,
                "logical_batches_processed": (
                    step if logical_batch_size_windows is not None else None
                ),
                "epochs_completed": epoch,
                "windows_processed": trained_windows,
                "examples_processed": trained_windows,
                "observed_targets_processed": observed_targets_processed,
                "logical_batch_size_windows": logical_batch_size_windows,
                "logical_batch_epoch_tail_policy": (
                    "single_smaller_final_batch_preserving_all_examples"
                    if logical_batch_size_windows is not None
                    else None
                ),
                "optimizer_batch_windows": {
                    "minimum": (
                        int(optimizer_batch_windows_min)
                        if math.isfinite(optimizer_batch_windows_min)
                        else None
                    ),
                    "maximum": optimizer_batch_windows_max,
                    "mean": optimizer_batch_windows_sum / max(step, 1),
                },
                "physical_microbatches_processed": physical_microbatches_processed,
                "physical_microbatch_windows": {
                    "minimum": (
                        int(physical_microbatch_windows_min)
                        if math.isfinite(physical_microbatch_windows_min)
                        else None
                    ),
                    "maximum": physical_microbatch_windows_max,
                    "mean": physical_microbatch_windows_sum
                    / max(physical_microbatches_processed, 1),
                },
                "train_metric_weight": train_weight_sum,
                "train_metric_weight_unit": (
                    "observed_target_positions"
                    if logical_batch_size_windows is not None
                    else "windows_historical"
                ),
                "elapsed_seconds": elapsed,
                "windows_per_second": trained_windows / elapsed,
                "precision": "bfloat16 autocast",
                "input_preprocessing": input_preprocessing,
                "validation_inference_policy": inference,
                "optimizer": "fused AdamW",
                "schedule": "cosine",
                "stopped_for_plateau": stopped_for_plateau,
                "early_stopping_enabled": not args.disable_early_stopping,
                "validate_at_start": validate_at_start,
                "maximum_steps": max_steps,
                "loss_weights": training["loss_weights"][args.variant],
                "validation_selection_metric": str(
                    training.get("validation_selection_metric", "student_pinball")
                ),
                "layout": "two_gpu_ddp" if args.distributed else "single_gpu",
                "world_size": world_size,
                "training_sequence_hash_scope": (
                    "ordered corpus and window indices at physical microbatch boundaries"
                ),
                "mean_preclip_gradient_norm": gradient_norm_sum / max(step, 1),
                "gradient_clip_count": gradient_clip_count,
                "gradient_clip_fraction": gradient_clip_count / max(step, 1),
            },
            "learning_curve": learning_curve,
            "final_checkpoint": str(final_checkpoint.resolve()),
            "best_checkpoint": str(
                (args.checkpoint_dir / f"student-{args.variant}-best.pt").resolve()
            ),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(local_rank),
            },
        }
    )
    record.succeed(metrics)
    record.write(args.output)
    print(args.output, flush=True)
    if args.distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
