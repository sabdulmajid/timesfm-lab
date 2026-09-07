#!/usr/bin/env python3
"""Train one matched production student variant to validation convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
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


def _partition_index_sha256(dataset: str, indices: npt.NDArray[Any]) -> str:
    digest = hashlib.sha256(dataset.encode("utf-8") + b"\0")
    digest.update(np.sort(np.asarray(indices)).astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _partition_identity_sha256(
    dataset: str,
    rows: npt.NDArray[Any],
    ends: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    context: int,
    horizon: int,
) -> str:
    """Hash exact cache identities independently of their shard/index order."""
    selected = np.asarray(indices, dtype=np.int64)
    selected_rows = np.asarray(rows)[selected].astype("<i8", copy=False)
    selected_ends = np.asarray(ends)[selected].astype("<i8", copy=False)
    order = np.lexsort((selected_ends, selected_rows))
    encoded = dataset.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<qqQ", context, horizon, len(selected)))
    pairs = np.column_stack((selected_rows[order], selected_ends[order])).astype("<i8", copy=False)
    digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


def _verify_partition(
    dataset: str,
    rows: npt.NDArray[Any],
    ends: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    context: int,
    horizon: int,
    expected: dict[str, Any],
    name: str,
) -> None:
    actual = {
        "count": len(indices),
        "cache_index_sha256": _partition_index_sha256(dataset, indices),
        "identity_sha256": _partition_identity_sha256(
            dataset, rows, ends, indices, context, horizon
        ),
    }
    for key, value in actual.items():
        if value != expected.get(key):
            raise ValueError(
                f"{dataset}: frozen {name} {key} mismatch: "
                f"expected={expected.get(key)!r}, actual={value!r}"
            )


def _frozen_selection_partitions(
    dataset: str,
    rows: npt.NDArray[Any],
    ends: npt.NDArray[Any],
    context: int,
    horizon: int,
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Reconstruct and verify the target-blind nested recovery split."""
    outer_config = manifest["outer_split"]
    outer, outer_report = split_cache_indices(
        rows,
        ends,
        context_length=context,
        horizon=horizon,
        validation_fraction=float(outer_config["validation_fraction"]),
        seed=int(outer_config["seed"]),
        mode=str(outer_config["mode"]),
    )
    outer_training = np.asarray(outer["training"], dtype=np.int64)
    outer_validation = np.asarray(outer["validation"], dtype=np.int64)
    all_indices = np.arange(len(rows), dtype=np.int64)
    outer_embargo = np.setdiff1d(all_indices, np.union1d(outer_training, outer_validation))

    nested_config = manifest["nested_split"]
    validation_rows = np.asarray(rows)[outer_validation]
    validation_ends = np.asarray(ends)[outer_validation]
    inner_mode = "held_out_series" if len(np.unique(validation_rows)) > 1 else "blocked_time"
    try:
        inner, inner_report = split_cache_indices(
            validation_rows,
            validation_ends,
            context_length=context,
            horizon=horizon,
            validation_fraction=float(nested_config["confirmation_fraction"]),
            seed=int(nested_config["seed"]),
            mode=inner_mode,
        )
    except ValueError as error:
        if inner_mode != "blocked_time" or "left no training windows" not in str(error):
            raise
        inner = {
            "training": np.arange(len(outer_validation), dtype=np.int64),
            "validation": np.asarray([], dtype=np.int64),
        }
        inner_report = {
            "mode": "development_only_no_valid_independent_confirmation",
            "leakage_control": "no confirmation examples emitted for this dataset",
            "excluded_embargo_windows": 0,
            "reason": str(error),
        }
    development = outer_validation[np.asarray(inner["training"], dtype=np.int64)]
    confirmation = outer_validation[np.asarray(inner["validation"], dtype=np.int64)]
    inner_embargo = np.setdiff1d(outer_validation, np.union1d(development, confirmation))
    partitions = {
        "outer_training": outer_training,
        "outer_validation": outer_validation,
        "outer_embargo": outer_embargo,
        "development": development,
        "confirmation": confirmation,
        "inner_embargo": inner_embargo,
    }
    if str(entry["dataset"]) != dataset:
        raise ValueError(f"selection entry key/name mismatch for {dataset}")
    if int(entry["context"]) != context or int(entry["horizon"]) != horizon:
        raise ValueError(f"{dataset}: frozen selection shape disagrees with corpus plan")
    if int(entry["cache_windows"]) != len(rows):
        raise ValueError(f"{dataset}: frozen selection cache count mismatch")
    for name, indices in partitions.items():
        _verify_partition(
            dataset,
            rows,
            ends,
            indices,
            context,
            horizon,
            entry["partitions"][name],
            name,
        )
    return partitions, {"outer": outer_report, "inner": inner_report}


def _load_corpus(
    item: dict[str, Any],
    *,
    data_root: Path,
    cache_root: Path,
    validation_fraction: float,
    validation_mode: str,
    seed: int,
    batch_sizes: dict[str, Any],
    selection_manifest: dict[str, Any] | None = None,
    selection_entry: dict[str, Any] | None = None,
    validation_partition: str | None = None,
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
    if selection_manifest is None:
        if selection_entry is not None or validation_partition is not None:
            raise ValueError("partial frozen selection configuration")
        splits, split_report = split_cache_indices(
            arrays["row_index"],
            arrays["context_end"],
            context_length=context_length,
            horizon=horizon,
            validation_fraction=validation_fraction,
            seed=seed,
            mode=validation_mode,
        )
        training_indices = splits["training"]
        validation_indices = splits["validation"]
    else:
        if selection_entry is None or validation_partition not in {
            "development",
            "confirmation",
        }:
            raise ValueError("frozen selection requires an entry and named partition")
        partitions, split_report = _frozen_selection_partitions(
            name,
            arrays["row_index"],
            arrays["context_end"],
            context_length,
            horizon,
            selection_manifest,
            selection_entry,
        )
        training_indices = partitions["outer_training"]
        validation_indices = partitions[validation_partition]
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
        training_indices=training_indices,
        validation_indices=validation_indices,
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
            row[missing] = np.interp(np.flatnonzero(missing), valid_indices, row[valid_indices])
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
        output = torch.where(nonnegative[..., None, None], output.clamp_min(0), output)
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
    loss_reduction: str = "observed_target_element",
) -> dict[str, Tensor]:
    weights = objective.weights
    include_univariate = (
        weights.univariate_kd > 0 or weights.cvrd > 0
    ) and corpus.view_class == "true_multivariate"
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
    if loss_reduction == "per_window_domain_balanced":
        return objective(
            student_mv,
            normalized_target,
            mask=mask,
            teacher_multivariate=(
                teacher_primary if weights.multivariate_kd > 0 or weights.cvrd > 0 else None
            ),
            student_univariate=student_uv,
            teacher_univariate=teacher_uv if student_uv is not None else None,
            reduction="per_window",
        )
    if loss_reduction != "observed_target_element":
        raise ValueError(f"unsupported training loss reduction: {loss_reduction!r}")
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
    if selection_metric == "balanced_forecast_error":
        return float(validation["balanced"]["forecast_error"])
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
                student_mv = _deployment_forecast(model, context, corpus.horizon, inference)
                student_uv = (
                    _deployment_forecast(
                        model,
                        context.reshape(context.shape[0] * context.shape[1], 1, context.shape[2]),
                        corpus.horizon,
                        inference,
                    ).reshape(context.shape[0], context.shape[1], corpus.horizon, 9)
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
            "student_mv_minus_uv_fraction": (true_mv_student / true_mv_student_univariate - 1.0),
            "teacher_multivariate_pinball": true_mv_teacher / true_mv_weight,
            "teacher_univariate_pinball": true_mv_teacher_univariate / true_mv_weight,
            "teacher_mv_minus_uv_fraction": (true_mv_teacher / true_mv_teacher_univariate - 1.0),
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
        * (balanced["true_mv_student_pinball"] / balanced["true_mv_teacher_pinball"]) ** (1 / 6)
    )
    # This target-only score is the recovery selection authority. Cached teacher
    # outputs used a deterministic single-pass policy, whereas final deployment
    # applies symmetric averaging and conditional nonnegative clipping. Avoid
    # silently using those non-parity teacher metrics as candidate weights.
    balanced["forecast_error"] = (
        balanced["student_normalized_median_mae"] ** (1 / 3)
        * balanced["student_pinball"] ** (1 / 3)
        * balanced["true_mv_student_normalized_median_mae"] ** (1 / 6)
        * balanced["true_mv_student_pinball"] ** (1 / 6)
    )
    balanced["selection_authority"] = (
        "target-only development error; cached teacher metrics are single-pass diagnostics"
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
    parser.add_argument(
        "--selection-split-manifest",
        type=Path,
        help="frozen recovery development/confirmation identities",
    )
    parser.add_argument(
        "--validation-partition",
        choices=("development", "confirmation"),
        help="named frozen partition used for checkpoint evaluation",
    )
    parser.add_argument(
        "--frozen-finalist-selection",
        type=Path,
        help=(
            "committed screen-finalist freeze required for an allowlisted full-training "
            "derivative"
        ),
    )
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
    import subprocess as confirmation_process_control

    import yaml as finalist_yaml

    git_common_dir = Path(
        confirmation_process_control.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    confirmation_burn = (
        git_common_dir / "timesfm-lab-performance-recovery-confirmation-access-burn.json"
    )
    if confirmation_burn.exists():
        raise ValueError("training is permanently closed after confirmation access is burned")
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.validation_partition == "confirmation":
        raise ValueError("confirmation is evaluation-only; use evaluate_recovery_confirmation.py")
    if (args.selection_split_manifest is None) != (args.validation_partition is None):
        raise ValueError(
            "--selection-split-manifest and --validation-partition must be provided together"
        )
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())

    # Recovery identity is an immutable config-path/hash/variant tuple.  ``run_id`` is
    # intentionally absent: changing a display label cannot opt a recipe into or out of
    # the frozen recovery protocol.
    derivative_allowlist_path = (
        ROOT / "configs/performance_recovery/finalist_derivatives.yaml"
    ).resolve()
    derivative_allowlist_expected_sha256 = (
        "a10a72f7e2334fc0f36308e1e396259c4506d794206fe54eb41a374ffc6e3b78"
    )
    if _sha256(derivative_allowlist_path) != derivative_allowlist_expected_sha256:
        raise ValueError("frozen finalist-derivative allowlist changed")
    derivative_allowlist = finalist_yaml.safe_load(derivative_allowlist_path.read_text())
    if not isinstance(derivative_allowlist, dict):
        raise ValueError("finalist-derivative allowlist must be a mapping")
    if (
        derivative_allowlist.get("schema_version") != 1
        or derivative_allowlist.get("status") != "frozen_predeclared_before_full_training"
        or derivative_allowlist.get("protocol_id") != "timesfm3-performance-recovery-v1.2"
    ):
        raise ValueError("unsupported finalist-derivative allowlist")
    try:
        config_relative = str(args.config.resolve().relative_to(ROOT))
    except ValueError as error:
        raise ValueError("training config must be inside the repository") from error
    config_sha256 = _sha256(args.config)
    derivatives = derivative_allowlist["derivatives"]
    known_variants = {str(row["variant"]) for row in derivatives.values()}
    known_config_paths = {str(row["config"]["path"]) for row in derivatives.values()}
    matches = [
        (str(candidate_id), row)
        for candidate_id, row in derivatives.items()
        if str(row["variant"]) == args.variant
        and row["config"] == {"path": config_relative, "sha256": config_sha256}
    ]
    recovery_identity_claimed = (
        args.variant in known_variants or config_relative in known_config_paths
    )
    if recovery_identity_claimed and len(matches) != 1:
        raise ValueError(
            "recovery recipe identity is not an exact allowlisted config/hash/variant tuple"
        )
    recovery_run = len(matches) == 1
    recovery_candidate_id = matches[0][0] if recovery_run else None
    derivative_entry = matches[0][1] if recovery_run else None
    if args.frozen_finalist_selection is not None and not recovery_run:
        raise ValueError("full-training finalist selection is only valid for an allowlisted recipe")
    if recovery_run and (
        args.selection_split_manifest is None or args.validation_partition != "development"
    ):
        raise ValueError("performance-recovery training requires the frozen DEVELOPMENT partition")
    training = config["training"]
    inference = dict(config.get("inference", {}))
    requested_max_steps = args.max_steps or int(training["max_steps"])
    full_training_selection: dict[str, Any] | None = None
    selected_screen_finalist: dict[str, Any] | None = None
    selected_screen_finalist_sha256: str | None = None
    full_training_initialization_mode: str | None = None
    training_git_commit: str | None = None
    training_git_artifacts: list[dict[str, str]] | None = None

    def canonical_sha256(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def repository_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError as error:
            raise ValueError(f"authority path escapes the repository: {path}") from error

    def committed_file_binding(path: Path, commit: str) -> dict[str, str]:
        relative = repository_relative(path)
        blob = confirmation_process_control.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if blob.returncode:
            raise ValueError(f"required authority is absent from training commit: {relative}")
        blob_sha256 = hashlib.sha256(blob.stdout).hexdigest()
        disk_sha256 = _sha256(path)
        if blob_sha256 != disk_sha256:
            raise ValueError(f"required authority differs from training commit: {relative}")
        dirty = confirmation_process_control.run(
            ["git", "status", "--porcelain=v1", "--", relative],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise ValueError(f"required authority has uncommitted changes: {relative}")
        return {"path": relative, "sha256": disk_sha256}

    if recovery_run:
        assert derivative_entry is not None
        authorities = derivative_allowlist["authorities"]
        fixed_file_authorities = {
            name: value
            for name, value in authorities.items()
            if isinstance(value, dict) and set(value) == {"path", "sha256"}
        }
        for name, binding in fixed_file_authorities.items():
            authority_path = (ROOT / str(binding["path"])).resolve()
            if (
                not authority_path.is_file()
                or _sha256(authority_path) != str(binding["sha256"])
            ):
                raise ValueError(f"recovery {name} authority changed")
        if args.plan.resolve() != (ROOT / authorities["corpus_plan"]["path"]).resolve():
            raise ValueError("recovery corpus-plan path changed")
        if args.selection_split_manifest is None or args.selection_split_manifest.resolve() != (
            ROOT / authorities["selection_split"]["path"]
        ).resolve():
            raise ValueError("recovery selection-split path changed")
        if args.data_root.resolve() != (ROOT / authorities["data_root"]).resolve():
            raise ValueError("recovery data root changed")
        if args.cache_root.resolve() != (ROOT / authorities["cache_root"]).resolve():
            raise ValueError("recovery cache root changed")
        if str(config["dataset_revision"]) != str(authorities["dataset_revision"]):
            raise ValueError("recovery dataset revision changed")

    if args.frozen_finalist_selection is not None:
        assert recovery_candidate_id is not None
        assert derivative_entry is not None
        authorities = derivative_allowlist["authorities"]
        required_selection_path = (
            ROOT / authorities["screen_selection"]["path"]
        ).resolve()
        if args.frozen_finalist_selection.resolve() != required_selection_path:
            raise ValueError("full training requires the predeclared screen-selection path")
        repository_relative(args.checkpoint_dir)
        repository_relative(args.output)
        full_training_selection = json.loads(required_selection_path.read_text())
        if (
            not isinstance(full_training_selection, dict)
            or full_training_selection.get("schema_version") != 1
            or full_training_selection.get("status")
            != authorities["screen_selection"]["required_status"]
            or full_training_selection.get("protocol_id")
            != derivative_allowlist["protocol_id"]
        ):
            raise ValueError("full-training screen selection is not a valid frozen authority")
        selected_rows = [
            row
            for row in full_training_selection.get("finalists", [])
            if str(row.get("candidate_id")) == recovery_candidate_id
        ]
        if len(selected_rows) != 1:
            raise ValueError("recovery candidate is not exactly once in the frozen selection")
        selected_screen_finalist = selected_rows[0]
        if not isinstance(selected_screen_finalist, dict):
            raise ValueError("selected finalist row is not a mapping")
        if (
            str(selected_screen_finalist.get("variant")) != args.variant
            or selected_screen_finalist.get("config") != derivative_entry["config"]
            or selected_screen_finalist.get("deployment_fingerprint_sha256")
            != derivative_entry["deployment_fingerprint_sha256"]
        ):
            raise ValueError("selected finalist deployment identity differs from the allowlist")
        current_source_files = [
            {"path": repository_relative(path), "sha256": _sha256(path)}
            for path in sorted((ROOT / "src/timesfm_lab").rglob("*.py"))
        ]
        current_model_source = {
            "files": current_source_files,
            "sha256": canonical_sha256(current_source_files),
        }
        if selected_screen_finalist.get("model_source") != current_model_source:
            raise ValueError("model source differs from the frozen selected deployment")
        selected_inference = selected_screen_finalist.get("inference_implementation", {})
        trainer_relative = repository_relative(Path(__file__))
        if (
            not isinstance(selected_inference, dict)
            or selected_inference.get("path") != trainer_relative
            or selected_inference.get("sha256") != _sha256(Path(__file__))
        ):
            raise ValueError("inference implementation differs from the frozen selection")
        selected_screen_finalist_sha256 = canonical_sha256(selected_screen_finalist)

        registry = finalist_yaml.safe_load(
            (
                ROOT / derivative_allowlist["authorities"]["candidate_registry"]["path"]
            ).read_text()
        )
        if not isinstance(registry, dict):
            raise ValueError("candidate registry must be a mapping")
        registry_candidate = registry["candidates"][recovery_candidate_id]
        if args.resume is not None:
            full_training_initialization_mode = "resume"
        elif args.initialize_from is not None:
            initialization_path = args.initialize_from.resolve()
            selected_checkpoint = selected_screen_finalist["checkpoint"]
            if (
                initialization_path == (ROOT / selected_checkpoint["path"]).resolve()
                and _sha256(initialization_path) == selected_checkpoint["sha256"]
            ):
                full_training_initialization_mode = (
                    "selected_screen_checkpoint_weights_only"
                )
            else:
                declared = registry_candidate.get("initialization", {})
                if (
                    derivative_entry["candidate_registry_initialization_allowed"] is True
                    and declared.get("kind") == "checkpoint"
                    and initialization_path == (ROOT / declared["path"]).resolve()
                    and _sha256(initialization_path) == declared["file_sha256"]
                ):
                    full_training_initialization_mode = "candidate_registry_initialization"
                else:
                    raise ValueError("full-training initialization is not predeclared")
        else:
            declared = registry_candidate.get("initialization", {})
            if (
                derivative_entry["candidate_registry_initialization_allowed"] is True
                and declared.get("kind") == "seeded_random"
            ):
                full_training_initialization_mode = "candidate_registry_initialization"
            else:
                raise ValueError("full-training initialization is not predeclared")

        runtime_policy = derivative_allowlist["allowed_runtime_differences"]
        configured_training_seed = (
            args.training_seed if args.training_seed is not None else int(config["seed"])
        )
        configured_split_seed = (
            args.split_seed if args.split_seed is not None else int(config["seed"])
        )
        if configured_training_seed not in runtime_policy["training_seed"]:
            raise ValueError("full-training seed is outside the frozen allowlist")
        if requested_max_steps not in runtime_policy["maximum_steps"]:
            raise ValueError("full-training step budget is outside the frozen allowlist")
        if configured_split_seed != int(runtime_policy["split_seed"]):
            raise ValueError("full-training split seed changed")
        if bool(args.distributed) != bool(runtime_policy["distributed"]):
            raise ValueError("full-training distributed policy changed")
        if (not args.disable_early_stopping) != bool(
            runtime_policy["early_stopping_enabled"]
        ):
            raise ValueError("full-training early-stopping policy changed")

        training_git_commit = confirmation_process_control.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        committed_paths = [
            derivative_allowlist_path,
            required_selection_path,
            args.config.resolve(),
            args.plan.resolve(),
            args.selection_split_manifest.resolve(),
            (ROOT / authorities["screen_selection_config"]["path"]).resolve(),
            (ROOT / authorities["candidate_registry"]["path"]).resolve(),
            (ROOT / authorities["cache_audit"]["path"]).resolve(),
            Path(__file__).resolve(),
            *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
        ]
        training_git_artifacts = [
            committed_file_binding(path, training_git_commit)
            for path in dict.fromkeys(committed_paths)
        ]
    input_preprocessing = str(training.get("input_preprocessing", "masked_raw"))
    loss_reduction = str(training.get("loss_reduction", "observed_target_element"))
    if loss_reduction not in {
        "observed_target_element",
        "per_window_domain_balanced",
    }:
        raise ValueError(f"unsupported training loss reduction: {loss_reduction!r}")
    domain_weights: dict[str, float] = {}
    expected_domain_counts: dict[str, int] = {}
    domain_weight_source: dict[str, str] | None = None
    if loss_reduction == "per_window_domain_balanced":
        frozen_domain_weights = {
            "Econ/Fin": 1.0927936269053846,
            "Energy": 1.8427371914080113,
            "Healthcare": 1.564169783858156,
            "Nature": 0.7437324094220139,
            "Sales": 2.366550935229951,
            "Transport": 0.5916377338074877,
            "Web/CloudOps": 2.366550935229951,
        }
        frozen_domain_counts = {
            "Econ/Fin": 143447,
            "Energy": 85068,
            "Healthcare": 100218,
            "Nature": 210772,
            "Sales": 17806,
            "Transport": 347922,
            "Web/CloudOps": 22113,
        }
        frozen_domain_weight_source = {
            "plan_sha256": "2cd2967fa0ef4e1f6e7a18d050d7af1b7533458abab96e52b45a4c0caff0828c",
            "selection_split_manifest_sha256": (
                "9d3e06b328b76baaab558c18717b7336961f07e81261989349c0f4e20c899cd9"
            ),
            "outer_training_identity_sha256": (
                "24cad944c659cfe437f54b0050dac19fa9698ff45e43ce31907ac8caf824b9f4"
            ),
        }
        domain_weights = {
            str(domain): float(weight)
            for domain, weight in training.get("domain_weights", {}).items()
        }
        expected_domain_counts = {
            str(domain): int(count)
            for domain, count in training.get("domain_weight_outer_training_counts", {}).items()
        }
        domain_weight_source = {
            str(key): str(value) for key, value in training.get("domain_weight_source", {}).items()
        }
        if domain_weights != frozen_domain_weights:
            raise ValueError("S5 domain weights differ from the frozen activation gate")
        if expected_domain_counts != frozen_domain_counts:
            raise ValueError("S5 outer-training domain counts differ from the frozen gate")
        if domain_weight_source != frozen_domain_weight_source:
            raise ValueError("S5 domain-weight source fingerprints differ from the frozen gate")
        if training.get("domain_weight_denominator") != "unweighted_valid_windows":
            raise ValueError("S5 must use the frozen unweighted valid-window denominator")
        if (
            training.get("zero_target_window_policy")
            != "sequence_only_excluded_from_loss_and_denominator"
        ):
            raise ValueError("S5 zero-target-window policy differs from the frozen gate")
        if _sha256(args.plan) != frozen_domain_weight_source["plan_sha256"]:
            raise ValueError("S5 corpus plan differs from the frozen domain-weight source")
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
    if loss_reduction == "per_window_domain_balanced" and logical_batch_size_windows is None:
        raise ValueError("S5 domain-balanced reduction requires fixed logical batches")
    configured_seed = int(config["seed"])
    training_seed = args.training_seed if args.training_seed is not None else configured_seed
    split_seed = args.split_seed if args.split_seed is not None else configured_seed
    selection_manifest: dict[str, Any] | None = None
    selection_manifest_sha256: str | None = None
    selection_entries: dict[str, dict[str, Any]] = {}
    if args.selection_split_manifest is not None:
        selection_manifest_sha256 = _sha256(args.selection_split_manifest)
        if recovery_run and (
            selection_manifest_sha256
            != "9d3e06b328b76baaab558c18717b7336961f07e81261989349c0f4e20c899cd9"
        ):
            raise ValueError("performance-recovery training split authority changed")
        selection_manifest = json.loads(args.selection_split_manifest.read_text())
        if selection_manifest.get("schema_version") != 1:
            raise ValueError("unsupported recovery selection manifest schema")
        if selection_manifest.get("protocol_id") != "timesfm3-performance-recovery-v1.1":
            raise ValueError("unexpected recovery protocol in selection manifest")
        if selection_manifest.get("status") != "frozen_uninspected":
            raise ValueError("selection manifest is not frozen and uninspected")
        if selection_manifest.get("target_accessed") is not False:
            raise ValueError("selection manifest creation accessed target data")
        source = selection_manifest["source"]
        if source.get("plan_sha256") != _sha256(args.plan):
            raise ValueError("selection manifest corpus-plan hash mismatch")
        if source.get("dataset_revision") != str(config["dataset_revision"]):
            raise ValueError("selection manifest dataset revision mismatch")
        outer = selection_manifest["outer_split"]
        if str(outer["mode"]) != str(training["validation_split"]):
            raise ValueError("config validation mode differs from frozen selection split")
        if not math.isclose(
            float(outer["validation_fraction"]),
            float(training["validation_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("config validation fraction differs from frozen selection split")
        if int(outer["seed"]) != split_seed:
            raise ValueError("requested split seed differs from frozen selection split")
        selection_entries = {
            str(entry["dataset"]): entry for entry in selection_manifest["datasets"]
        }
        plan_datasets = {str(item["dataset"]) for item in plan["datasets"]}
        if set(selection_entries) != plan_datasets:
            raise ValueError("selection manifest dataset names differ from corpus plan")
    if loss_reduction == "per_window_domain_balanced":
        assert domain_weight_source is not None
        if (
            selection_manifest is None
            or selection_manifest_sha256 != domain_weight_source["selection_split_manifest_sha256"]
        ):
            raise ValueError("S5 requires its frozen recovery selection split")
        outer_training_identity = str(
            selection_manifest["totals"]["outer_training"]["identity_sha256"]
        )
        if outer_training_identity != domain_weight_source["outer_training_identity_sha256"]:
            raise ValueError("S5 outer-training identity differs from the frozen gate")
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
            selection_manifest=selection_manifest,
            selection_entry=selection_entries.get(str(item["dataset"])),
            validation_partition=args.validation_partition,
        )
        for item in plan["datasets"]
    ]
    corpus_load_seconds = time.perf_counter() - load_started
    if loss_reduction == "per_window_domain_balanced":
        actual_domain_counts = {domain: 0 for domain in domain_weights}
        for corpus in corpora:
            if corpus.domain not in actual_domain_counts:
                raise ValueError(f"S5 encountered an unfrozen domain: {corpus.domain!r}")
            actual_domain_counts[corpus.domain] += len(corpus.training_indices)
        if actual_domain_counts != expected_domain_counts:
            raise ValueError(
                "S5 realized outer-training domain counts differ from the frozen gate: "
                f"expected={expected_domain_counts}, realized={actual_domain_counts}"
            )
    student = build_student(config["student"])
    random_initialization_sha256 = _state_sha256(student)
    initialization_checkpoint = (
        str(args.initialize_from.resolve()) if args.initialize_from is not None else None
    )
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
    if loss_reduction == "per_window_domain_balanced" and weights != LossWeights(
        ground_truth=1.0,
        multivariate_kd=0.0,
        univariate_kd=0.0,
        cvrd=0.0,
    ):
        raise ValueError("activated S5 must preserve the selected GT-only objective exactly")
    objective = DistillationLoss(weights)
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    max_steps = requested_max_steps
    training_source_paths = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
    ]
    training_source_sha256 = {
        str(path.relative_to(ROOT)): _sha256(path) for path in training_source_paths
    }
    domain_weight_configuration_sha256 = (
        hashlib.sha256(
            json.dumps(
                {
                    "domain_weight_source": domain_weight_source,
                    "domain_weights": domain_weights,
                    "outer_training_counts": expected_domain_counts,
                    "denominator": training.get("domain_weight_denominator"),
                    "zero_target_window_policy": training.get("zero_target_window_policy"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if loss_reduction == "per_window_domain_balanced"
        else None
    )
    training_recipe_fingerprint = {
        "schema_version": 1,
        "config_sha256": _sha256(args.config),
        "corpus_plan_sha256": _sha256(args.plan),
        "selection_split_manifest_sha256": selection_manifest_sha256,
        "training_source_sha256": training_source_sha256,
        "variant": args.variant,
        "loss_weights": training["loss_weights"][args.variant],
        "loss_reduction": loss_reduction,
        "domain_weight_configuration_sha256": domain_weight_configuration_sha256,
        "training_seed": training_seed,
        "split_seed": split_seed,
        "validation_partition": args.validation_partition,
        "logical_batch_size_windows": logical_batch_size_windows,
        "maximum_steps": max_steps,
        "distributed": args.distributed,
        "early_stopping_enabled": not args.disable_early_stopping,
    }
    training_recipe_sha256 = hashlib.sha256(
        json.dumps(training_recipe_fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    initialization_origin_fingerprint = {
        "schema_version": 1,
        "kind": "checkpoint" if args.initialize_from is not None else "seeded_random",
        "random_initialization_sha256": random_initialization_sha256,
        "initialization_sha256": initialization_sha256,
        "initialization_checkpoint": initialization_checkpoint,
        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
    }
    initialization_origin_sha256 = hashlib.sha256(
        json.dumps(
            initialization_origin_fingerprint, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    full_training_launch_authority: dict[str, Any] | None = None
    full_training_launch_authority_sha256: str | None = None
    resume_full_training_launch_authority: dict[str, Any] | None = None
    resume_full_training_launch_authority_sha256: str | None = None
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    batch_offset = 0
    best_score = math.inf
    plateau_reference_score = math.inf
    stale_evaluations = 0
    learning_curve: list[dict[str, Any]] = []
    trained_windows = 0
    observed_targets_processed = 0
    trainable_windows_processed: int | None = (
        0 if loss_reduction == "per_window_domain_balanced" else None
    )
    weighted_trainable_windows_processed: float | None = (
        0.0 if loss_reduction == "per_window_domain_balanced" else None
    )
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
    plateau_state_upgraded_from_legacy_resume = False

    def replay_plateau_state(
        curve: list[dict[str, Any]],
    ) -> tuple[float, int]:
        reference = math.inf
        stale = 0
        threshold = float(training["plateau_min_relative_improvement"])
        for row in curve:
            score = _validation_score(row["validation"], training)
            if not math.isfinite(score) or score <= 0:
                raise ValueError("learning curve contains an invalid validation score")
            if not math.isfinite(reference):
                reference = score
                stale = 0
                continue
            relative = (reference - score) / reference
            if relative >= threshold:
                reference = score
                stale = 0
            else:
                stale += 1
        return reference, stale

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        resume_full_training_launch_authority = state.get("full_training_launch_authority")
        resume_full_training_launch_authority_sha256 = state.get(
            "full_training_launch_authority_sha256"
        )
        if full_training_selection is not None:
            if not isinstance(resume_full_training_launch_authority, dict):
                raise ValueError("full-training resume lacks its frozen launch authority")
            full_training_initialization_mode = str(
                resume_full_training_launch_authority.get("initialization_mode", "")
            )
        checkpoint_loss_reduction = str(state.get("loss_reduction", "observed_target_element"))
        if checkpoint_loss_reduction != loss_reduction:
            raise ValueError(
                "resume loss-reduction mismatch: "
                f"checkpoint={checkpoint_loss_reduction}, requested={loss_reduction}"
            )
        if (
            state.get("domain_weight_configuration_sha256") != domain_weight_configuration_sha256
            and loss_reduction == "per_window_domain_balanced"
        ):
            raise ValueError("resume domain-weight configuration mismatch")
        checkpoint_recipe_sha256 = state.get("training_recipe_sha256")
        checkpoint_recipe_fingerprint = state.get("training_recipe_fingerprint")
        if (
            loss_reduction == "per_window_domain_balanced"
            or checkpoint_recipe_sha256 is not None
            or checkpoint_recipe_fingerprint is not None
        ) and (
            checkpoint_recipe_sha256 != training_recipe_sha256
            or checkpoint_recipe_fingerprint != training_recipe_fingerprint
        ):
            raise ValueError(
                "resume training-recipe fingerprint mismatch: "
                f"checkpoint={checkpoint_recipe_sha256}, requested={training_recipe_sha256}"
            )
        checkpoint_initialization_origin_sha256 = state.get("initialization_origin_sha256")
        checkpoint_initialization_origin_fingerprint = state.get(
            "initialization_origin_fingerprint"
        )
        if (
            checkpoint_initialization_origin_sha256 is not None
            or checkpoint_initialization_origin_fingerprint is not None
        ):
            if not isinstance(checkpoint_initialization_origin_fingerprint, dict):
                raise ValueError("resume initialization-origin fingerprint is malformed")
            recomputed_initialization_origin_sha256 = hashlib.sha256(
                json.dumps(
                    checkpoint_initialization_origin_fingerprint,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if checkpoint_initialization_origin_sha256 != recomputed_initialization_origin_sha256:
                raise ValueError("resume initialization-origin fingerprint is corrupt")
            initialization_origin_fingerprint = checkpoint_initialization_origin_fingerprint
            initialization_origin_sha256 = checkpoint_initialization_origin_sha256
            random_initialization_sha256 = initialization_origin_fingerprint[
                "random_initialization_sha256"
            ]
            initialization_sha256 = initialization_origin_fingerprint["initialization_sha256"]
            initialization_checkpoint = initialization_origin_fingerprint[
                "initialization_checkpoint"
            ]
            initialization_checkpoint_sha256 = initialization_origin_fingerprint[
                "initialization_checkpoint_sha256"
            ]
        elif loss_reduction == "per_window_domain_balanced":
            raise ValueError("S5 resume lacks its immutable initialization origin")
        else:
            initialization_origin_fingerprint = {
                "schema_version": 1,
                "kind": "legacy_resume_unrecorded",
                "resume_checkpoint_sha256": _sha256(args.resume),
            }
            initialization_origin_sha256 = hashlib.sha256(
                json.dumps(
                    initialization_origin_fingerprint,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            random_initialization_sha256 = None
            initialization_sha256 = None
            initialization_checkpoint = None
            initialization_checkpoint_sha256 = None
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
        if state.get("selection_split_manifest_sha256") != selection_manifest_sha256:
            raise ValueError("resume selection-split manifest mismatch")
        if state.get("validation_partition") != args.validation_partition:
            raise ValueError("resume validation partition mismatch")
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        epoch = int(state["epoch"])
        batch_offset = int(state["batch_offset"])
        best_score = float(state["best_score"])
        learning_curve = list(state["learning_curve"])
        replayed_reference, replayed_stale = replay_plateau_state(learning_curve)
        if "plateau_reference_score" in state:
            if state.get("early_stopping_state_schema_version") != 2:
                raise ValueError("resume plateau state has an unsupported schema")
            plateau_reference_score = float(state["plateau_reference_score"])
            stale_evaluations = int(state["stale_evaluations"])
            if (
                not math.isclose(
                    plateau_reference_score,
                    replayed_reference,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or stale_evaluations != replayed_stale
            ):
                raise ValueError("resume plateau state disagrees with its learning curve")
        else:
            if "early_stopping_state_schema_version" in state:
                raise ValueError("resume plateau schema exists without its reference score")
            plateau_reference_score = replayed_reference
            stale_evaluations = replayed_stale
            plateau_state_upgraded_from_legacy_resume = True
        trained_windows = int(state.get("trained_windows", 0))
        observed_targets_processed = int(state.get("observed_targets_processed", 0))
        if loss_reduction == "per_window_domain_balanced":
            trainable_windows_processed = int(state["trainable_windows_processed"])
            weighted_trainable_windows_processed = float(
                state["weighted_trainable_windows_processed"]
            )
        physical_microbatches_processed = int(state.get("physical_microbatches_processed", step))
        physical_microbatch_windows_sum = int(
            state.get("physical_microbatch_windows_sum", trained_windows)
        )
        physical_microbatch_windows_min = float(
            state.get("physical_microbatch_windows_min", math.inf)
        )
        physical_microbatch_windows_max = int(state.get("physical_microbatch_windows_max", 0))
        optimizer_batch_windows_sum = int(state.get("optimizer_batch_windows_sum", trained_windows))
        optimizer_batch_windows_min = float(state.get("optimizer_batch_windows_min", math.inf))
        optimizer_batch_windows_max = int(state.get("optimizer_batch_windows_max", 0))
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        sequence_chain = bytes.fromhex(state.get("training_sequence_sha256", bytes(32).hex()))
        train_sums.update(state.get("train_sums", {}))
        train_weight_sum = float(state.get("train_weight_sum", trained_windows))
        gradient_norm_sum = float(state.get("gradient_norm_sum", 0.0))
        gradient_clip_count = int(state.get("gradient_clip_count", 0))

    if full_training_selection is not None:
        assert recovery_candidate_id is not None
        assert derivative_entry is not None
        assert selected_screen_finalist is not None
        assert selected_screen_finalist_sha256 is not None
        assert training_git_commit is not None
        assert training_git_artifacts is not None
        assert full_training_initialization_mode is not None
        authorities = derivative_allowlist["authorities"]
        full_training_launch_authority = {
            "schema_version": 1,
            "status": "frozen_full_training_launch_authority",
            "protocol_id": derivative_allowlist["protocol_id"],
            "source_candidate_id": recovery_candidate_id,
            "variant": args.variant,
            "derivative_allowlist": {
                "path": repository_relative(derivative_allowlist_path),
                "sha256": derivative_allowlist_expected_sha256,
                "entry": derivative_entry,
            },
            "screen_selection": {
                "path": repository_relative(args.frozen_finalist_selection),
                "sha256": _sha256(args.frozen_finalist_selection),
                "selected_finalist_sha256": selected_screen_finalist_sha256,
                "selected_deployment": {
                    name: selected_screen_finalist[name]
                    for name in (
                        "candidate_id",
                        "variant",
                        "config",
                        "checkpoint",
                        "deployment_fingerprint_sha256",
                        "model_source",
                        "inference_implementation",
                    )
                },
            },
            "training_commit": training_git_commit,
            "repository_git_artifacts": training_git_artifacts,
            "config": {"path": config_relative, "sha256": config_sha256},
            "corpus": {
                "plan": authorities["corpus_plan"],
                "cache_audit": authorities["cache_audit"],
                "data_root": authorities["data_root"],
                "cache_root": authorities["cache_root"],
                "dataset_revision": authorities["dataset_revision"],
            },
            "selection_split": authorities["selection_split"],
            "training_recipe_fingerprint": training_recipe_fingerprint,
            "training_recipe_sha256": training_recipe_sha256,
            "initialization_mode": full_training_initialization_mode,
            "initialization_origin_fingerprint": initialization_origin_fingerprint,
            "initialization_origin_sha256": initialization_origin_sha256,
        }
        full_training_launch_authority_sha256 = canonical_sha256(
            full_training_launch_authority
        )
        if args.resume is not None and (
            resume_full_training_launch_authority != full_training_launch_authority
            or resume_full_training_launch_authority_sha256
            != full_training_launch_authority_sha256
        ):
            raise ValueError("resume checkpoint has another full-training launch authority")
    elif args.resume is not None and (
        resume_full_training_launch_authority is not None
        or resume_full_training_launch_authority_sha256 is not None
    ):
        raise ValueError("cannot resume a full-training finalist outside its frozen authority")

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
    stopped_for_plateau = (
        args.resume is not None
        and not args.disable_early_stopping
        and step >= int(training["plateau_min_steps"])
        and stale_evaluations >= int(training["plateau_patience_evaluations"])
    )
    started = time.perf_counter()
    training_model.train()
    validate_at_start = bool(training.get("validate_at_start", False))
    if validate_at_start and step == 0 and not learning_curve:
        validation_started = time.perf_counter()
        initial_validation = (
            _validate(student, corpora, device, input_preprocessing, inference) if is_main else None
        )
        if args.distributed:
            shared_validation = [initial_validation]
            torch.distributed.broadcast_object_list(shared_validation, src=0)
            initial_validation = shared_validation[0]
        assert initial_validation is not None
        initial_validation_seconds = time.perf_counter() - validation_started
        best_score = _validation_score(initial_validation, training)
        plateau_reference_score = best_score
        learning_curve.append(
            {
                "step": 0,
                "epoch": epoch,
                "windows_processed": trained_windows,
                "observed_targets_processed": observed_targets_processed,
                "trainable_windows_processed": trainable_windows_processed,
                "weighted_trainable_windows_processed": (weighted_trainable_windows_processed),
                "optimizer_batches_processed": step,
                "physical_microbatches_processed": physical_microbatches_processed,
                "learning_rate": float(training["learning_rate"]),
                "validation_seconds": initial_validation_seconds,
                "relative_improvement_from_best": None,
                "relative_improvement_from_plateau_reference": None,
                "plateau_reference_score": plateau_reference_score,
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

            # The opt-in paths scan only the small target slices first. S5
            # additionally records valid windows so all-zero-target windows
            # stay in the frozen sequence but receive no loss weight.
            microbatch_target_counts: list[tuple[int, int]] | None = None
            logical_valid_windows: int | None = None
            if loss_reduction == "per_window_domain_balanced":
                microbatch_target_counts = []
                for corpus_index, global_indices in optimizer_batch:
                    corpus = corpora[corpus_index]
                    microbatch_observed_targets = 0
                    microbatch_valid_windows = 0
                    for index in global_indices:
                        row = int(corpus.row_index[index])
                        end = int(corpus.context_end[index])
                        target_values = corpus.source[row][:, end : end + corpus.horizon]
                        observed = int(np.count_nonzero(np.isfinite(target_values)))
                        microbatch_observed_targets += observed
                        microbatch_valid_windows += int(observed > 0)
                    microbatch_target_counts.append(
                        (microbatch_observed_targets, microbatch_valid_windows)
                    )
                logical_observed_targets = sum(
                    observed_targets for observed_targets, _ in microbatch_target_counts
                )
                logical_valid_windows = sum(
                    valid_windows for _, valid_windows in microbatch_target_counts
                )
            else:
                # Historical one-microbatch behavior remains the default,
                # including its established two-rank weighting.
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
            if logical_valid_windows is not None and logical_valid_windows <= 0:
                raise ValueError(
                    f"optimizer batch at epoch={epoch} offset={offset} has no valid windows"
                )

            optimizer.zero_grad(set_to_none=True)
            statistics = torch.zeros(len(train_sums) + 1, dtype=torch.float64, device=device)
            datasets_in_batch: list[str] = []
            realized_observed_targets = 0
            realized_valid_windows = 0
            for microbatch_position, (corpus_index, global_indices) in enumerate(optimizer_batch):
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
                        loss_reduction,
                    )
                if loss_reduction == "per_window_domain_balanced":
                    finite = torch.tensor(
                        float(all(torch.isfinite(value).all() for value in values.values())),
                        device=device,
                    )
                else:
                    finite = torch.tensor(
                        float(all(torch.isfinite(value) for value in values.values())),
                        device=device,
                    )
                if args.distributed:
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError(f"non-finite loss at step {step + 1} on {corpus.name}")
                local_observed_targets = torch.isfinite(target).sum().to(torch.float64)
                local_weight = local_observed_targets * 9
                global_weight = local_weight.clone()
                if args.distributed:
                    torch.distributed.all_reduce(global_weight, op=torch.distributed.ReduceOp.SUM)
                if loss_reduction == "per_window_domain_balanced":
                    assert microbatch_target_counts is not None
                    assert logical_valid_windows is not None
                    valid_windows = torch.isfinite(target).flatten(1).any(dim=1)
                    local_valid_windows = int(valid_windows.sum().item())
                    expected_observed_targets, expected_valid_windows = microbatch_target_counts[
                        microbatch_position
                    ]
                    if int(local_observed_targets.item()) != expected_observed_targets:
                        raise RuntimeError(
                            "CPU/GPU microbatch observed-target count mismatch: "
                            f"expected={expected_observed_targets}, "
                            f"realized={int(local_observed_targets.item())}"
                        )
                    if local_valid_windows != expected_valid_windows:
                        raise RuntimeError(
                            "CPU/GPU microbatch valid-window count mismatch: "
                            f"expected={expected_valid_windows}, "
                            f"realized={local_valid_windows}"
                        )
                    expected_shape = (len(indices),)
                    if any(value.shape != expected_shape for value in values.values()):
                        raise RuntimeError(
                            "S5 per-window loss component shape mismatch: "
                            f"expected={expected_shape}, "
                            f"realized={tuple(values['loss'].shape)}"
                        )
                    domain_weight = domain_weights[corpus.domain]
                    if local_valid_windows:
                        (
                            values["loss"][valid_windows].sum()
                            * (domain_weight / logical_valid_windows)
                        ).backward()
                    microbatch_statistics = torch.stack(
                        [
                            values[key][valid_windows].detach().to(torch.float64).sum()
                            * domain_weight
                            for key in train_sums
                        ]
                        + [
                            torch.tensor(
                                local_valid_windows,
                                dtype=torch.float64,
                                device=device,
                            )
                        ]
                    )
                    realized_valid_windows += local_valid_windows
                    assert trainable_windows_processed is not None
                    assert weighted_trainable_windows_processed is not None
                    trainable_windows_processed += local_valid_windows
                    weighted_trainable_windows_processed += local_valid_windows * domain_weight
                else:
                    if logical_observed_targets is None:
                        loss_scale = world_size * local_weight / global_weight.clamp_min(1)
                    else:
                        loss_scale = local_observed_targets / logical_observed_targets
                    (values["loss"] * loss_scale).backward()
                    microbatch_statistics = torch.stack(
                        [
                            values[key].detach().to(torch.float64) * local_weight
                            for key in train_sums
                        ]
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
            if microbatch_target_counts is not None and realized_valid_windows != sum(
                valid_windows for _, valid_windows in microbatch_target_counts
            ):
                raise RuntimeError(
                    "CPU/GPU logical-batch valid-window count mismatch: "
                    f"expected={sum(valid for _, valid in microbatch_target_counts)}, "
                    f"realized={realized_valid_windows}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(),
                float(training["gradient_clip"]),
                error_if_nonfinite=True,
            )
            gradient_norm_value = float(gradient_norm.detach())
            if not math.isfinite(gradient_norm_value):
                raise FloatingPointError(
                    f"non-finite gradient norm at step {step + 1}; optimizer not advanced"
                )
            gradient_norm_sum += gradient_norm_value
            gradient_clip_count += int(gradient_norm_value > float(training["gradient_clip"]))
            optimizer.step()
            step += 1
            trained_windows += optimizer_batch_windows
            optimizer_batch_windows_sum += optimizer_batch_windows
            optimizer_batch_windows_min = min(optimizer_batch_windows_min, optimizer_batch_windows)
            optimizer_batch_windows_max = max(optimizer_batch_windows_max, optimizer_batch_windows)
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
                float(logical_valid_windows)
                if logical_valid_windows is not None
                else (
                    float(logical_observed_targets)
                    if logical_observed_targets is not None
                    else float(optimizer_batch_windows)
                )
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
                plateau_relative_improvement = (
                    (plateau_reference_score - score) / plateau_reference_score
                    if math.isfinite(plateau_reference_score)
                    else math.inf
                )
                improved = score < best_score
                materially_improved = plateau_relative_improvement >= float(
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
                if materially_improved:
                    plateau_reference_score = score
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                learning_curve.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "windows_processed": trained_windows,
                        "observed_targets_processed": observed_targets_processed,
                        "trainable_windows_processed": trainable_windows_processed,
                        "weighted_trainable_windows_processed": (
                            weighted_trainable_windows_processed
                        ),
                        "optimizer_batches_processed": step,
                        "physical_microbatches_processed": physical_microbatches_processed,
                        "learning_rate": lr,
                        "validation_seconds": validation_seconds,
                        "relative_improvement_from_best": relative_improvement,
                        "relative_improvement_from_plateau_reference": (
                            plateau_relative_improvement
                        ),
                        "plateau_reference_score": plateau_reference_score,
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
                    plateau_reference_score=plateau_reference_score,
                    early_stopping_state_schema_version=2,
                    stale_evaluations=stale_evaluations,
                    learning_curve=learning_curve,
                    trained_windows=trained_windows,
                    observed_targets_processed=observed_targets_processed,
                    trainable_windows_processed=trainable_windows_processed,
                    weighted_trainable_windows_processed=(weighted_trainable_windows_processed),
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
                    selection_split_manifest_sha256=selection_manifest_sha256,
                    validation_partition=args.validation_partition,
                    loss_reduction=loss_reduction,
                    domain_weight_configuration_sha256=(domain_weight_configuration_sha256),
                    training_recipe_fingerprint=training_recipe_fingerprint,
                    training_recipe_sha256=training_recipe_sha256,
                    initialization_origin_fingerprint=(initialization_origin_fingerprint),
                    initialization_origin_sha256=initialization_origin_sha256,
                    full_training_launch_authority=full_training_launch_authority,
                    full_training_launch_authority_sha256=(
                        full_training_launch_authority_sha256
                    ),
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
    best_checkpoint = args.checkpoint_dir / f"student-{args.variant}-best.pt"
    best_checkpoint_sha256 = _sha256(best_checkpoint) if is_main else None
    final_checkpoint_sha256 = _sha256(final_checkpoint) if is_main else None
    full_training_authority = None
    full_training_authority_sha256 = None
    if is_main and full_training_launch_authority is not None:
        assert best_checkpoint_sha256 is not None
        assert final_checkpoint_sha256 is not None
        full_training_authority = {
            "schema_version": 1,
            "status": "completed_full_training_authority",
            "launch": full_training_launch_authority,
            "launch_sha256": full_training_launch_authority_sha256,
            "best_checkpoint": {
                "path": repository_relative(best_checkpoint),
                "sha256": best_checkpoint_sha256,
            },
            "final_checkpoint": {
                "path": repository_relative(final_checkpoint),
                "sha256": final_checkpoint_sha256,
            },
        }
        full_training_authority_sha256 = canonical_sha256(full_training_authority)
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
            "selection_split_manifest": (
                str(args.selection_split_manifest.resolve())
                if args.selection_split_manifest is not None
                else None
            ),
            "selection_split_manifest_sha256": selection_manifest_sha256,
            "training_recipe_fingerprint": training_recipe_fingerprint,
            "training_recipe_sha256": training_recipe_sha256,
            "initialization_origin_fingerprint": initialization_origin_fingerprint,
            "initialization_origin_sha256": initialization_origin_sha256,
            "training_source_sha256": training_source_sha256,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "final_checkpoint_sha256": final_checkpoint_sha256,
            "full_training_launch_authority": full_training_launch_authority,
            "full_training_launch_authority_sha256": (
                full_training_launch_authority_sha256
            ),
            "full_training_authority": full_training_authority,
            "full_training_authority_sha256": full_training_authority_sha256,
            "validation_partition": args.validation_partition,
            "confirmation_partition_accessed": args.validation_partition == "confirmation",
            "gift_eval_data_accessed": False,
            "resume_checkpoint": str(args.resume.resolve()) if args.resume is not None else None,
            "initialization_checkpoint": initialization_checkpoint,
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
                "trainable_windows_processed": trainable_windows_processed,
                "weighted_trainable_windows_processed": (weighted_trainable_windows_processed),
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
                    "unweighted_valid_windows"
                    if loss_reduction == "per_window_domain_balanced"
                    else (
                        "observed_target_positions"
                        if logical_batch_size_windows is not None
                        else "windows_historical"
                    )
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
                "plateau_reference_score": plateau_reference_score,
                "plateau_stale_evaluations": stale_evaluations,
                "plateau_state_schema_version": 2,
                "plateau_state_upgraded_from_legacy_resume": (
                    plateau_state_upgraded_from_legacy_resume
                ),
                "validate_at_start": validate_at_start,
                "maximum_steps": max_steps,
                "loss_weights": training["loss_weights"][args.variant],
                "loss_reduction": loss_reduction,
                "domain_weights": domain_weights or None,
                "domain_weight_denominator": training.get("domain_weight_denominator"),
                "zero_target_window_policy": training.get("zero_target_window_policy"),
                "domain_weight_source": domain_weight_source,
                "domain_weight_outer_training_counts": (expected_domain_counts or None),
                "domain_weight_configuration_sha256": (domain_weight_configuration_sha256),
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
            "best_checkpoint": str(best_checkpoint.resolve()),
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
