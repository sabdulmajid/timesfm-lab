#!/usr/bin/env python3
"""Characterize teacher and optional student cross-variate responses on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import split_cache_indices

QUANTILES = np.arange(0.1, 1.0, 0.1, dtype=np.float64)
NEAR_ZERO_THRESHOLD = 1e-3
MAXIMUM_SPEARMAN_ELEMENTS = 2_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _correlations(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    count = len(left)
    if count < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return {
            "pearson_count": count,
            "pearson": None,
            "spearman_count": count,
            "spearman": None,
        }
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    pearson = np.dot(left_centered, right_centered) / denominator
    if count > MAXIMUM_SPEARMAN_ELEMENTS:
        # Evenly spread deterministic coordinates avoid a multi-gigabyte rank sort
        # while retaining the complete tensor for exact Pearson and error metrics.
        indices = np.linspace(0, count - 1, MAXIMUM_SPEARMAN_ELEMENTS, dtype=np.int64)
        spearman_left = left[indices]
        spearman_right = right[indices]
    else:
        spearman_left = left
        spearman_right = right
    return {
        "pearson_count": count,
        "pearson": _finite_number(float(pearson)),
        "spearman_count": len(spearman_left),
        "spearman": _finite_number(float(spearmanr(spearman_left, spearman_right).statistic)),
    }


def _distribution(values: npt.NDArray[Any]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "maximum"),
                quantiles,
                strict=True,
            )
        },
    }


def _safe_mean_scale(context: npt.NDArray[Any], epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Float64 equivalent of the project's overflow-stable TimesFM normalization."""
    values = np.asarray(context, dtype=np.float64)
    observed = np.isfinite(values)
    count = observed.sum(axis=-1, keepdims=True).clip(min=1)
    absolute_max = np.max(np.where(observed, np.abs(values), 0.0), axis=-1, keepdims=True)
    mean_divisor = np.where(absolute_max > 0, absolute_max, 1.0)
    scaled = np.where(observed, values / mean_divisor, 0.0)
    mean = scaled.sum(axis=-1, keepdims=True) / count * mean_divisor
    centered = np.where(observed, values - mean, 0.0)
    amplitude = np.max(np.abs(centered), axis=-1, keepdims=True)
    variance_divisor = np.where(amplitude > 0, amplitude, 1.0)
    variance = np.square(centered / variance_divisor).sum(axis=-1, keepdims=True) / count
    raw_scale = variance_divisor * np.sqrt(variance)
    scale = np.where(raw_scale < epsilon, 1.0, raw_scale)
    return mean, scale


def _pinball_per_window(
    prediction: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    return _pinball_per_window_horizon(prediction, target).mean(axis=1)


def _pinball_per_window_horizon(
    prediction: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.isfinite(target)
    error = target[..., None] - prediction
    loss = np.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
    expanded = np.broadcast_to(observed[..., None], loss.shape)
    numerator = np.where(expanded, loss, 0.0).sum(axis=(1, 3))
    denominator = expanded.sum(axis=(1, 3)).clip(min=1)
    return numerator / denominator


def _median_mae_per_window(
    prediction: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    return _median_mae_per_window_horizon(prediction, target).mean(axis=1)


def _median_mae_per_window_horizon(
    prediction: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    prediction = np.asarray(prediction, dtype=np.float64)[..., 4]
    target = np.asarray(target, dtype=np.float64)
    observed = np.isfinite(target)
    error = np.where(observed, np.abs(target - prediction), 0.0)
    return error.sum(axis=1) / observed.sum(axis=1).clip(min=1)


def _teacher_summary(
    response_raw: npt.NDArray[Any],
    response_normalized: npt.NDArray[Any],
    mv_pinball: npt.NDArray[Any],
    uv_pinball: npt.NDArray[Any],
    mv_mae: npt.NDArray[Any],
    uv_mae: npt.NDArray[Any],
) -> dict[str, Any]:
    response = np.asarray(response_normalized, dtype=np.float64)
    flat = response.reshape(-1)
    strength = np.mean(np.abs(response), axis=(1, 2, 3))
    pinball_gain = np.asarray(uv_pinball) - np.asarray(mv_pinball)
    median_mae_gain = np.asarray(uv_mae) - np.asarray(mv_mae)
    signs = np.where(flat > NEAR_ZERO_THRESHOLD, 1, np.where(flat < -NEAR_ZERO_THRESHOLD, -1, 0))
    count = max(len(flat), 1)
    edges = np.quantile(strength, np.linspace(0.0, 1.0, 6))
    response_bins = []
    for index in range(5):
        upper_inclusive = index == 4
        mask = (strength >= edges[index]) & (
            (strength <= edges[index + 1]) if upper_inclusive else (strength < edges[index + 1])
        )
        response_bins.append(
            {
                "bin": index + 1,
                "lower_response_strength": float(edges[index]),
                "upper_response_strength": float(edges[index + 1]),
                "windows": int(mask.sum()),
                "mean_pinball_gain_uv_minus_mv": (
                    float(pinball_gain[mask].mean()) if mask.any() else None
                ),
                "mean_median_mae_gain_uv_minus_mv": (
                    float(median_mae_gain[mask].mean()) if mask.any() else None
                ),
            }
        )
    return {
        "normalization": "response divided by TimesFM-safe per-window context scale",
        "near_zero_absolute_threshold_normalized": NEAR_ZERO_THRESHOLD,
        "absolute_response_raw": _distribution(np.abs(response_raw)),
        "absolute_response_normalized": _distribution(np.abs(response)),
        "near_zero_fraction": float(np.count_nonzero(signs == 0) / count),
        "sign_fraction": {
            "negative": float(np.count_nonzero(signs == -1) / count),
            "near_zero": float(np.count_nonzero(signs == 0) / count),
            "positive": float(np.count_nonzero(signs == 1) / count),
        },
        "teacher_accuracy": {
            "mv_normalized_pinball_mean": float(np.mean(mv_pinball)),
            "uv_normalized_pinball_mean": float(np.mean(uv_pinball)),
            "pinball_gain_uv_minus_mv_mean": float(np.mean(pinball_gain)),
            "mv_better_pinball_fraction": float(np.mean(pinball_gain > 0)),
            "mv_normalized_median_mae_mean": float(np.mean(mv_mae)),
            "uv_normalized_median_mae_mean": float(np.mean(uv_mae)),
            "median_mae_gain_uv_minus_mv_mean": float(np.mean(median_mae_gain)),
            "mv_better_median_mae_fraction": float(np.mean(median_mae_gain > 0)),
        },
        "response_strength_vs_teacher_gain": {
            "normalized_pinball": _correlations(strength, pinball_gain),
            "normalized_median_mae": _correlations(strength, median_mae_gain),
            "response_strength_quintiles": response_bins,
        },
    }


def _student_summary(
    student_response: npt.NDArray[Any], teacher_response: npt.NDArray[Any]
) -> dict[str, Any]:
    student = np.asarray(student_response, dtype=np.float64)
    teacher = np.asarray(teacher_response, dtype=np.float64)
    flat_student = student.reshape(-1)
    flat_teacher = teacher.reshape(-1)
    teacher_denominator = np.abs(flat_teacher).sum()
    threshold = NEAR_ZERO_THRESHOLD
    student_sign = np.where(flat_student > threshold, 1, np.where(flat_student < -threshold, -1, 0))
    teacher_sign = np.where(flat_teacher > threshold, 1, np.where(flat_teacher < -threshold, -1, 0))
    student_vectors = student.reshape(len(student), -1)
    teacher_vectors = teacher.reshape(len(teacher), -1)
    dot = np.sum(student_vectors * teacher_vectors, axis=1)
    denominator = np.linalg.norm(student_vectors, axis=1) * np.linalg.norm(teacher_vectors, axis=1)
    valid_cosine = denominator > 0
    per_window_cosine = dot[valid_cosine] / denominator[valid_cosine]
    global_denominator = np.linalg.norm(flat_student) * np.linalg.norm(flat_teacher)
    return {
        "response_nmae": (
            float(np.abs(flat_student - flat_teacher).sum() / teacher_denominator)
            if teacher_denominator > 0
            else None
        ),
        "response_correlations": _correlations(flat_student, flat_teacher),
        "sign_agreement_fraction": float(np.mean(student_sign == teacher_sign)),
        "directional_cosine_mean_over_windows": (
            float(per_window_cosine.mean()) if len(per_window_cosine) else None
        ),
        "directional_cosine_global": (
            float(np.dot(flat_student, flat_teacher) / global_denominator)
            if global_denominator > 0
            else None
        ),
        "magnitude_mae_normalized": float(np.mean(np.abs(np.abs(student) - np.abs(teacher)))),
    }


def _load_arrays(cache_paths: list[Path]) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in cache_paths:
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if sidecar["sha256"] != _sha256(path):
            raise ValueError(f"cache checksum failed: {path}")
        with np.load(path) as shard:
            for key in shard.files:
                pieces[key].append(shard[key])
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _student_predictions(
    checkpoint: Path,
    student_config: dict[str, Any],
    contexts: list[np.ndarray],
    horizon: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from timesfm_lab.models import StudentConfig, TimesFMStudent

    if not torch.cuda.is_available():
        raise RuntimeError("student response diagnostics require CUDA")
    device = torch.device("cuda:0")
    model = TimesFMStudent(StudentConfig(**student_config)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    mv_parts = []
    uv_parts = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            values = torch.from_numpy(np.stack(contexts[start : start + batch_size])).to(device)
            batch, variates, length = values.shape
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mv = model(values, horizon)
                uv = model(values.reshape(batch * variates, 1, length), horizon).reshape(
                    batch, variates, horizon, 9
                )
            mv_parts.append(mv.float().cpu().numpy())
            uv_parts.append(uv.float().cpu().numpy())
    return np.concatenate(mv_parts), np.concatenate(uv_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-validation-windows-per-dataset", type=int, default=4096)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--student-label")
    parser.add_argument("--student-batch-size", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    training = config["training"]
    epsilon = float(config["student"]["normalization_epsilon"])
    seed = int(config["seed"])

    from datasets import load_from_disk  # type: ignore[import-untyped]

    records: list[dict[str, Any]] = []
    aggregate: dict[str, list[np.ndarray]] = defaultdict(list)
    by_domain: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    by_horizon: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    by_quantile: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for item in plan["datasets"]:
        if item["view_class"] != "true_multivariate":
            continue
        name = str(item["dataset"])
        cache_paths = sorted((args.cache_root / name).glob("shard-*.npz"))
        if not cache_paths:
            raise FileNotFoundError(f"no production cache for {name}")
        arrays = _load_arrays(cache_paths)
        forbidden = {"teacher_output"} & arrays.keys()
        if forbidden or "teacher_multivariate" not in arrays or "teacher_univariate" not in arrays:
            raise ValueError(f"invalid true-MV cache schema for {name}")
        splits, split_report = split_cache_indices(
            arrays["row_index"],
            arrays["context_end"],
            context_length=int(item["context"]),
            horizon=int(item["horizon"]),
            validation_fraction=float(training["validation_fraction"]),
            seed=seed,
            mode=str(training["validation_split"]),
        )
        validation = np.asarray(splits["validation"])
        if len(validation) > args.maximum_validation_windows_per_dataset:
            rng = np.random.default_rng(seed + sum(name.encode()))
            validation = np.sort(
                rng.choice(validation, args.maximum_validation_windows_per_dataset, replace=False)
            )
        dataset = load_from_disk(str(args.data_root / name), keep_in_memory=False)
        source = [
            np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
            for row in range(len(dataset))
        ]
        contexts = []
        targets = []
        means = []
        scales = []
        context_length = int(item["context"])
        horizon = int(item["horizon"])
        for index in validation:
            row = int(arrays["row_index"][index])
            end = int(arrays["context_end"][index])
            context = source[row][:, end - context_length : end]
            target = source[row][:, end : end + horizon]
            mean, scale = _safe_mean_scale(context, epsilon)
            contexts.append(np.ascontiguousarray(context))
            targets.append(target)
            means.append(mean)
            scales.append(scale)
        target = np.stack(targets).astype(np.float64)
        mean = np.stack(means).astype(np.float64)
        scale = np.stack(scales).astype(np.float64)
        mv = np.asarray(arrays["teacher_multivariate"][validation], dtype=np.float64)
        uv = np.asarray(arrays["teacher_univariate"][validation], dtype=np.float64)
        normalized_target = (target - mean) / scale
        normalized_mv = (mv - mean[..., None]) / scale[..., None]
        normalized_uv = (uv - mean[..., None]) / scale[..., None]
        response_raw = mv - uv
        response = normalized_mv - normalized_uv
        mv_pinball_horizon = _pinball_per_window_horizon(normalized_mv, normalized_target)
        uv_pinball_horizon = _pinball_per_window_horizon(normalized_uv, normalized_target)
        mv_mae_horizon = _median_mae_per_window_horizon(normalized_mv, normalized_target)
        uv_mae_horizon = _median_mae_per_window_horizon(normalized_uv, normalized_target)
        mv_pinball = mv_pinball_horizon.mean(axis=1)
        uv_pinball = uv_pinball_horizon.mean(axis=1)
        mv_mae = mv_mae_horizon.mean(axis=1)
        uv_mae = uv_mae_horizon.mean(axis=1)
        record: dict[str, Any] = {
            "dataset": name,
            "domain": item["domain"],
            "actual_variates": int(mv.shape[1]),
            "context_length": context_length,
            "horizon": horizon,
            "available_validation_windows": len(splits["validation"]),
            "analyzed_validation_windows": len(validation),
            "split": split_report,
            "teacher": _teacher_summary(
                response_raw, response, mv_pinball, uv_pinball, mv_mae, uv_mae
            ),
        }
        student_response = None
        if args.student_checkpoint is not None:
            student_mv, student_uv = _student_predictions(
                args.student_checkpoint,
                config["student"],
                contexts,
                horizon,
                args.student_batch_size,
            )
            student_response = (student_mv.astype(np.float64) - student_uv) / scale[..., None]
            record["student"] = _student_summary(student_response, response)
        records.append(record)
        values = {
            "response_raw": response_raw.astype(np.float32),
            "response": response.astype(np.float32),
            "mv_pinball": mv_pinball,
            "uv_pinball": uv_pinball,
            "mv_mae": mv_mae,
            "uv_mae": uv_mae,
        }
        if student_response is not None:
            values["student_response"] = student_response.astype(np.float32)
        for key, value in values.items():
            aggregate[key].append(value)
            by_domain[str(item["domain"])][key].append(value)
        for step in range(horizon):
            by_horizon[step + 1]["response"].append(response[:, :, step : step + 1, :])
            by_horizon[step + 1]["mv_pinball"].append(mv_pinball_horizon[:, step])
            by_horizon[step + 1]["uv_pinball"].append(uv_pinball_horizon[:, step])
            by_horizon[step + 1]["mv_mae"].append(mv_mae_horizon[:, step])
            by_horizon[step + 1]["uv_mae"].append(uv_mae_horizon[:, step])
            if student_response is not None:
                by_horizon[step + 1]["student_response"].append(
                    student_response[:, :, step : step + 1, :]
                )
        for quantile_index in range(9):
            by_quantile[quantile_index]["response"].append(
                response[..., quantile_index : quantile_index + 1]
            )
            if student_response is not None:
                by_quantile[quantile_index]["student_response"].append(
                    student_response[..., quantile_index : quantile_index + 1]
                )
        print(f"diagnosed {name}: {len(validation)} validation windows", flush=True)

    def combined_summary(parts: dict[str, list[np.ndarray]]) -> dict[str, Any]:
        response = np.concatenate([value.reshape(-1) for value in parts["response"]])
        signs = np.where(
            response > NEAR_ZERO_THRESHOLD,
            1,
            np.where(response < -NEAR_ZERO_THRESHOLD, -1, 0),
        )
        result: dict[str, Any] = {
            "absolute_response_normalized": _distribution(np.abs(response)),
            "near_zero_fraction": float(np.mean(np.abs(response) <= NEAR_ZERO_THRESHOLD)),
            "sign_fraction": {
                "negative": float(np.mean(signs == -1)),
                "near_zero": float(np.mean(signs == 0)),
                "positive": float(np.mean(signs == 1)),
            },
        }
        if "response_raw" in parts:
            result["absolute_response_raw"] = _distribution(
                np.abs(np.concatenate([value.reshape(-1) for value in parts["response_raw"]]))
            )
        if "mv_pinball" in parts:
            mv_pinball = np.concatenate(parts["mv_pinball"])
            uv_pinball = np.concatenate(parts["uv_pinball"])
            mv_mae = np.concatenate(parts["mv_mae"])
            uv_mae = np.concatenate(parts["uv_mae"])
            strength_parts = [np.mean(np.abs(value), axis=(1, 2, 3)) for value in parts["response"]]
            strength = np.concatenate(strength_parts)
            result["teacher_accuracy"] = {
                "mv_normalized_pinball_mean": float(mv_pinball.mean()),
                "uv_normalized_pinball_mean": float(uv_pinball.mean()),
                "pinball_gain_uv_minus_mv_mean": float((uv_pinball - mv_pinball).mean()),
                "mv_better_pinball_fraction": float(np.mean(uv_pinball > mv_pinball)),
                "mv_normalized_median_mae_mean": float(mv_mae.mean()),
                "uv_normalized_median_mae_mean": float(uv_mae.mean()),
                "median_mae_gain_uv_minus_mv_mean": float((uv_mae - mv_mae).mean()),
                "mv_better_median_mae_fraction": float(np.mean(uv_mae > mv_mae)),
            }
            result["response_strength_vs_teacher_gain"] = {
                "normalized_pinball": _correlations(strength, uv_pinball - mv_pinball),
                "normalized_median_mae": _correlations(strength, uv_mae - mv_mae),
            }
        if "student_response" in parts:
            student_part_values = parts["student_response"]
            student = np.concatenate([value.reshape(-1) for value in student_part_values])
            student_result = _student_summary(student[None, :], response[None, :])
            window_cosines = []
            for student_part, teacher_part in zip(
                student_part_values, parts["response"], strict=True
            ):
                student_vectors = student_part.reshape(len(student_part), -1)
                teacher_vectors = teacher_part.reshape(len(teacher_part), -1)
                denominator = np.linalg.norm(student_vectors, axis=1) * np.linalg.norm(
                    teacher_vectors, axis=1
                )
                valid = denominator > 0
                window_cosines.append(
                    np.sum(student_vectors[valid] * teacher_vectors[valid], axis=1)
                    / denominator[valid]
                )
            valid_window_cosines = np.concatenate(window_cosines)
            student_result["directional_cosine_mean_over_windows"] = (
                float(valid_window_cosines.mean()) if len(valid_window_cosines) else None
            )
            result["student"] = student_result
        return result

    result = {
        "status": "succeeded",
        "methodology": {
            "scope": "actual V>1 production examples only; no reconstructed cross-series groups",
            "partition": training["validation_split"],
            "validation_fraction": float(training["validation_fraction"]),
            "maximum_validation_windows_per_dataset": args.maximum_validation_windows_per_dataset,
            "selection": "deterministic uniform sample without replacement within validation",
            "spearman_element_cap": MAXIMUM_SPEARMAN_ELEMENTS,
            "normalization_epsilon": epsilon,
            "near_zero_absolute_threshold_normalized": NEAR_ZERO_THRESHOLD,
            "accuracy_gain_sign": "positive means the teacher MV view has lower error than UV",
            "student_checkpoint": (
                str(args.student_checkpoint.resolve()) if args.student_checkpoint else None
            ),
            "student_label": args.student_label,
        },
        "summary": combined_summary(aggregate),
        "by_domain": {
            domain: combined_summary(parts) for domain, parts in sorted(by_domain.items())
        },
        "by_dataset": records,
        "by_horizon_step": {
            str(step): combined_summary(parts) for step, parts in sorted(by_horizon.items())
        },
        "by_quantile": {
            f"{QUANTILES[index]:.1f}": combined_summary(parts)
            for index, parts in sorted(by_quantile.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
