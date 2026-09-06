#!/usr/bin/env python3
"""Measure a compact student's development error with past covariates on versus off.

This diagnostic is deliberately separate from GIFT-Eval and teacher supervision. It reads
only cache window identities, reconstructs the frozen development partition, and scores both
conditions against the same real ground-truth targets. Use ``--preflight-only`` to verify the
entire identity/schema contract without loading source rows, a checkpoint, or a GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import yaml

from timesfm_lab.distill.data import split_cache_indices
from timesfm_lab.models import build_student, masked_mean_and_scale

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class _FrozenDataset:
    name: str
    context: int
    horizon: int
    target_variates: int
    covariate_variates: int
    rows: npt.NDArray[np.int64]
    ends: npt.NDArray[np.int64]
    development: npt.NDArray[np.int64]


@dataclass
class _MetricSums:
    pinball: float = 0.0
    median_absolute_error: float = 0.0
    observed_targets: int = 0

    def add(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        levels = torch.arange(1, 10, device=prediction.device, dtype=torch.float32) / 10
        error = target.unsqueeze(-1) - prediction.float()
        pinball = torch.maximum(levels * error, (levels - 1.0) * error)
        expanded = mask.unsqueeze(-1).expand_as(pinball)
        self.pinball += float(pinball.masked_select(expanded).double().sum()) / 9.0
        self.median_absolute_error += float(
            (prediction.float()[..., 4] - target).abs().masked_select(mask).double().sum()
        )
        self.observed_targets += int(mask.sum())

    def result(self) -> dict[str, float | int]:
        if self.observed_targets <= 0:
            raise ValueError("metric population contains no observed targets")
        return {
            "normalized_pinball": self.pinball / self.observed_targets,
            "normalized_median_mae": self.median_absolute_error / self.observed_targets,
            "observed_targets": self.observed_targets,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes repository: {path}") from error
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}")


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
    selected = np.asarray(indices, dtype=np.int64)
    selected_rows = np.asarray(rows)[selected].astype("<i8", copy=False)
    selected_ends = np.asarray(ends)[selected].astype("<i8", copy=False)
    order = np.lexsort((selected_ends, selected_rows))
    encoded = dataset.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<qqQ", context, horizon, len(selected)))
    pairs = np.column_stack((selected_rows[order], selected_ends[order])).astype(
        "<i8", copy=False
    )
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


def _reconstruct_partitions(
    dataset: str,
    rows: npt.NDArray[Any],
    ends: npt.NDArray[Any],
    context: int,
    horizon: int,
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, npt.NDArray[np.int64]]:
    outer_config = manifest["outer_split"]
    outer, _ = split_cache_indices(
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
    outer_embargo = np.setdiff1d(
        all_indices, np.union1d(outer_training, outer_validation)
    )

    nested_config = manifest["nested_split"]
    validation_rows = np.asarray(rows)[outer_validation]
    validation_ends = np.asarray(ends)[outer_validation]
    inner_mode = "held_out_series" if len(np.unique(validation_rows)) > 1 else "blocked_time"
    try:
        inner, _ = split_cache_indices(
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
    development = outer_validation[np.asarray(inner["training"], dtype=np.int64)]
    confirmation = outer_validation[np.asarray(inner["validation"], dtype=np.int64)]
    inner_embargo = np.setdiff1d(
        outer_validation, np.union1d(development, confirmation)
    )
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
        raise ValueError(f"{dataset}: frozen shape disagrees with corpus plan")
    if int(entry["cache_windows"]) != len(rows):
        raise ValueError(f"{dataset}: frozen cache count mismatch")
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
    return partitions


def _load_identities(
    cache_root: Path, item: dict[str, Any]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    name = str(item["dataset"])
    context = int(item["context"])
    horizon = int(item["horizon"])
    paths = sorted((cache_root / name).glob("shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no cache identity shards for {name}")
    rows: list[npt.NDArray[np.int64]] = []
    ends: list[npt.NDArray[np.int64]] = []
    for path in paths:
        sidecar_path = path.with_suffix(".json")
        sidecar = _load_json(sidecar_path)
        if sidecar.get("dataset") != name:
            raise ValueError(f"cache sidecar dataset mismatch: {sidecar_path}")
        if int(sidecar.get("context_length", -1)) != context:
            raise ValueError(f"cache sidecar context mismatch: {sidecar_path}")
        if int(sidecar.get("horizon", -1)) != horizon:
            raise ValueError(f"cache sidecar horizon mismatch: {sidecar_path}")
        # NPZ members are lazy. Deliberately read identity arrays only; cached
        # teacher forecast members are neither indexed nor materialized.
        with np.load(path) as shard:
            required = {"row_index", "context_end", "context_length", "horizon"}
            if not required.issubset(shard.files):
                raise ValueError(f"cache identity schema incomplete: {path}")
            shard_rows = np.asarray(shard["row_index"], dtype=np.int64)
            shard_ends = np.asarray(shard["context_end"], dtype=np.int64)
            shard_context = np.asarray(shard["context_length"])
            shard_horizon = np.asarray(shard["horizon"])
        if shard_rows.shape != shard_ends.shape or shard_rows.ndim != 1:
            raise ValueError(f"cache identity shape mismatch: {path}")
        if len(shard_rows) != int(sidecar.get("windows", -1)):
            raise ValueError(f"cache sidecar window count mismatch: {path}")
        if not np.all(shard_context == context) or not np.all(shard_horizon == horizon):
            raise ValueError(f"cache identity shape metadata mismatch: {path}")
        rows.append(shard_rows)
        ends.append(shard_ends)
    result_rows = np.concatenate(rows)
    result_ends = np.concatenate(ends)
    if len(result_rows) != int(item["requested_windows"]):
        raise ValueError(f"{name}: identity count differs from corpus plan")
    return result_rows, result_ends


def _discover_covariate_datasets(
    plan: dict[str, Any], data_root: Path
) -> dict[str, dict[str, Any]]:
    result = {}
    for item in plan["datasets"]:
        name = str(item["dataset"])
        info = _load_json(data_root / name / "dataset_info.json")
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"{name}: dataset feature schema is absent")
        if "past_feat_dynamic_real" in features:
            result[name] = item
    return result


def _schema_variates(feature: dict[str, Any], label: str) -> int:
    if feature.get("_type") != "Sequence" or not isinstance(feature.get("feature"), dict):
        raise ValueError(f"{label}: expected a sequence feature")
    nested = feature["feature"]
    if nested.get("_type") == "Sequence":
        count = feature.get("length")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{label}: nested sequence has no fixed positive variate count")
        return count
    return 1


def _preflight(config_path: Path) -> tuple[dict[str, Any], list[_FrozenDataset]]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic configuration schema")
    if config.get("protocol_id") != "timesfm3-performance-recovery-covariate-development-v1":
        raise ValueError("unexpected diagnostic protocol")
    guardrails = config["guardrails"]
    forbidden = (
        "gift_eval_access",
        "cached_teacher_output_access",
        "confirmation_partition_access",
    )
    if any(guardrails.get(key) != "forbidden" for key in forbidden):
        raise ValueError("diagnostic data-access guardrails must remain forbidden")
    if config["evaluation"].get("partition") != "development":
        raise ValueError("this diagnostic may use only the development partition")

    sources = config["sources"]
    plan_path = _root_path(sources["corpus_plan"])
    split_path = _root_path(sources["selection_split"])
    student_config_path = _root_path(sources["student_config"])
    _verify_file(plan_path, str(sources["corpus_plan_sha256"]), "corpus plan")
    _verify_file(split_path, str(sources["selection_split_sha256"]), "selection split")
    _verify_file(
        student_config_path, str(sources["student_config_sha256"]), "student configuration"
    )
    plan = _load_json(plan_path)
    manifest = _load_json(split_path)
    revision = str(config["dataset_revision"])
    if plan.get("dataset_revision") != revision:
        raise ValueError("corpus-plan dataset revision mismatch")
    if manifest.get("protocol_id") != "timesfm3-performance-recovery-v1.1":
        raise ValueError("unexpected frozen-selection protocol")
    if manifest.get("status") != "frozen_uninspected":
        raise ValueError("selection split is not frozen and uninspected")
    if manifest.get("target_accessed") is not False:
        raise ValueError("selection manifest creation accessed targets")
    if manifest.get("teacher_output_accessed") is not False:
        raise ValueError("selection manifest creation accessed teacher outputs")
    if manifest["source"].get("plan_sha256") != _sha256(plan_path):
        raise ValueError("selection manifest corpus-plan hash mismatch")
    if manifest["source"].get("dataset_revision") != revision:
        raise ValueError("selection manifest dataset revision mismatch")

    evaluation = config["evaluation"]
    if len(plan["datasets"]) != int(evaluation["expected_plan_dataset_count"]):
        raise ValueError("unexpected production corpus dataset count")
    data_root = _root_path(sources["data_root"])
    cache_root = _root_path(sources["cache_root"])
    discovered = _discover_covariate_datasets(plan, data_root)
    expected = evaluation["expected_datasets"]
    if set(discovered) != set(expected):
        raise ValueError(
            "covariate-bearing dataset discovery mismatch: "
            f"expected={sorted(expected)}, actual={sorted(discovered)}"
        )
    if len(discovered) != int(evaluation["expected_covariate_dataset_count"]):
        raise ValueError("unexpected covariate-bearing dataset count")
    manifest_entries = {str(item["dataset"]): item for item in manifest["datasets"]}
    plan_entries = {str(item["dataset"]): item for item in plan["datasets"]}
    if set(manifest_entries) != set(plan_entries):
        raise ValueError("selection manifest dataset names differ from corpus plan")

    frozen = []
    for name in sorted(discovered):
        item = discovered[name]
        features = _load_json(data_root / name / "dataset_info.json")["features"]
        expected_shape = expected[name]
        target_variates = _schema_variates(features["target"], f"{name}.target")
        covariate_variates = _schema_variates(
            features["past_feat_dynamic_real"], f"{name}.past_feat_dynamic_real"
        )
        if target_variates != int(expected_shape["target_variates"]):
            raise ValueError(f"{name}: target variate schema differs from frozen diagnostic")
        if covariate_variates != int(expected_shape["past_covariate_variates"]):
            raise ValueError(f"{name}: covariate schema differs from frozen diagnostic")
        rows, ends = _load_identities(cache_root, item)
        partitions = _reconstruct_partitions(
            name,
            rows,
            ends,
            int(item["context"]),
            int(item["horizon"]),
            manifest,
            manifest_entries[name],
        )
        frozen.append(
            _FrozenDataset(
                name=name,
                context=int(item["context"]),
                horizon=int(item["horizon"]),
                target_variates=target_variates,
                covariate_variates=covariate_variates,
                rows=rows,
                ends=ends,
                development=partitions["development"],
            )
        )
    windows = sum(len(item.development) for item in frozen)
    if windows != int(evaluation["expected_development_windows"]):
        raise ValueError(
            "frozen development window count mismatch: "
            f"expected={evaluation['expected_development_windows']}, actual={windows}"
        )
    return config, frozen


def _linear_interpolation(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32).copy()
    for row in result:
        missing = np.isnan(row)
        if not missing.any():
            continue
        valid = np.flatnonzero(~missing)
        if valid.size:
            row[missing] = np.interp(np.flatnonzero(missing), valid, row[valid])
        else:
            row[missing] = 0.0
    return result


def _prepare_history(
    target: npt.NDArray[np.float32], covariates: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    target = np.atleast_2d(np.asarray(target, dtype=np.float32)).copy()
    covariates = np.atleast_2d(np.asarray(covariates, dtype=np.float32)).copy()
    if target.shape[-1] != covariates.shape[-1]:
        raise ValueError("target and past covariate histories are not aligned")
    all_missing = np.isnan(target).all(axis=0)
    first_valid = target.shape[-1] if all_missing.all() else int(np.argmax(~all_missing))
    if 0 < first_valid < target.shape[-1]:
        target = target[:, first_valid:]
        covariates = covariates[:, first_valid:]
    elif first_valid == target.shape[-1] and target.shape[-1]:
        target = np.zeros_like(target)
    return _linear_interpolation(target), _linear_interpolation(covariates)


def _pad(
    arrays: list[npt.NDArray[np.float32]], channels: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    maximum_length = max(value.shape[-1] for value in arrays)
    values = np.zeros((len(arrays), channels, maximum_length), dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.bool_)
    for index, value in enumerate(arrays):
        if value.shape[0] != channels:
            raise ValueError("channel count changed within a dataset")
        length = value.shape[-1]
        values[index, :, -length:] = value
        observed[index, :, -length:] = np.isfinite(value)
    return values, observed


def _model_call(
    model: torch.nn.Module,
    context: torch.Tensor,
    observed: torch.Tensor,
    horizon: int,
    covariates: torch.Tensor | None,
    covariate_observed: torch.Tensor | None,
    *,
    sort_quantiles: bool,
) -> torch.Tensor:
    output = model(
        context,
        horizon,
        observed_mask=observed,
        past_only_covariates=covariates,
        past_only_observed_mask=covariate_observed,
    )
    return torch.sort(output, dim=-1).values if sort_quantiles else output


def _forecast(
    model: torch.nn.Module,
    context: torch.Tensor,
    observed: torch.Tensor,
    horizon: int,
    covariates: torch.Tensor | None,
    covariate_observed: torch.Tensor | None,
    nonnegative: torch.Tensor,
    evaluation: dict[str, Any],
) -> torch.Tensor:
    maximum_variates = int(evaluation["maximum_variates_per_forward"])
    target_count = int(context.shape[1])
    covariate_count = 0 if covariates is None else int(covariates.shape[1])
    if target_count + covariate_count > maximum_variates:
        raise ValueError(
            "declared development datasets require unsupported chunking; freeze a new "
            "diagnostic version before changing the packing semantics"
        )
    positive = _model_call(
        model,
        context,
        observed,
        horizon,
        covariates,
        covariate_observed,
        sort_quantiles=bool(evaluation["sort_quantiles"]),
    )
    if bool(evaluation["use_symmetric_averaging"]):
        negative = _model_call(
            model,
            -context,
            observed,
            horizon,
            None if covariates is None else -covariates,
            covariate_observed,
            sort_quantiles=bool(evaluation["sort_quantiles"]),
        )
        output = (positive - negative.flip(-1)) / 2
    else:
        output = positive
    if bool(evaluation["make_positive"]):
        output = torch.where(nonnegative[..., None, None], output.clamp_min(0), output)
    return output


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric aggregation requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_clean_code(config_path: Path, student_config_path: Path) -> str:
    paths = [SCRIPT, config_path, student_config_path, ROOT / "src/timesfm_lab"]
    relative = [str(path.resolve().relative_to(ROOT)) for path in paths]
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(("--", *relative))
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise ValueError("diagnostic-relevant tracked code has uncommitted changes")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    model.load_state_dict(state, strict=True)


def _delta(on: dict[str, Any], off: dict[str, Any]) -> dict[str, float]:
    result = {}
    for metric in ("normalized_pinball", "normalized_median_mae"):
        absolute = float(on[metric]) - float(off[metric])
        result[f"{metric}_on_minus_off"] = absolute
        result[f"{metric}_relative_on_minus_off"] = absolute / float(off[metric])
    return result


@torch.inference_mode()
def _evaluate(
    config: dict[str, Any],
    frozen: list[_FrozenDataset],
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    sources = config["sources"]
    evaluation = config["evaluation"]
    student_config_path = _root_path(sources["student_config"])
    student_config = _load_yaml(student_config_path)
    torch.manual_seed(42)
    model = build_student(student_config["student"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(evaluation["model_parameter_count"]):
        raise ValueError("student parameter count differs from the frozen diagnostic")
    _load_checkpoint(model, checkpoint)
    loaded_state_sha256 = _state_sha256(model)
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    overall = {"on": _MetricSums(), "off": _MetricSums()}
    by_dataset: dict[str, Any] = {}
    data_root = _root_path(sources["data_root"])
    batch_sizes = {
        int(key): int(value)
        for key, value in evaluation["batch_size_by_context"].items()
    }
    for item in frozen:
        if item.context not in batch_sizes:
            raise ValueError(f"no diagnostic batch size for context={item.context}")
        dataset = load_from_disk(str(data_root / item.name), keep_in_memory=False)
        source = {}
        development_rows = np.unique(item.rows[item.development])
        for row_index in development_rows:
            row_index = int(row_index)
            entry = dataset[row_index]
            target = np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
            raw_covariates = entry.get("past_feat_dynamic_real")
            if raw_covariates is None:
                raise ValueError(f"{item.name}[{row_index}] has no declared past covariates")
            covariates = np.atleast_2d(np.asarray(raw_covariates, dtype=np.float32))
            if target.shape != (item.target_variates, target.shape[-1]):
                raise ValueError(f"{item.name}[{row_index}] target variate count mismatch")
            if covariates.shape != (item.covariate_variates, target.shape[-1]):
                raise ValueError(f"{item.name}[{row_index}] covariate shape mismatch")
            source[row_index] = (target, covariates)

        metrics = {"on": _MetricSums(), "off": _MetricSums()}
        batch_size = batch_sizes[item.context]
        for start in range(0, len(item.development), batch_size):
            indices = item.development[start : start + batch_size]
            contexts = []
            covariate_contexts = []
            targets = []
            nonnegative = []
            for index in indices:
                row = int(item.rows[index])
                end = int(item.ends[index])
                full_target, full_covariates = source[row]
                raw_context = full_target[:, end - item.context : end]
                raw_covariates = full_covariates[:, end - item.context : end]
                target = full_target[:, end : end + item.horizon]
                if raw_context.shape[-1] != item.context or target.shape[-1] != item.horizon:
                    raise ValueError(f"{item.name}: cache window is outside the source row")
                prepared_context, prepared_covariates = _prepare_history(
                    raw_context, raw_covariates
                )
                contexts.append(prepared_context)
                covariate_contexts.append(prepared_covariates)
                targets.append(target)
                nonnegative.append(
                    np.asarray(
                        [
                            bool(finite.size and np.all(finite >= 0))
                            for values in raw_context
                            for finite in (values[np.isfinite(values)],)
                        ],
                        dtype=np.bool_,
                    )
                )
            context_values, context_masks = _pad(contexts, item.target_variates)
            covariate_values, covariate_masks = _pad(
                covariate_contexts, item.covariate_variates
            )
            target_values = np.stack(targets)
            target_mask_values = np.isfinite(target_values)

            context = torch.from_numpy(context_values).to(device)
            context_mask = torch.from_numpy(context_masks).to(device)
            covariates = torch.from_numpy(covariate_values).to(device)
            covariate_mask = torch.from_numpy(covariate_masks).to(device)
            target = torch.from_numpy(target_values).to(device)
            target_mask = torch.from_numpy(target_mask_values).to(device)
            nonnegative_tensor = torch.from_numpy(np.stack(nonnegative)).to(device)
            mean, scale, _ = masked_mean_and_scale(
                context,
                context_mask,
                epsilon=float(model.config.normalization_epsilon),
            )
            safe_target = torch.where(target_mask, target, mean)
            normalized_target = (safe_target - mean) / scale
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=evaluation["precision"] == "bfloat16",
            ):
                prediction_on = _forecast(
                    model,
                    context,
                    context_mask,
                    item.horizon,
                    covariates,
                    covariate_mask,
                    nonnegative_tensor,
                    evaluation,
                )
                prediction_off = _forecast(
                    model,
                    context,
                    context_mask,
                    item.horizon,
                    None,
                    None,
                    nonnegative_tensor,
                    evaluation,
                )
            forecast_mean = mean.unsqueeze(-1)
            forecast_scale = scale.unsqueeze(-1)
            normalized_on = (prediction_on.float() - forecast_mean) / forecast_scale
            normalized_off = (prediction_off.float() - forecast_mean) / forecast_scale
            metrics["on"].add(normalized_on, normalized_target, target_mask)
            metrics["off"].add(normalized_off, normalized_target, target_mask)
            overall["on"].add(normalized_on, normalized_target, target_mask)
            overall["off"].add(normalized_off, normalized_target, target_mask)

        on_result = metrics["on"].result()
        off_result = metrics["off"].result()
        by_dataset[item.name] = {
            "context": item.context,
            "horizon": item.horizon,
            "target_variates": item.target_variates,
            "past_covariate_variates": item.covariate_variates,
            "windows": len(item.development),
            "available_past_covariates_on": on_result,
            "available_past_covariates_off": off_result,
            "delta": _delta(on_result, off_result),
        }

    on_overall = overall["on"].result()
    off_overall = overall["off"].result()
    balanced = {}
    for condition in (
        "available_past_covariates_on",
        "available_past_covariates_off",
    ):
        balanced[condition] = {
            metric: _geometric_mean(
                [float(result[condition][metric]) for result in by_dataset.values()]
            )
            for metric in ("normalized_pinball", "normalized_median_mae")
        }
    balanced["delta"] = _delta(
        balanced["available_past_covariates_on"],
        balanced["available_past_covariates_off"],
    )
    return {
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "loaded_state_sha256": loaded_state_sha256,
        "parameter_count": parameter_count,
        "device": str(device),
        "precision": str(evaluation["precision"]),
        "micro_average": {
            "available_past_covariates_on": on_overall,
            "available_past_covariates_off": off_overall,
            "delta": _delta(on_overall, off_overall),
        },
        "balanced_geometric_mean": balanced,
        "by_dataset": by_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "configs/performance_recovery/covariate_development_diagnostic.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = _root_path(args.config)
    config, frozen = _preflight(config_path)
    preflight = {
        "protocol_id": config["protocol_id"],
        "status": "preflight_passed",
        "partition": "development",
        "dataset_count": len(frozen),
        "development_windows": sum(len(item.development) for item in frozen),
        "datasets": [
            {
                "dataset": item.name,
                "context": item.context,
                "horizon": item.horizon,
                "windows": len(item.development),
                "target_variates": item.target_variates,
                "past_covariate_variates": item.covariate_variates,
            }
            for item in frozen
        ],
        "development_targets_accessed": False,
        "confirmation_targets_accessed": False,
        "cached_teacher_outputs_accessed": False,
        "gift_eval_accessed": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if args.checkpoint is None or args.output is None:
        raise ValueError("--checkpoint and --output are required for metric evaluation")
    checkpoint = _root_path(args.checkpoint)
    output = _root_path(args.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    if config["guardrails"].get("require_clean_tracked_code") is not True:
        raise ValueError("clean tracked diagnostic code must remain required")
    student_config_path = _root_path(config["sources"]["student_config"])
    commit = _require_clean_code(config_path, student_config_path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("full diagnostic evaluation requires an available CUDA device")
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "established_at_utc": _utc_now(),
        "repository_commit": commit,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "corpus_plan_sha256": str(config["sources"]["corpus_plan_sha256"]),
        "selection_split_sha256": str(config["sources"]["selection_split_sha256"]),
        "dataset_revision": str(config["dataset_revision"]),
        "partition": "development",
        "development_targets_accessed": True,
        "confirmation_targets_accessed": False,
        "cached_teacher_outputs_accessed": False,
        "gift_eval_accessed": False,
        "selection_authority": "diagnostic_only_not_checkpoint_selection",
        "results": _evaluate(config, frozen, checkpoint, device),
    }
    _atomic_json(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
