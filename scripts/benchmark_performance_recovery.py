#!/usr/bin/env python3
"""Run one isolated trial of the frozen performance-recovery inference benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from timesfm_lab.config import load_config
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = tuple(value / 10 for value in range(1, 10))
BenchmarkFunctions = tuple[
    Callable[[], Any],
    Callable[[], np.ndarray],
    Callable[[], None],
    Callable[[], None],
    Callable[[], np.ndarray] | None,
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_array_digest(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())


def _load_workloads(
    config: dict[str, Any], data_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    from gift_eval.data import Dataset

    scope = config["scope"]
    authority_path = ROOT / scope["target_authority"]
    authority = yaml.safe_load(authority_path.read_text())
    if not isinstance(authority, dict):
        raise ValueError("performance-recovery target authority must be a mapping")
    target_speed = authority["speed"]
    target_execution = target_speed["execution"]
    target_input = target_speed["input_contract"]
    expected_authority_values = {
        "gating_statistic": "p50",
        "aggregate": "equal_weight_geometric_mean_of_paired_per_shape_speedups",
        "gating_teacher_reference": "optimized",
        "hardware_execution": "same_single_gpu_for_teacher_and_student",
        "concurrency": 1,
        "run_order": "serial_interleaved",
        "output_quantiles": 9,
        "selection": "first_eligible_indices_in_dataset_order",
    }
    actual_authority_values = {
        "gating_statistic": str(target_speed["gating_statistic"]),
        "aggregate": str(target_speed["aggregate"]),
        "gating_teacher_reference": str(target_speed["gating_teacher_reference"]),
        "hardware_execution": str(target_execution["physical_gpu"]),
        "concurrency": int(target_execution["concurrency"]),
        "run_order": str(target_execution["run_order"]),
        "output_quantiles": int(target_execution["output_quantiles"]),
        "selection": str(target_input["selection"]),
    }
    if actual_authority_values != expected_authority_values:
        raise ValueError(
            "unexpected target-authority speed protocol: "
            f"{actual_authority_values} != {expected_authority_values}"
        )
    if not bool(target_input["include_target_history"]):
        raise ValueError("target history must be included")
    if not bool(target_input["include_past_dynamic_real_when_present"]):
        raise ValueError("available past dynamic covariates must be included")
    if not bool(target_input["forbid_future_target_values"]):
        raise ValueError("future target values must be forbidden")
    if str(authority["revisions"]["teacher"]["revision"]) != config["model_revision"]:
        raise ValueError("teacher revision disagrees with the target authority")
    if str(authority["revisions"]["gift_eval"]["revision"]) != config["dataset_revision"]:
        raise ValueError("GIFT-Eval revision disagrees with the target authority")
    authority_sha256 = _sha256_file(authority_path)
    expected_authority_sha256 = str(scope["target_authority_sha256"])
    if authority_sha256 != expected_authority_sha256:
        raise ValueError(
            "target authority changed after the systems workload was frozen: "
            f"{authority_sha256} != {expected_authority_sha256}"
        )
    authority_contract = {
        "shared_context_limit": int(target_input["context_cap"]),
        "warmup_iterations": int(target_execution["warmup_calls_per_shape"]),
        "steady_state_repetitions": int(target_execution["measured_calls_per_shape"]),
        "target_end_to_end_speedup": float(target_speed["minimum_end_to_end_speedup"]),
        "required_gpu_name": str(target_speed["hardware"]),
    }
    systems_contract = {
        "shared_context_limit": int(scope["shared_context_limit"]),
        "warmup_iterations": int(config["measurement"]["warmup_iterations"]),
        "steady_state_repetitions": int(
            config["measurement"]["steady_state_repetitions"]
        ),
        "target_end_to_end_speedup": float(
            config["measurement"]["target_end_to_end_speedup"]
        ),
        "required_gpu_name": str(config["measurement"]["required_gpu_name"]),
    }
    if systems_contract != authority_contract:
        raise ValueError(
            f"systems/target authority contract mismatch: {systems_contract} "
            f"!= {authority_contract}"
        )
    authority_workloads = []
    for item in target_speed["workloads"]:
        dataset, term = str(item["configuration"]).rsplit("/", 1)
        authority_workloads.append(
            {
                "name": str(item["name"]),
                "dataset": dataset,
                "term": term,
                "batch": int(item["batch"]),
                "variates": int(item["variates"]),
                "context": int(item["context"]),
                "horizon": int(item["horizon"]),
            }
        )
    systems_workloads = [
        {
            field: (str(item[field]) if field in {"name", "dataset", "term"} else int(item[field]))
            for field in ("name", "dataset", "term", "batch", "variates", "context", "horizon")
        }
        for item in config["workloads"]
    ]
    if systems_workloads != authority_workloads:
        raise ValueError("systems workload matrix diverges from the target authority")
    quantiles = tuple(float(value) for value in scope["quantiles"])
    if quantiles != QUANTILES:
        raise ValueError("the benchmark requires quantiles 0.1 through 0.9")
    if scope["input_policy"] != "target_and_available_past_dynamic_real":
        raise ValueError("unexpected frozen input policy")
    if scope["covariate_policy"] != "include_past_dynamic_real_when_present":
        raise ValueError("the primary benchmark must include available past covariates")
    for field in ("use_symmetric_averaging", "make_positive", "sort_quantiles"):
        if config["teacher"]["inference"][field] != config["student"]["inference"][field]:
            raise ValueError(f"teacher/student deployment mismatch for {field}")

    scope55 = load_config(ROOT / scope["source_configurations"])
    scope19 = load_config(ROOT / scope["multivariate_subset"])
    configurations55 = {
        (str(item["name"]), str(item["term"]))
        for item in scope55["evaluation"]["datasets"]
    }
    configurations19 = {
        (str(item["name"]), str(item["term"]))
        for item in scope19["evaluation"]["datasets"]
    }
    if scope55["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("55-configuration scope and systems dataset revisions disagree")
    if scope19["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("19-configuration scope and systems dataset revisions disagree")

    seen_names: set[str] = set()
    loaded: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    suite_digest = hashlib.sha256()
    shared_context_limit = int(scope["shared_context_limit"])
    for raw_spec in config["workloads"]:
        spec = dict(raw_spec)
        name = str(spec["name"])
        if name in seen_names:
            raise ValueError(f"duplicate workload name: {name}")
        seen_names.add(name)
        key = (str(spec["dataset"]), str(spec["term"]))
        if key not in configurations55:
            raise ValueError(f"{name} is outside the frozen 55-configuration scope")
        groups = tuple(str(value) for value in spec["groups"])
        if "short55" not in groups:
            raise ValueError(f"{name} must belong to short55")
        if ("mv19" in groups) != (key in configurations19):
            raise ValueError(f"{name} has an incorrect mv19 group assignment")
        indices = [int(value) for value in spec["instance_indices"]]
        batch = int(spec["batch"])
        variates = int(spec["variates"])
        context_length = int(spec["context"])
        horizon = int(spec["horizon"])
        if len(indices) != batch or len(set(indices)) != batch or min(indices) < 0:
            raise ValueError(f"{name} requires exactly {batch} distinct nonnegative indices")
        if context_length > shared_context_limit:
            raise ValueError(f"{name} exceeds the shared context limit")

        dataset = Dataset(name=key[0], term=key[1], to_univariate=False)
        if int(dataset.target_dim) != variates:
            raise ValueError(
                f"{name} expected {variates} variates, dataset exposes {dataset.target_dim}"
            )
        if int(dataset.prediction_length) != horizon:
            raise ValueError(
                f"{name} expected horizon {horizon}, dataset requires {dataset.prediction_length}"
            )
        wanted = set(indices)
        selected: dict[int, dict[str, Any]] = {}
        for index, entry in enumerate(dataset.test_data.input):
            if index in wanted:
                selected[index] = entry
            if len(selected) == len(wanted):
                break
        missing_indices = sorted(wanted.difference(selected))
        if missing_indices:
            raise ValueError(f"{name} is missing input indices {missing_indices}")

        contexts: list[np.ndarray] = []
        source_lengths: list[int] = []
        effective_context_lengths: list[int] = []
        past_covariates: list[np.ndarray | None] = []
        workload_digest = hashlib.sha256()
        for index in indices:
            entry = selected[index]
            target = np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
            if target.shape[0] != variates:
                raise ValueError(
                    f"{name}[{index}] has target shape {target.shape}, requires {variates} rows"
                )
            context = np.ascontiguousarray(target[:, -context_length:])
            contexts.append(context)
            source_lengths.append(int(target.shape[-1]))
            effective_context_lengths.append(int(context.shape[-1]))
            raw_covariates = entry.get("past_feat_dynamic_real")
            if raw_covariates is None:
                covariates = None
                workload_digest.update(b"no-past-covariates")
            else:
                full_covariates = np.atleast_2d(
                    np.asarray(raw_covariates, dtype=np.float32)
                )
                if full_covariates.shape[-1] != target.shape[-1]:
                    raise ValueError(
                        f"{name}[{index}] target/covariate history lengths disagree: "
                        f"{target.shape[-1]} vs {full_covariates.shape[-1]}"
                    )
                covariates = np.ascontiguousarray(
                    full_covariates[:, -context.shape[-1] :]
                )
                workload_digest.update(b"past-covariates")
                _update_array_digest(workload_digest, covariates)
            past_covariates.append(covariates)
            workload_digest.update(str(index).encode())
            _update_array_digest(workload_digest, context)

        covariate_presence = [value is not None for value in past_covariates]
        if any(covariate_presence) and not all(covariate_presence):
            raise ValueError(f"{name} mixes requests with and without past covariates")
        input_sha256 = workload_digest.hexdigest()
        suite_digest.update(name.encode())
        suite_digest.update(input_sha256.encode())
        missing_values = sum(
            int(np.size(context) - np.isfinite(context).sum()) for context in contexts
        )
        included_covariates = [value for value in past_covariates if value is not None]
        row = {
            "name": name,
            "groups": list(groups),
            "dataset": key[0],
            "term": key[1],
            "instance_indices": indices,
            "batch": batch,
            "variates": variates,
            "context": context_length,
            "horizon": horizon,
            "source_lengths": source_lengths,
            "effective_context_lengths": effective_context_lengths,
            "missing_values": missing_values,
            "input_values": sum(int(context.size) for context in contexts),
            "past_covariates_included": bool(included_covariates),
            "past_covariate_variates": (
                0 if not included_covariates else int(included_covariates[0].shape[0])
            ),
            "past_covariate_missing_values": (
                sum(int(np.size(value) - np.isfinite(value).sum()) for value in included_covariates)
            ),
            "input_sha256": input_sha256,
        }
        loaded.append(
            {
                "spec": row,
                "contexts": contexts,
                "past_covariates": past_covariates,
            }
        )
        manifest_rows.append(row)

    manifest = {
        "status": "succeeded",
        "protocol": str(config["run_id"]),
        "target_authority": str(scope["target_authority"]),
        "target_authority_sha256": authority_sha256,
        "dataset_revision": str(config["dataset_revision"]),
        "scope": str(scope["name"]),
        "context_policy": str(scope["context_policy"]),
        "input_policy": str(scope["input_policy"]),
        "covariate_policy": str(scope["covariate_policy"]),
        "workload_count": len(manifest_rows),
        "suite_input_sha256": suite_digest.hexdigest(),
        "workloads": manifest_rows,
    }
    expected_suite_sha256 = str(scope["suite_input_sha256"])
    if manifest["suite_input_sha256"] != expected_suite_sha256:
        raise ValueError(
            "real-data workload changed after it was frozen: "
            f"{manifest['suite_input_sha256']} != {expected_suite_sha256}"
        )
    return loaded, manifest


def _gpu_identity(physical_index: int) -> dict[str, Any]:
    query = "index,uuid,name,driver_version,memory.total"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [value.strip() for value in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi identity output: {result.stdout!r}")
    return {
        "physical_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": int(fields[4]),
    }


def _gpu_state(physical_index: int) -> dict[str, float]:
    query = "temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [value.strip() for value in result.stdout.strip().split(",")]
    return {
        "temperature_c": float(fields[0]),
        "pstate": float(fields[1].lstrip("P")),
        "sm_clock_mhz": float(fields[2]),
        "memory_clock_mhz": float(fields[3]),
        "power_watts": float(fields[4]),
    }


def _timed_call(function: Callable[[], Any], torch: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def _distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "samples": samples,
    }


def _throughput(samples: list[float], spec: dict[str, Any]) -> dict[str, float]:
    mean_seconds = float(np.mean(samples)) / 1000.0
    batch = int(spec["batch"])
    variates = int(spec["variates"])
    horizon = int(spec["horizon"])
    return {
        "requests_per_second": batch / mean_seconds,
        "series_per_second": batch * variates / mean_seconds,
        "forecast_points_per_second": batch * variates * horizon / mean_seconds,
    }


def _memory_snapshot(torch: Any, baseline: dict[str, int]) -> dict[str, int]:
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "baseline_allocated": baseline["allocated"],
        "baseline_reserved": baseline["reserved"],
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "incremental_peak_allocated": max(0, peak_allocated - baseline["allocated"]),
        "incremental_peak_reserved": max(0, peak_reserved - baseline["reserved"]),
    }


def _teacher_functions(
    config: dict[str, Any],
    workload: dict[str, Any],
    evaluator: Any,
    torch: Any,
    reference_kind: str,
) -> BenchmarkFunctions:
    from timesfm3.evaluator import _MAX_VARIATES_PER_FORWARD
    from timesfm3.timesfm3_forecaster import _is_nonnegative, _Query, linear_interpolation

    inference = config["teacher"]["inference"]
    if not inference["use_symmetric_averaging"]:
        raise ValueError("the frozen teacher deployment requires symmetric averaging")
    if inference["use_znorm"] or inference["padding_mode"] != "none":
        raise ValueError("unexpected frozen teacher normalization or padding mode")
    contexts = workload["contexts"]
    past_covariates = workload["past_covariates"]
    horizon = int(workload["spec"]["horizon"])
    variates = int(workload["spec"]["variates"])
    global_horizon = (
        math.ceil(horizon / evaluator.config.output_patch_length)
        * evaluator.config.output_patch_length
    )
    if reference_kind not in {"teacher_stock", "teacher_optimized"}:
        raise ValueError(f"unsupported teacher reference: {reference_kind}")

    def prepare_plans() -> list[dict[str, Any]]:
        covariate_count = (
            0
            if past_covariates[0] is None
            else int(np.asarray(past_covariates[0]).shape[0])
        )
        selected_covariates = past_covariates
        if variates + covariate_count > _MAX_VARIATES_PER_FORWARD:
            retained_covariates = min(covariate_count, _MAX_VARIATES_PER_FORWARD - 1)
            if covariate_count > retained_covariates:
                indices = np.sort(
                    np.random.default_rng(42).choice(
                        covariate_count, retained_covariates, replace=False
                    )
                )
                selected_covariates = [
                    None if value is None else value[indices] for value in past_covariates
                ]
                covariate_count = retained_covariates
            targets_per_chunk = _MAX_VARIATES_PER_FORWARD - covariate_count
        else:
            targets_per_chunk = variates

        chunk_plans: list[dict[str, Any]] = []
        per_core_batch = int(evaluator.config.per_core_batch_size)
        for variate_start in range(0, variates, targets_per_chunk):
            variate_end = min(variate_start + targets_per_chunk, variates)
            actual_chunk_size = variate_end - variate_start
            chunk_contexts = [value[variate_start:variate_end] for value in contexts]
            if actual_chunk_size < targets_per_chunk:
                padded = []
                pad_needed = targets_per_chunk - actual_chunk_size
                for full_context, chunk_context in zip(
                    contexts, chunk_contexts, strict=True
                ):
                    repeats = math.ceil(pad_needed / full_context.shape[0])
                    extra = np.tile(full_context, (repeats, 1))[:pad_needed]
                    padded.append(np.concatenate((chunk_context, extra), axis=0))
                chunk_contexts = padded

            cleaned_contexts: list[np.ndarray] = []
            cleaned_covariates: list[np.ndarray | None] = []
            for raw_context, raw_covariates in zip(
                chunk_contexts, selected_covariates, strict=True
            ):
                target = np.atleast_2d(np.array(raw_context, dtype=np.float32))
                covariates = (
                    None
                    if raw_covariates is None
                    else np.atleast_2d(np.array(raw_covariates, dtype=np.float32))
                )
                all_nan = np.isnan(target).all(axis=0)
                first_valid = (
                    target.shape[-1] if all_nan.all() else int(np.argmax(~all_nan))
                )
                if 0 < first_valid < target.shape[-1]:
                    target = target[:, first_valid:]
                    if covariates is not None:
                        covariates = covariates[:, first_valid:]
                elif first_valid == target.shape[-1] and target.shape[-1] > 0:
                    target = np.zeros_like(target)
                cleaned_contexts.append(np.atleast_2d(linear_interpolation(target)))
                cleaned_covariates.append(
                    None
                    if covariates is None
                    else np.atleast_2d(linear_interpolation(covariates))
                )

            symmetric_contexts = [
                value for context in cleaned_contexts for value in (context, -context)
            ]
            symmetric_covariates = [
                value
                for covariates in cleaned_covariates
                for value in (covariates, None if covariates is None else -covariates)
            ]
            queries = [
                _Query(
                    horizon=global_horizon,
                    targets=context,
                    past_only_covariates=covariates,
                )
                for context, covariates in zip(
                    symmetric_contexts, symmetric_covariates, strict=True
                )
            ]
            resident_batches: list[tuple[Any, Any, Any | None, int]] = []
            for start in range(0, len(queries), per_core_batch):
                query_batch = queries[start : start + per_core_batch]
                batch_context = min(
                    math.ceil(max(query.context_length for query in query_batch) / 32)
                    * 32,
                    int(evaluator.global_context),
                )
                batch_context = max(batch_context, 32)
                formatted = [query.format(batch_context) for query in query_batch]
                horizons, targets, masks, formatted_covariates, _ = tuple(
                    list(values) for values in zip(*formatted, strict=True)
                )
                resident_covariates = (
                    None
                    if not any(value is not None for value in formatted_covariates)
                    else torch.from_numpy(np.stack(formatted_covariates)).to(
                        evaluator.device, dtype=torch.float32
                    )
                )
                resident_batches.append(
                    (
                        torch.from_numpy(np.stack(targets)).to(
                            evaluator.device, dtype=torch.float32
                        ),
                        torch.from_numpy(np.stack(masks)).to(
                            evaluator.device, dtype=torch.bool
                        ),
                        resident_covariates,
                        int(horizons[0]),
                    )
                )
            nonnegative = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(_is_nonnegative(context), dtype=bool)
                        for context in chunk_contexts
                    ]
                )
            ).to(evaluator.device)
            chunk_plans.append(
                {
                    "actual_chunk_size": actual_chunk_size,
                    "target_slots": targets_per_chunk,
                    "resident_batches": resident_batches,
                    "nonnegative": nonnegative,
                }
            )
        return chunk_plans

    def execute(chunk_plans: list[dict[str, Any]]) -> Any:
        chunk_outputs = []
        with torch.inference_mode():
            for plan in chunk_plans:
                outputs = []
                for target, mask, covariates, decode_horizon in plan["resident_batches"]:
                    outputs.append(
                        evaluator.model.decode(
                            target=target,
                            horizon=decode_horizon,
                            past_only_covariates=covariates,
                            mask=mask,
                        )
                    )
                raw = torch.cat(outputs, dim=0)[:, : plan["target_slots"]]
                raw = torch.sort(raw, dim=-1).values
                raw = (raw[0::2] - raw[1::2].flip(-1)) / 2
                if inference["make_positive"]:
                    raw = torch.where(
                        plan["nonnegative"][..., None, None], raw.clamp_min(0), raw
                    )
                chunk_outputs.append(raw[:, : plan["actual_chunk_size"], :horizon, :])
            return torch.cat(chunk_outputs, dim=1)

    resident: dict[str, list[dict[str, Any]] | None] = {"plans": None}

    def prepare_model_only() -> None:
        resident["plans"] = prepare_plans()

    def model_only() -> Any:
        if resident["plans"] is None:
            raise RuntimeError("model-only resident inputs were not prepared")
        return execute(resident["plans"])

    def stock_end_to_end() -> np.ndarray:
        outputs = list(
            evaluator.predict_batch(
                contexts=contexts,
                horizon=horizon,
                past_only_covariates=past_covariates,
                past_future_covariates=None,
                return_quantiles=True,
                use_symmetric_averaging=True,
                make_positive=True,
                sort_quantiles=True,
                use_znorm=False,
                padding_mode="none",
                univariate=False,
            )
        )
        return np.stack([np.asarray(output.quantiles) for output in outputs])

    def end_to_end() -> np.ndarray:
        if reference_kind == "teacher_stock":
            return stock_end_to_end()
        return execute(prepare_plans()).float().cpu().numpy()

    def release_model_only() -> None:
        resident["plans"] = None

    semantic_reference = stock_end_to_end if reference_kind == "teacher_optimized" else None
    return (
        model_only,
        end_to_end,
        prepare_model_only,
        release_model_only,
        semantic_reference,
    )


def _import_factory(path: str) -> Callable[[dict[str, Any]], Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("student factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"student factory is not callable: {path}")
    return factory


def _load_student(
    student_config: dict[str, Any], checkpoint: Path, factory_path: str | None, torch: Any
) -> tuple[Any, int, int]:
    if factory_path is None:
        from timesfm_lab.models import build_student

        model = build_student(student_config["student"])
    else:
        model = _import_factory(factory_path)(student_config["student"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, total_parameters, trainable_parameters


def _student_functions(
    workload: dict[str, Any], model: Any, torch: Any
) -> BenchmarkFunctions:
    from timesfm3.timesfm3_forecaster import _is_nonnegative, linear_interpolation

    raw_contexts = workload["contexts"]
    raw_covariates = workload["past_covariates"]
    batch = int(workload["spec"]["batch"])
    variates = int(workload["spec"]["variates"])
    context_length = int(workload["spec"]["context"])
    horizon = int(workload["spec"]["horizon"])

    def prepare_host() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        cleaned_contexts: list[np.ndarray] = []
        cleaned_covariates: list[np.ndarray | None] = []
        for original, original_covariates in zip(
            raw_contexts, raw_covariates, strict=True
        ):
            context = np.atleast_2d(np.array(original, dtype=np.float32))
            covariates = (
                None
                if original_covariates is None
                else np.atleast_2d(np.array(original_covariates, dtype=np.float32))
            )
            all_missing = np.isnan(context).all(axis=0)
            first_valid = (
                context.shape[-1]
                if all_missing.all()
                else int(np.argmax(~all_missing))
            )
            if 0 < first_valid < context.shape[-1]:
                context = context[:, first_valid:]
                if covariates is not None:
                    covariates = covariates[:, first_valid:]
            elif first_valid == context.shape[-1] and context.shape[-1] > 0:
                context = np.zeros_like(context)
            context = np.atleast_2d(linear_interpolation(context))[:, -context_length:]
            if covariates is not None:
                covariates = np.atleast_2d(linear_interpolation(covariates))[
                    :, -context.shape[-1] :
                ]
            cleaned_contexts.append(context)
            cleaned_covariates.append(covariates)

        packed_length = max(context.shape[-1] for context in cleaned_contexts)
        context_values = np.zeros((batch, variates, packed_length), dtype=np.float32)
        context_observed = np.zeros_like(context_values, dtype=np.bool_)
        for index, context in enumerate(cleaned_contexts):
            length = context.shape[-1]
            context_values[index, :, -length:] = context
            context_observed[index, :, -length:] = np.isfinite(context)

        covariate_count = max(
            (value.shape[0] for value in cleaned_covariates if value is not None),
            default=0,
        )
        if covariate_count == 0:
            covariate_values = None
            covariate_observed = None
        else:
            covariate_values = np.zeros(
                (batch, covariate_count, packed_length), dtype=np.float32
            )
            covariate_observed = np.zeros_like(covariate_values, dtype=np.bool_)
            for index, covariates in enumerate(cleaned_covariates):
                if covariates is None:
                    continue
                length = covariates.shape[-1]
                covariate_values[index, : covariates.shape[0], -length:] = covariates
                covariate_observed[index, : covariates.shape[0], -length:] = np.isfinite(
                    covariates
                )
        return context_values, context_observed, covariate_values, covariate_observed

    def to_device() -> tuple[Any, Any, Any | None, Any | None, Any]:
        context, observed, covariates, covariate_observed = prepare_host()
        nonnegative = np.stack(
            [np.asarray(_is_nonnegative(value), dtype=bool) for value in raw_contexts]
        )
        return (
            torch.from_numpy(context).to("cuda:0"),
            torch.from_numpy(observed).to("cuda:0"),
            None if covariates is None else torch.from_numpy(covariates).to("cuda:0"),
            (
                None
                if covariate_observed is None
                else torch.from_numpy(covariate_observed).to("cuda:0")
            ),
            torch.from_numpy(nonnegative).to("cuda:0"),
        )

    def forecast(
        context: Any,
        observed: Any,
        covariates: Any | None,
        covariate_observed: Any | None,
        nonnegative: Any,
    ) -> Any:
        maximum_variates = 32
        if covariates is not None and covariates.shape[1] > maximum_variates - 1:
            indices = np.sort(
                np.random.default_rng(42).choice(
                    covariates.shape[1], maximum_variates - 1, replace=False
                )
            )
            index = torch.as_tensor(indices, device=covariates.device)
            covariates = covariates.index_select(1, index)
            if covariate_observed is None:
                raise RuntimeError("covariate mask is absent")
            covariate_observed = covariate_observed.index_select(1, index)
        covariate_count = 0 if covariates is None else int(covariates.shape[1])
        targets_per_chunk = maximum_variates - covariate_count
        if targets_per_chunk < 1:
            raise RuntimeError("past covariates leave no target slot")

        outputs = []
        target_count = context.shape[1]
        for start in range(0, target_count, targets_per_chunk):
            stop = min(start + targets_per_chunk, target_count)
            chunk = context[:, start:stop]
            chunk_observed = observed[:, start:stop]
            actual_count = stop - start
            if actual_count < targets_per_chunk:
                needed = targets_per_chunk - actual_count
                repeats = (needed + target_count - 1) // target_count
                chunk = torch.cat(
                    (chunk, context.repeat(1, repeats, 1)[:, :needed]), dim=1
                )
                chunk_observed = torch.cat(
                    (
                        chunk_observed,
                        observed.repeat(1, repeats, 1)[:, :needed],
                    ),
                    dim=1,
                )
            positive = model(
                chunk,
                horizon,
                observed_mask=chunk_observed,
                past_only_covariates=covariates,
                past_only_observed_mask=covariate_observed,
            )
            negative = model(
                -chunk,
                horizon,
                observed_mask=chunk_observed,
                past_only_covariates=None if covariates is None else -covariates,
                past_only_observed_mask=covariate_observed,
            )
            positive = torch.sort(positive, dim=-1).values
            negative = torch.sort(negative, dim=-1).values
            outputs.append(((positive - negative.flip(-1)) / 2)[:, :actual_count])
        output = torch.cat(outputs, dim=1)
        return torch.where(nonnegative[..., None, None], output.clamp_min(0), output)

    resident: dict[str, tuple[Any, Any, Any | None, Any | None, Any] | None] = {
        "inputs": None
    }

    def prepare_model_only() -> None:
        resident["inputs"] = to_device()

    def model_only() -> Any:
        if resident["inputs"] is None:
            raise RuntimeError("model-only resident inputs were not prepared")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return forecast(*resident["inputs"])

    def end_to_end() -> np.ndarray:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = forecast(*to_device())
        return output.float().cpu().numpy()

    def release_model_only() -> None:
        resident["inputs"] = None

    return model_only, end_to_end, prepare_model_only, release_model_only, None


def _run_workload(
    config: dict[str, Any],
    workload: dict[str, Any],
    model_only: Callable[[], Any],
    end_to_end: Callable[[], np.ndarray],
    prepare_model_only: Callable[[], None],
    release_model_only: Callable[[], None],
    semantic_reference: Callable[[], np.ndarray] | None,
    torch: Any,
) -> dict[str, Any]:
    measurement = config["measurement"]
    warmup = int(measurement["warmup_iterations"])
    repeats = int(measurement["steady_state_repetitions"])
    atol = float(measurement["correctness"]["model_only_vs_end_to_end_atol"])
    rtol = float(measurement["correctness"]["model_only_vs_end_to_end_rtol"])
    spec = workload["spec"]

    cold_output, first_end_to_end_ms = _timed_call(end_to_end, torch)
    prepare_model_only()
    first_model_output, first_model_only_ms = _timed_call(model_only, torch)
    for _ in range(warmup):
        model_only()
        end_to_end()
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    memory_warmup = model_only()
    torch.cuda.synchronize()
    del memory_warmup
    gc.collect()
    baseline_model = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()
    model_samples: list[float] = []
    model_output = first_model_output
    for _ in range(repeats):
        model_output, latency_ms = _timed_call(model_only, torch)
        model_samples.append(latency_ms)
    model_memory = _memory_snapshot(torch, baseline_model)
    model_numpy = model_output.float().cpu().numpy()
    del model_output, first_model_output
    release_model_only()
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    memory_warmup = end_to_end()
    torch.cuda.synchronize()
    del memory_warmup
    gc.collect()
    baseline_end_to_end = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()
    end_to_end_samples: list[float] = []
    output = cold_output
    for _ in range(repeats):
        output, latency_ms = _timed_call(end_to_end, torch)
        end_to_end_samples.append(latency_ms)
    end_to_end_memory = _memory_snapshot(torch, baseline_end_to_end)

    expected_shape = (
        int(spec["batch"]),
        int(spec["variates"]),
        int(spec["horizon"]),
        9,
    )
    if output.shape != expected_shape or model_numpy.shape != expected_shape:
        raise RuntimeError(
            f"{spec['name']} expected {expected_shape}, got "
            f"model-only={model_numpy.shape}, end-to-end={output.shape}"
        )
    if not np.isfinite(output).all() or not np.isfinite(model_numpy).all():
        raise RuntimeError(f"{spec['name']} produced non-finite forecasts")
    if not (np.diff(output, axis=-1) >= 0).all() or not (
        np.diff(model_numpy, axis=-1) >= 0
    ).all():
        raise RuntimeError(f"{spec['name']} produced unordered quantiles")
    maximum_absolute_difference = float(np.max(np.abs(output - model_numpy)))
    if not np.allclose(output, model_numpy, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{spec['name']} model-only/end-to-end disagreement: "
            f"max_abs={maximum_absolute_difference}"
        )
    semantic_check = None
    if semantic_reference is not None:
        reference = semantic_reference()
        semantic_difference = float(np.max(np.abs(output - reference)))
        semantic_allclose = bool(np.allclose(output, reference, atol=atol, rtol=rtol))
        if not semantic_allclose:
            raise RuntimeError(
                f"{spec['name']} optimized/stock teacher disagreement: "
                f"max_abs={semantic_difference}"
            )
        semantic_check = {
            "allclose": True,
            "atol": atol,
            "rtol": rtol,
            "maximum_absolute_difference": semantic_difference,
            "reference_output_sha256": hashlib.sha256(
                np.ascontiguousarray(reference).tobytes()
            ).hexdigest(),
        }

    return {
        **spec,
        "warmup_iterations": warmup,
        "steady_state_repetitions": repeats,
        "cold_latency_ms": {
            "first_end_to_end": first_end_to_end_ms,
            "first_model_only_after_end_to_end": first_model_only_ms,
        },
        "latency_ms": {
            "model_only": _distribution(model_samples),
            "end_to_end": _distribution(end_to_end_samples),
        },
        "throughput": {
            "model_only": _throughput(model_samples, spec),
            "end_to_end": _throughput(end_to_end_samples, spec),
        },
        "memory_bytes": {
            "model_only": model_memory,
            "end_to_end": end_to_end_memory,
        },
        "correctness": {
            "allclose": True,
            "atol": atol,
            "rtol": rtol,
            "maximum_absolute_difference": maximum_absolute_difference,
            "output_sha256": hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
        },
        "semantic_reference": semantic_check,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("validate", "teacher_stock", "teacher_optimized", "student"),
        required=True,
    )
    parser.add_argument("--student-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--student-factory")
    parser.add_argument("--label", default="reference")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config = load_config(args.config)
    workloads, manifest = _load_workloads(config, args.data_root)
    config_sha256 = _sha256_file(args.config)
    canonical_manifest_path = ROOT / str(config["scope"]["manifest_path"])
    canonical_manifest = json.loads(canonical_manifest_path.read_text())
    canonical_manifest.pop("config_path", None)
    expected_manifest = manifest | {"config_sha256": config_sha256}
    if canonical_manifest != expected_manifest:
        raise ValueError(
            "generated workload manifest disagrees with the committed frozen manifest: "
            f"{canonical_manifest_path}"
        )
    if args.model_kind == "validate":
        result = manifest | {
            "config_path": str(args.config),
            "config_sha256": config_sha256,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(args.output)
        return 0

    import torch

    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(config["measurement"]["physical_gpu_index"])
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != str(physical_gpu):
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to the single physical GPU named by "
            f"--physical-gpu-index (expected {physical_gpu}, got {visible_devices!r})"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible CUDA GPU")
    gpu_identity = _gpu_identity(physical_gpu)
    required_gpu_name = str(config["measurement"]["required_gpu_name"])
    if gpu_identity["name"] != required_gpu_name:
        raise RuntimeError(
            f"benchmark requires {required_gpu_name!r}, got {gpu_identity['name']!r}"
        )
    gpu_state_before = _gpu_state(physical_gpu)
    record = RunRecord.start(
        run_id=f"{config['run_id']}-{args.model_kind}-{args.label}-trial{args.trial}",
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )

    checkpoint_sha256: str | None = None
    student_config_sha256: str | None = None
    parameter_count: int
    trainable_parameter_count: int
    load_started = time.perf_counter()
    if args.model_kind.startswith("teacher_"):
        if args.checkpoint is not None or args.student_config is not None or args.compile:
            raise ValueError("teacher benchmark does not accept student/compile arguments")
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        teacher = config["teacher"]
        evaluator = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=str(teacher["id"]),
                revision=str(config["model_revision"]),
                device=str(teacher["device"]),
                per_core_batch_size=int(teacher["per_core_batch_size"]),
            )
        )
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        parameter_count = sum(parameter.numel() for parameter in evaluator.model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in evaluator.model.parameters()
            if parameter.requires_grad
        )

        def functions(workload: dict[str, Any]) -> BenchmarkFunctions:
            return _teacher_functions(
                config, workload, evaluator, torch, args.model_kind
            )

        precision = str(next(evaluator.model.parameters()).dtype)
        deployment_implementation = config["teacher"]["references"][
            args.model_kind.removeprefix("teacher_")
        ]
    else:
        if args.student_config is None or args.checkpoint is None:
            raise ValueError("student benchmark requires --student-config and --checkpoint")
        student_config = load_config(args.student_config)
        student_config_sha256 = _sha256_file(args.student_config)
        model, parameter_count, trainable_parameter_count = _load_student(
            student_config, args.checkpoint, args.student_factory, torch
        )
        checkpoint_sha256 = _sha256_file(args.checkpoint)
        model.to("cuda:0").eval()
        if args.compile:
            model = torch.compile(model, mode=args.compile_mode)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started

        def functions(workload: dict[str, Any]) -> BenchmarkFunctions:
            return _student_functions(workload, model, torch)

        precision = str(config["student"]["precision"])
        deployment_implementation = "native_student_matched_postprocessing"

    results: list[dict[str, Any]] = []
    for workload in workloads:
        (
            model_only,
            end_to_end,
            prepare_model_only,
            release_model_only,
            semantic_reference,
        ) = functions(workload)
        result = _run_workload(
            config,
            workload,
            model_only,
            end_to_end,
            prepare_model_only,
            release_model_only,
            semantic_reference,
            torch,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "name": result["name"],
                    "model_only_p50_ms": result["latency_ms"]["model_only"]["p50"],
                    "end_to_end_p50_ms": result["latency_ms"]["end_to_end"]["p50"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    gpu_state_after = _gpu_state(physical_gpu)
    record.extra.update(
        {
            "protocol_version": 1,
            "process_id": os.getpid(),
            "model_kind": args.model_kind,
            "label": args.label,
            "trial": args.trial,
            "config_sha256": config_sha256,
            "target_authority_sha256": manifest["target_authority_sha256"],
            "suite_input_sha256": manifest["suite_input_sha256"],
            "model_load_seconds": model_load_seconds,
            "checkpoint_path": None if args.checkpoint is None else str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "student_config_sha256": student_config_sha256,
            "student_factory": args.student_factory,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "precision": precision,
            "deployment_implementation": deployment_implementation,
            "gpu": gpu_identity,
            "gpu_state": {"before": gpu_state_before, "after": gpu_state_after},
            "timing_boundaries": {
                "model_only": config["measurement"]["model_only_boundary"],
                "end_to_end": config["measurement"]["end_to_end_boundary"],
                "synchronization": config["measurement"]["synchronization"],
            },
            "input_policy": config["scope"]["input_policy"],
            "covariate_policy": config["scope"]["covariate_policy"],
            "results": results,
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
        }
    )
    record.succeed(
        {
            f"{result['name']}/{scope}/latency_p50_ms": result["latency_ms"][scope]["p50"]
            for result in results
            for scope in ("model_only", "end_to_end")
        }
    )
    record.write(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
