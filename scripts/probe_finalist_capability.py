#!/usr/bin/env python3
"""Fail-closed native-multivariate capability probe for one frozen student."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import split_cache_indices
from timesfm_lab.models import build_student

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/reproduction/distillation/production-1m-corpus-plan.json"
CACHE_AUDIT = ROOT / "results/reproduction/distillation/production-1m-cache-audit.json"
SELECTION = ROOT / "results/reproduction/distillation/performance-recovery-selection-split.json"
EXPECTED = {
    "dataset_revision": "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874",
    "plan_sha256": "2cd2967fa0ef4e1f6e7a18d050d7af1b7533458abab96e52b45a4c0caff0828c",
    "cache_audit_sha256": "a471d51a48140acf66f8ccffd6920615eda4562474174595ea683b541014449e",
    "selection_sha256": "9d3e06b328b76baaab558c18717b7336961f07e81261989349c0f4e20c899cd9",
    "selection_protocol_id": "timesfm3-performance-recovery-v1.1",
    "dataset": "godaddy",
    "view_class": "true_multivariate",
    "context": 8,
    "horizon": 8,
    "variates": 2,
    "batch_context_sha256": "ef18c5f1c7998a46eb4cf2cd7f4f5d8d165b8d6a513e8955438bee99f82abf18",
}
FIXED_WINDOWS = (
    {
        "row_index": 18,
        "context_end": 8,
        "cache_index": 18,
        "context_sha256": "ab0cb26e449e60d1e86698eccd520abe07503d3e9ac7996d84500e3254613f43",
    },
    {
        "row_index": 1711,
        "context_end": 20,
        "cache_index": 29926,
        "context_sha256": "b68d54dd6f6bfe2d0d697f3b44acb17edd999907798304364e6f20e09aa6a2c3",
    },
    {
        "row_index": 3120,
        "context_end": 33,
        "cache_index": 58933,
        "context_sha256": "67f8208bb46f98dcf8c02131a6533a139d82c549ed34b0920ad7428dbd5aaced",
    },
)
DATA_FILES = {
    "data-00000-of-00001.arrow": "2dae2972e105e180b9baf1b8f1f0d67f4196c455f3ba1adf2d6d61efcd1622cf",
    "dataset_info.json": "a80271657c722e3189e9fea3f980bf3314cafb886385bf725b45a3a531e516f9",
    "state.json": "8f2c800309a09edb8c2e4a9277541261f00c95940a6f38080b7d2e375bcd4c45",
}
INVARIANCE_ATOL = 1e-4
INVARIANCE_RTOL = 1e-4
MINIMUM_NORMALIZED_INTERVENTION_EFFECT = 1e-6
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: npt.NDArray[Any]) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    header = json.dumps(
        {"dtype": "float32-le", "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    rows = []
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        row = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(_canonical_sha256(rows).encode())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not SHA256.fullmatch(expected):
        raise ValueError(f"{label} expected SHA-256 is not lowercase hexadecimal")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, observed={observed}")
    return observed


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _partition_index_sha256(dataset: str, indices: npt.NDArray[Any]) -> str:
    digest = hashlib.sha256(dataset.encode() + b"\0")
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
    encoded = dataset.encode()
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<qqQ", context, horizon, len(selected)))
    pairs = np.column_stack((selected_rows[order], selected_ends[order])).astype("<i8", copy=False)
    digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


def _cache_identities(
    cache_root: Path, audit: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    dataset_audit = next(row for row in audit["datasets"] if row["dataset"] == EXPECTED["dataset"])
    required = {
        "generated_windows": 58945,
        "duplicate_windows": 0,
        "context_length": EXPECTED["context"],
        "horizon": EXPECTED["horizon"],
        "actual_variates": [EXPECTED["variates"]],
    }
    for key, value in required.items():
        if dataset_audit.get(key) != value:
            raise ValueError(f"cache audit has unexpected {EXPECTED['dataset']} {key}")
    directory = cache_root / str(EXPECTED["dataset"])
    expected_shards = {Path(row["path"]).name: row for row in dataset_audit["shards"]}
    actual_shards = {path.name: path for path in directory.glob("shard-*.npz")}
    if set(actual_shards) != set(expected_shards):
        raise ValueError("cache shard roster differs from the pinned cache audit")
    row_parts, end_parts, bindings = [], [], []
    for name in sorted(expected_shards):
        path = actual_shards[name]
        expected_hash = str(expected_shards[name]["sha256"])
        _require_hash(path, expected_hash, f"cache shard {name}")
        with np.load(path) as arrays:
            # Teacher arrays are deliberately never indexed by this probe.
            row_parts.append(np.asarray(arrays["row_index"]))
            end_parts.append(np.asarray(arrays["context_end"]))
        bindings.append({"path": _relative_or_absolute(path), "sha256": expected_hash})
    return np.concatenate(row_parts), np.concatenate(end_parts), bindings


def _development_indices(
    rows: np.ndarray, ends: np.ndarray, selection: dict[str, Any]
) -> np.ndarray:
    outer_config = selection["outer_split"]
    outer, _ = split_cache_indices(
        rows,
        ends,
        context_length=int(EXPECTED["context"]),
        horizon=int(EXPECTED["horizon"]),
        validation_fraction=float(outer_config["validation_fraction"]),
        seed=int(outer_config["seed"]),
        mode=str(outer_config["mode"]),
    )
    outer_validation = np.asarray(outer["validation"], dtype=np.int64)
    nested = selection["nested_split"]
    inner, _ = split_cache_indices(
        rows[outer_validation],
        ends[outer_validation],
        context_length=int(EXPECTED["context"]),
        horizon=int(EXPECTED["horizon"]),
        validation_fraction=float(nested["confirmation_fraction"]),
        seed=int(nested["seed"]),
        mode="held_out_series",
    )
    development = outer_validation[np.asarray(inner["training"], dtype=np.int64)]
    entry = next(row for row in selection["datasets"] if row["dataset"] == EXPECTED["dataset"])
    expected_partition = entry["partitions"]["development"]
    actual = {
        "count": len(development),
        "cache_index_sha256": _partition_index_sha256(str(EXPECTED["dataset"]), development),
        "identity_sha256": _partition_identity_sha256(
            str(EXPECTED["dataset"]),
            rows,
            ends,
            development,
            int(EXPECTED["context"]),
            int(EXPECTED["horizon"]),
        ),
    }
    if actual != expected_partition:
        raise ValueError("reconstructed DEVELOPMENT partition differs from frozen manifest")
    return development


def _load_fixed_batch(
    data_root: Path, rows: np.ndarray, ends: np.ndarray, development: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset_root = data_root / str(EXPECTED["dataset"])
    actual_files = {
        str(path.relative_to(dataset_root)): path
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    if set(actual_files) != set(DATA_FILES):
        raise ValueError("fixed source dataset file roster changed")
    data_bindings = []
    for name, expected_hash in sorted(DATA_FILES.items()):
        path = actual_files[name]
        _require_hash(path, expected_hash, f"source dataset file {name}")
        data_bindings.append({"path": _relative_or_absolute(path), "sha256": expected_hash})

    development_set = set(np.asarray(development, dtype=np.int64).tolist())
    dataset = load_from_disk(str(dataset_root), keep_in_memory=False)
    contexts, identities = [], []
    for frozen in FIXED_WINDOWS:
        index = int(frozen["cache_index"])
        row = int(frozen["row_index"])
        end = int(frozen["context_end"])
        if index not in development_set or (int(rows[index]), int(ends[index])) != (row, end):
            raise ValueError("fixed window is not the declared DEVELOPMENT cache identity")
        values = np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
        if values.shape[0] != EXPECTED["variates"]:
            raise ValueError("fixed real example is no longer genuinely multivariate")
        context = values[:, end - int(EXPECTED["context"]) : end]
        if context.shape != (EXPECTED["variates"], EXPECTED["context"]):
            raise ValueError("fixed history has unexpected shape")
        if _array_sha256(context) != frozen["context_sha256"]:
            raise ValueError("fixed real DEVELOPMENT history content changed")
        contexts.append(context)
        identities.append(dict(frozen))
    batch = np.stack(contexts)
    if _array_sha256(batch) != EXPECTED["batch_context_sha256"]:
        raise ValueError("fixed batch content hash mismatch")
    return batch, identities, data_bindings


def _interpolate(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for sample in result:
        all_missing = np.isnan(sample).all(axis=0)
        first_valid = sample.shape[-1] if all_missing.all() else int(np.argmax(~all_missing))
        if first_valid == sample.shape[-1]:
            sample.fill(0.0)
            continue
        for row in sample[:, first_valid:]:
            missing = np.isnan(row)
            valid = np.flatnonzero(~missing)
            if missing.any():
                row[missing] = (
                    np.interp(np.flatnonzero(missing), valid, row[valid]) if valid.size else 0.0
                )
    return result


@torch.inference_mode()
def _forecast(
    model: torch.nn.Module, context: torch.Tensor, horizon: int, inference: dict[str, Any]
) -> torch.Tensor:
    observed = torch.isfinite(context)

    def call(values: torch.Tensor) -> torch.Tensor:
        output = model(values, horizon, observed_mask=observed)
        return torch.sort(output, dim=-1).values if inference["sort_quantiles"] else output

    positive = call(context)
    output = (
        (positive - call(-context).flip(-1)) / 2
        if inference["use_symmetric_averaging"]
        else positive
    )
    if inference["make_positive"]:
        nonnegative = observed.any(dim=-1) & torch.where(
            observed, context >= 0, torch.ones_like(observed)
        ).all(dim=-1)
        output = torch.where(nonnegative[..., None, None], output.clamp_min(0), output)
    return output


def _code_binding() -> dict[str, Any]:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/timesfm_lab/config.py",
        ROOT / "src/timesfm_lab/distill/data.py",
        *sorted((ROOT / "src/timesfm_lab/models").glob("*.py")),
    ]
    relative = [str(path.relative_to(ROOT)) for path in files]
    for path in relative:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode:
            raise ValueError(f"capability code is not committed: {path}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"capability code differs from HEAD:\n{dirty}")
    bindings = [{"path": path, "sha256": _sha256(ROOT / path)} for path in relative]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"git_commit": commit, "files": bindings, "sha256": _canonical_sha256(bindings)}


def _atomic_json_no_clobber(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite capability evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite capability evidence: {args.output}")

    os.environ.pop("GIFT_EVAL", None)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    _require_hash(config_path, args.config_sha256, "student config")
    _require_hash(checkpoint_path, args.checkpoint_sha256, "student checkpoint")
    _require_hash(PLAN, str(EXPECTED["plan_sha256"]), "production corpus plan")
    _require_hash(CACHE_AUDIT, str(EXPECTED["cache_audit_sha256"]), "production cache audit")
    _require_hash(SELECTION, str(EXPECTED["selection_sha256"]), "selection split")
    config = load_config(config_path)
    if config["dataset_revision"] != EXPECTED["dataset_revision"]:
        raise ValueError("student config dataset revision differs from the fixed source")
    student = config["student"]
    inference = config["inference"]
    if int(student["num_quantiles"]) != 9:
        raise ValueError("student is not configured for exactly nine quantiles")
    if int(student["max_context"]) < EXPECTED["context"]:
        raise ValueError("student cannot consume the fixed context")
    if int(student["max_horizon"]) < EXPECTED["horizon"]:
        raise ValueError("student cannot emit the fixed horizon")
    if inference.get("missing_value_preprocessing") != "timesfm3_linear_interpolation":
        raise ValueError("capability probe requires the frozen deployment preprocessing")
    for key in ("sort_quantiles", "use_symmetric_averaging", "make_positive"):
        if not isinstance(inference.get(key), bool):
            raise ValueError(f"deployment inference field {key} must be boolean")

    plan = _load_json(PLAN)
    audit = _load_json(CACHE_AUDIT)
    selection = _load_json(SELECTION)
    if plan.get("dataset_revision") != EXPECTED["dataset_revision"]:
        raise ValueError("production plan dataset revision changed")
    if audit.get("dataset_revision") != EXPECTED["dataset_revision"]:
        raise ValueError("cache audit dataset revision changed")
    if (
        selection.get("protocol_id") != EXPECTED["selection_protocol_id"]
        or selection.get("status") != "frozen_uninspected"
        or selection.get("target_accessed") is not False
        or selection.get("teacher_output_accessed") is not False
    ):
        raise ValueError("selection split is not frozen, target-blind, and uninspected")
    item = next(row for row in plan["datasets"] if row["dataset"] == EXPECTED["dataset"])
    for key in ("view_class", "context", "horizon"):
        if item.get(key) != EXPECTED[key]:
            raise ValueError(f"production plan fixed dataset {key} changed")

    rows, ends, shard_bindings = _cache_identities(args.cache_root.resolve(), audit)
    development = _development_indices(rows, ends, selection)
    batch, identities, data_bindings = _load_fixed_batch(
        args.data_root.resolve(), rows, ends, development
    )
    preflight = {
        "status": "preflight_passed",
        "dataset": EXPECTED["dataset"],
        "partition": "development",
        "input_shape": list(batch.shape),
        "output_shape": [len(FIXED_WINDOWS), EXPECTED["variates"], EXPECTED["horizon"], 9],
        "identities": identities,
        "config_sha256": args.config_sha256,
        "checkpoint_sha256": args.checkpoint_sha256,
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "teacher_outputs_used": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    code = _code_binding()
    model = build_student(student)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint is not a direct state-dict mapping")
    model.load_state_dict(state, strict=True)
    loaded_state_sha256 = _state_sha256(model)
    model.eval()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA capability probe but CUDA is unavailable")
        torch.cuda.set_device(device)
    model.to(device)
    context = torch.from_numpy(_interpolate(batch)).to(device)
    precision = str(config.get("training", {}).get("precision", "float32"))
    if precision not in {"bfloat16", "float32"}:
        raise ValueError(f"unsupported frozen inference precision: {precision}")
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    )
    with autocast:
        baseline = _forecast(model, context, int(EXPECTED["horizon"]), inference)
        intervened_context = context.clone()
        intervened_context[:, 1:] = context.roll(shifts=-1, dims=0)[:, 1:]
        if not torch.equal(intervened_context[:, 0], context[:, 0]):
            raise AssertionError("auxiliary intervention changed target history")
        if torch.equal(intervened_context[:, 1:], context[:, 1:]):
            raise AssertionError("auxiliary intervention did not change auxiliary histories")
        intervened = _forecast(model, intervened_context, int(EXPECTED["horizon"]), inference)
        standalone = _forecast(model, context[0:1], int(EXPECTED["horizon"]), inference)[0]
        regroup_order = torch.tensor([2, 0, 1], device=device)
        regrouped = _forecast(
            model,
            context.index_select(0, regroup_order),
            int(EXPECTED["horizon"]),
            inference,
        )[1]

    expected_shape = (len(FIXED_WINDOWS), EXPECTED["variates"], EXPECTED["horizon"], 9)
    if tuple(baseline.shape) != expected_shape:
        raise ValueError(f"forecast shape {tuple(baseline.shape)} != {expected_shape}")
    if tuple(intervened.shape) != expected_shape:
        raise ValueError("intervention forecast shape changed")
    expected_request_shape = (EXPECTED["variates"], EXPECTED["horizon"], 9)
    if tuple(standalone.shape) != expected_request_shape:
        raise ValueError("standalone request forecast shape changed")
    if tuple(regrouped.shape) != expected_request_shape:
        raise ValueError("regrouped request forecast shape changed")
    if not bool(
        torch.isfinite(baseline).all()
        and torch.isfinite(intervened).all()
        and torch.isfinite(standalone).all()
        and torch.isfinite(regrouped).all()
    ):
        raise ValueError("capability forecasts contain nonfinite values")
    history_scale = torch.nan_to_num(context[:, 0].float().std(dim=-1), nan=1.0).clamp_min(1e-6)
    target_effect = (intervened[:, 0].float() - baseline[:, 0].float()).abs()
    normalized_effect = target_effect / history_scale[:, None, None]
    maximum_effect = float(normalized_effect.max().cpu())
    if (
        not math.isfinite(maximum_effect)
        or maximum_effect <= MINIMUM_NORMALIZED_INTERVENTION_EFFECT
    ):
        raise ValueError("fixed auxiliary-history intervention has no nonzero target effect")
    difference = (standalone.float() - regrouped.float()).abs()
    tolerance = INVARIANCE_ATOL + INVARIANCE_RTOL * standalone.float().abs()
    if not bool((difference <= tolerance).all()):
        raise ValueError("same request changed when regrouped with unrelated requests")

    assert args.output is not None
    result = {
        "schema_version": 1,
        "protocol_id": "timesfm3-finalist-native-mv-capability-v1",
        "status": "passed",
        "established_at_utc": datetime.now(UTC).isoformat(),
        "partition": "development",
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "teacher_outputs_used": False,
        "model": {
            "config": {"path": _relative_or_absolute(config_path), "sha256": args.config_sha256},
            "checkpoint": {
                "path": _relative_or_absolute(checkpoint_path),
                "sha256": args.checkpoint_sha256,
                "loaded_state_sha256": loaded_state_sha256,
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "precision": precision,
            "device": str(device),
        },
        "code": code,
        "inputs": {
            "dataset_revision": EXPECTED["dataset_revision"],
            "plan": {"path": _relative_or_absolute(PLAN), "sha256": EXPECTED["plan_sha256"]},
            "cache_audit": {
                "path": _relative_or_absolute(CACHE_AUDIT),
                "sha256": EXPECTED["cache_audit_sha256"],
            },
            "selection_split": {
                "path": _relative_or_absolute(SELECTION),
                "sha256": EXPECTED["selection_sha256"],
            },
            "dataset_files": data_bindings,
            "cache_shards": shard_bindings,
            "fixed_batch": {
                "dataset": EXPECTED["dataset"],
                "view_class": EXPECTED["view_class"],
                "identities": identities,
                "context_sha256": EXPECTED["batch_context_sha256"],
                "shape": list(batch.shape),
            },
        },
        "checks": {
            "native_multivariate_input": True,
            "output_shape_b_v_h_q": list(baseline.shape),
            "all_outputs_finite": True,
            "nine_quantiles_per_target": baseline.shape[-1] == 9,
            "auxiliary_intervention": {
                "operation": "cyclic replacement across distinct real DEVELOPMENT requests",
                "target_variate_index": 0,
                "target_history_unchanged": True,
                "normalized_maximum_target_forecast_effect": maximum_effect,
                "minimum_required_effect": MINIMUM_NORMALIZED_INTERVENTION_EFFECT,
                "passed": True,
            },
            "cross_request_batch_invariance": {
                "reference_request": identities[0],
                "regroup_order": [2, 0, 1],
                "maximum_absolute_difference": float(difference.max().cpu()),
                "absolute_tolerance": INVARIANCE_ATOL,
                "relative_tolerance": INVARIANCE_RTOL,
                "passed": True,
            },
        },
    }
    result["evidence_payload_sha256"] = _canonical_sha256(result)
    _atomic_json_no_clobber(args.output.resolve(), result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
