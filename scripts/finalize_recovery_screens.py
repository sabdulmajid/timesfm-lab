#!/usr/bin/env python3
"""Audit recovery screens, score them on development evidence, and freeze finalists.

The script never opens GIFT-Eval or the confirmation partition.  ``prepare``
emits the exact checkpoint bindings required from a development-only latency
run.  ``audit`` recomputes the frozen score from reconciled screen results and
raw latency samples.  ``freeze`` immutably records the top one or two recipes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from timesfm_lab.distill.data import split_cache_indices

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/performance_recovery/screen_selection.yaml"


class AuditError(RuntimeError):
    """A frozen selection invariant was not satisfied."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_tree(root: Path) -> dict[str, Any]:
    files = [
        {"path": _relative(path), "sha256": _sha256(path)}
        for path in sorted(root.rglob("*.py"))
    ]
    return {"files": files, "sha256": _canonical_sha256(files)}


def _semantic_surface(source: bytes, excluded_symbols: list[str]) -> str:
    tree = ast.parse(source.decode("utf-8"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(set(excluded_symbols) - definitions)
    if missing:
        raise AuditError(f"semantic exclusion names missing definitions: {missing}")
    retained = [
        node
        for node in tree.body
        if not (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in excluded_symbols
        )
    ]
    surface = ast.Module(body=retained, type_ignores=tree.type_ignores)
    return hashlib.sha256(ast.dump(surface, include_attributes=False).encode()).hexdigest()


def _model_evaluation_source(
    files: list[dict[str, str]],
    exceptions: list[dict[str, Any]],
    read_source: Any,
) -> dict[str, Any]:
    exception_paths = {str(row["path"]) for row in exceptions}
    declared_paths = {str(row["path"]) for row in files}
    if not exception_paths.issubset(declared_paths):
        raise AuditError("model-source semantic exception is absent from the source tree")
    exact_files = [row for row in files if row["path"] not in exception_paths]
    semantic_files = []
    for row in exceptions:
        path = str(row["path"])
        semantic_files.append(
            {
                "path": path,
                "excluded_training_only_symbols": list(row["excluded_symbols"]),
                "semantics_sha256": _semantic_surface(
                    read_source(path), list(row["excluded_symbols"])
                ),
            }
        )
    payload = {"exact_files": exact_files, "semantic_files": semantic_files}
    return {**payload, "sha256": _canonical_sha256(payload)}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object in {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise AuditError(f"expected a YAML mapping in {path}")
    return value


def _path(value: str | Path) -> Path:
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise AuditError(f"path escapes repository: {path}") from error
    return path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
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


def _finite_positive(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise AuditError(f"{label} must be finite and positive, got {parsed}")
    return parsed


def _metric_components(validation: dict[str, Any], label: str) -> dict[str, float]:
    balanced = validation.get("balanced")
    if not isinstance(balanced, dict):
        raise AuditError(f"{label} is missing balanced development metrics")
    mapping = {
        "overall_dev_mase_like": "student_normalized_median_mae",
        "overall_dev_mwql_like": "student_pinball",
        "true_mv_dev_mase_like": "true_mv_student_normalized_median_mae",
        "true_mv_dev_mwql_like": "true_mv_student_pinball",
    }
    values = {
        public: _finite_positive(balanced.get(source), f"{label}.{source}")
        for public, source in mapping.items()
    }
    if int(balanced.get("dataset_count", -1)) != 77:
        raise AuditError(f"{label} must aggregate exactly 77 development datasets")
    if int(balanced.get("true_mv_dataset_count", -1)) != 11:
        raise AuditError(f"{label} must aggregate exactly 11 true-MV datasets")
    expected_forecast_error = math.exp(
        math.fsum(
            (
                math.log(values["overall_dev_mase_like"]) / 3,
                math.log(values["overall_dev_mwql_like"]) / 3,
                math.log(values["true_mv_dev_mase_like"]) / 6,
                math.log(values["true_mv_dev_mwql_like"]) / 6,
            )
        )
    )
    observed = _finite_positive(balanced.get("forecast_error"), f"{label}.forecast_error")
    if not math.isclose(observed, expected_forecast_error, rel_tol=1e-12, abs_tol=0.0):
        raise AuditError(f"{label} balanced forecast score does not recompute exactly")
    return values


def _deployment_fingerprint(config_path: Path) -> str:
    config = _load_yaml(config_path)
    return _canonical_sha256(
        {
            "student": config["student"],
            "inference": config.get("inference", {}),
        }
    )


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _git_blob(commit: str, path: str) -> bytes:
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise AuditError(f"launch input path is not repository-relative: {path}")
    return _git("show", f"{commit}:{path}")


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AuditError(f"{label} is not an ancestor of {descendant}")


def _input_hashes(launch: dict[str, Any]) -> dict[str, str]:
    rows = launch.get("input_hashes")
    if not isinstance(rows, list) or not rows:
        raise AuditError("launch record has no input hashes")
    result: dict[str, str] = {}
    for row in rows:
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not path or len(digest) != 64 or path in result:
            raise AuditError("launch input hashes contain an invalid or duplicate path")
        result[path] = digest
    return result


def _verify_launch_git_snapshot(
    launch: dict[str, Any], *, expected_registry_sha256: str, target_sha256: str
) -> None:
    commit = str(launch.get("git_commit", ""))
    if len(commit) != 40:
        raise AuditError("launch record does not contain a full git commit")
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise AuditError("launch git commit does not resolve exactly")
    _require_ancestor(commit, "HEAD", "launch commit")
    hashes = _input_hashes(launch)
    required = {
        "configs/performance_recovery/candidates.yaml": expected_registry_sha256,
        "configs/performance_recovery/targets.yaml": target_sha256,
    }
    for path, expected in required.items():
        if hashes.get(path) != expected:
            raise AuditError(f"launch did not bind the required authority: {path}")
    for path, expected in hashes.items():
        observed = hashlib.sha256(_git_blob(commit, path)).hexdigest()
        if observed != expected:
            raise AuditError(f"launch input differs from its git snapshot: {path}")


def _registry_except_protocol_and_s5(registry: dict[str, Any]) -> dict[str, Any]:
    comparable = {key: value for key, value in registry.items() if key != "protocol"}
    candidates = dict(comparable.get("candidates", {}))
    candidates.pop("S5", None)
    comparable["candidates"] = candidates
    return comparable


def _verify_protocol_transition(
    config: dict[str, Any], registry_path: Path, registry: dict[str, Any], ledger: dict[str, Any]
) -> None:
    declared = config["protocol_transition"]
    registry_protocol = registry["protocol"]
    supersedes = registry_protocol.get("supersedes", {})
    predecessor_id = str(declared["predecessor_protocol_id"])
    predecessor_sha = str(declared["predecessor_registry_sha256"])
    reference_commit = str(declared["predecessor_registry_git_commit"])
    if supersedes != {
        "protocol_id": predecessor_id,
        "registry_sha256": predecessor_sha,
        "reference_git_commit": reference_commit,
        "changed_scope": declared["successor_changed_scope"],
    }:
        raise AuditError("selection transition differs from the registry amendment")
    _require_ancestor(reference_commit, "HEAD", "predecessor registry commit")
    historical_bytes = _git_blob(reference_commit, _relative(registry_path))
    if hashlib.sha256(historical_bytes).hexdigest() != predecessor_sha:
        raise AuditError("predecessor registry git blob hash mismatch")
    historical = yaml.safe_load(historical_bytes)
    if not isinstance(historical, dict):
        raise AuditError("predecessor registry git blob is not a mapping")
    if historical.get("protocol", {}).get("id") != predecessor_id:
        raise AuditError("predecessor registry protocol mismatch")
    if _registry_except_protocol_and_s5(historical) != _registry_except_protocol_and_s5(
        registry
    ):
        raise AuditError("v1.2 changed registry scope outside protocol metadata and S5")

    allowlisted: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in registry_protocol.get("grandfathered_launch_records", []):
        key = (str(entry.get("candidate_id", "")), int(entry.get("attempt", -1)))
        if key in allowlisted:
            raise AuditError("duplicate grandfathered launch record")
        if key[0] not in set(declared["grandfathered_candidates"]):
            raise AuditError("registry grandfathered a candidate outside the frozen allowlist")
        path = _path(entry["path"])
        if _sha256(path) != entry["sha256"]:
            raise AuditError(f"grandfathered launch record hash changed: {path}")
        launch = _load_json(path)
        if launch.get("candidate_id") != key[0] or launch.get("protocol_id") != predecessor_id:
            raise AuditError(f"invalid grandfathered launch identity: {path}")
        _require_ancestor(str(launch["git_commit"]), reference_commit, "grandfathered launch")
        _verify_launch_git_snapshot(
            launch,
            expected_registry_sha256=predecessor_sha,
            target_sha256=config["target_authority"]["sha256"],
        )
        allowlisted[key] = {"path": _relative(path), "sha256": entry["sha256"]}

    ledger_protocol = str(ledger.get("protocol_id", ""))
    ledger_registry_sha = str(ledger.get("registry_sha256", ""))
    current_protocol = str(config["protocol_id"])
    current_registry_sha = str(config["registry_sha256"])
    current_ledger = (
        ledger_protocol == current_protocol and ledger_registry_sha == current_registry_sha
    )
    predecessor_ledger = (
        bool(declared["allow_predecessor_ledger_until_explicit_migration"])
        and ledger_protocol == predecessor_id
        and ledger_registry_sha == predecessor_sha
    )
    if not (current_ledger or predecessor_ledger):
        raise AuditError("ledger is neither the current authority nor the declared predecessor")

    observed_grandfathered: set[tuple[str, int]] = set()
    for candidate_id, state in ledger.get("candidates", {}).items():
        for attempt in state.get("attempts", []):
            attempt_number = int(attempt.get("attempt", -1))
            launch_path = _path(attempt["launch_record"])
            launch = _load_json(launch_path)
            key = (str(candidate_id), attempt_number)
            if launch.get("protocol_id") == predecessor_id:
                expected = allowlisted.get(key)
                if expected is None:
                    raise AuditError(
                        f"unlisted predecessor launch: {candidate_id} attempt {attempt_number}"
                    )
                if expected != {"path": _relative(launch_path), "sha256": _sha256(launch_path)}:
                    raise AuditError(f"grandfathered ledger launch mismatch: {candidate_id}")
                observed_grandfathered.add(key)
            else:
                if not current_ledger or launch.get("protocol_id") != current_protocol:
                    raise AuditError(
                        f"unapproved launch protocol: {candidate_id} attempt {attempt_number}"
                    )
                _verify_launch_git_snapshot(
                    launch,
                    expected_registry_sha256=current_registry_sha,
                    target_sha256=config["target_authority"]["sha256"],
                )
    declared_records = set(allowlisted)
    if observed_grandfathered != declared_records:
        missing = sorted(declared_records - observed_grandfathered)
        raise AuditError(f"grandfathered launch allowlist/ledger mismatch; missing={missing}")


def _verify_protocol(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_yaml(config_path)
    registry_path = _path(config["registry"])
    ledger_path = _path(config["ledger"])
    registry = _load_yaml(registry_path)
    ledger = _load_json(ledger_path)
    protocol_id = str(config["protocol_id"])
    if registry["protocol"]["id"] != protocol_id:
        raise AuditError("post-screen config and current registry protocol IDs differ")
    if _sha256(registry_path) != str(config["registry_sha256"]):
        raise AuditError("current candidate registry changed after selection freeze")
    target_path = _path(registry["protocol"]["target_config"])
    targets = _load_yaml(target_path)
    target_authority = config["target_authority"]
    registry_target = registry["protocol"].get("target_authority", {})
    if target_path != _path(target_authority["path"]):
        raise AuditError("selection config and registry name different target files")
    if _sha256(target_path) != target_authority["sha256"]:
        raise AuditError("target authority hash changed")
    if targets.get("protocol_id") != target_authority["protocol_id"]:
        raise AuditError("target authority protocol mismatch")
    if registry_target.get("protocol_id") != target_authority["protocol_id"] or registry_target.get(
        "sha256"
    ) != target_authority["sha256"]:
        raise AuditError("registry and selection config target authorities differ")
    if registry_target.get("policy") != "unchanged_by_v1.2":
        raise AuditError("registry does not declare the unchanged v1.1 target authority")
    _verify_protocol_transition(config, registry_path, registry, ledger)
    target_weights = {
        str(row["metric"]): float(row["weight"])
        for row in targets["selection"]["screening_score"]["terms"]
    }
    expected_target_weights = {
        "overall_dev_mase": 0.30,
        "overall_dev_mwql": 0.30,
        "true_mv_dev_mase": 0.15,
        "true_mv_dev_mwql": 0.15,
        "aggregate_end_to_end_latency": 0.10,
    }
    if target_weights != expected_target_weights:
        raise AuditError("target-authority screening weights changed")
    configured_weights = {
        str(key): float(value) for key, value in config["score"]["weights"].items()
    }
    expected_configured = {
        "overall_dev_mase_like": 0.30,
        "overall_dev_mwql_like": 0.30,
        "true_mv_dev_mase_like": 0.15,
        "true_mv_dev_mwql_like": 0.15,
        "aggregate_end_to_end_latency_ms": 0.10,
    }
    if configured_weights != expected_configured or not math.isclose(
        sum(configured_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AuditError("post-screen score is not the frozen 30/30/15/15/10 rule")
    split_path = _path(config["selection_split_manifest"])
    if _sha256(split_path) != str(config["selection_split_manifest_sha256"]):
        raise AuditError("selection split hash changed")
    split = _load_json(split_path)
    if split.get("protocol_id") != target_authority["protocol_id"]:
        raise AuditError("selection split is not bound to the unchanged target authority")
    if split.get("status") != "frozen_uninspected":
        raise AuditError("confirmation split is no longer frozen and uninspected")
    if (
        split.get("target_accessed") is not False
        or split.get("teacher_output_accessed") is not False
    ):
        raise AuditError("selection manifest reports forbidden access during its creation")
    return config, registry, ledger


def _launch_trainer_sha(launch: dict[str, Any], trainer_path: str) -> str:
    matches = [row["sha256"] for row in launch["input_hashes"] if row["path"] == trainer_path]
    if len(matches) != 1:
        raise AuditError("launch record does not contain exactly one trainer hash")
    return str(matches[0])


def _launch_bound_file(launch: dict[str, Any], path: Path, label: str) -> str:
    relative = _relative(path)
    matches = [row["sha256"] for row in launch["input_hashes"] if row["path"] == relative]
    if len(matches) != 1:
        raise AuditError(f"launch record does not bind exactly one {label}: {relative}")
    observed = _sha256(path)
    if observed != matches[0]:
        raise AuditError(f"{label} changed after the candidate launch: {relative}")
    return observed


def _launch_bound_model_source(launch: dict[str, Any]) -> dict[str, Any]:
    declared = sorted(
        (
            {"path": str(row["path"]), "sha256": str(row["sha256"])}
            for row in launch["input_hashes"]
            if str(row["path"]).startswith("src/timesfm_lab/")
            and str(row["path"]).endswith(".py")
        ),
        key=lambda row: row["path"],
    )
    if not declared:
        raise AuditError("candidate launch does not bind model source files")
    return {"files": declared, "sha256": _canonical_sha256(declared)}


def _candidate_record(
    slot: str,
    candidate: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    attempts = state.get("attempts", [])
    if state.get("status") != "succeeded" or not attempts:
        raise AuditError(f"{slot} is not a reconciled successful screen")
    attempt = attempts[-1]
    if attempt.get("status") != "succeeded" or attempt.get("actual_gpu_hours") is None:
        raise AuditError(f"{slot} latest attempt is not terminal and cost-reconciled")
    validation_record = attempt.get("result_validation", {})
    if not validation_record or not all(validation_record.get("checks", {}).values()):
        raise AuditError(f"{slot} lacks a passing manager result validation")
    output_path = _path(candidate["output"])
    if _sha256(output_path) != validation_record.get("result_sha256"):
        raise AuditError(f"{slot} result changed after manager reconciliation")
    result = _load_json(output_path)
    launch_path = _path(attempt["launch_record"])
    worker_path = _path(attempt["worker_record"])
    launch = _load_json(launch_path)
    worker = _load_json(worker_path)
    if worker.get("exit_code") != 0 or launch.get("gift_evaluation") is not False:
        raise AuditError(f"{slot} launch/worker record is not an eligible screen")
    if launch.get("selection_partition") != "development":
        raise AuditError(f"{slot} launch did not declare development-only selection")
    if result.get("status") != "succeeded" or result.get("git_commit") != launch.get("git_commit"):
        raise AuditError(f"{slot} result status/commit disagrees with its launch")
    extra = result.get("extra", {})
    training = extra.get("training", {})
    if extra.get("variant") != candidate["variant"]:
        raise AuditError(f"{slot} variant mismatch")
    if extra.get("validation_partition") != "development":
        raise AuditError(f"{slot} did not evaluate the development partition")
    if extra.get("confirmation_partition_accessed") is not False:
        raise AuditError(f"{slot} reports confirmation access")
    if extra.get("selection_split_manifest_sha256") != registry["selection_split"][
        "manifest_sha256"
    ]:
        raise AuditError(f"{slot} selection-manifest hash mismatch")
    if training.get("validation_selection_metric") != "balanced_forecast_error":
        raise AuditError(f"{slot} used a different checkpoint-selection rule")
    if int(training.get("world_size", -1)) != 1:
        raise AuditError(f"{slot} screen was not the declared single-GPU layout")
    if int(extra.get("training_seed", -1)) != int(registry["screening"]["seed"]):
        raise AuditError(f"{slot} training seed mismatch")
    if int(extra.get("validation_split_seed", -1)) != int(registry["screening"]["split_seed"]):
        raise AuditError(f"{slot} split seed mismatch")
    steps = int(training.get("steps", -1))
    windows = int(training.get("windows_processed", -1))
    if not 0 < steps <= int(registry["screening"]["maximum_steps"]):
        raise AuditError(f"{slot} has an invalid update count")
    if not 0 < windows <= int(registry["screening"]["maximum_examples_processed"]):
        raise AuditError(f"{slot} has an invalid example count")

    curve = extra.get("learning_curve")
    if not isinstance(curve, list) or not curve:
        raise AuditError(f"{slot} has no development learning curve")
    curve_steps = [int(row["step"]) for row in curve]
    if curve_steps != sorted(set(curve_steps)) or curve_steps[-1] != steps:
        raise AuditError(f"{slot} learning-curve steps are incomplete or unordered")
    scored_rows = []
    for row in curve:
        components = _metric_components(row["validation"], f"{slot}.step{row['step']}")
        scored_rows.append(
            (float(row["validation"]["balanced"]["forecast_error"]), row, components)
        )
    _, best_row, components = min(scored_rows, key=lambda item: (item[0], int(item[1]["step"])))

    best = validation_record.get("checkpoints", {}).get("best_checkpoint", {})
    best_path = _path(best.get("path", ""))
    if _sha256(best_path) != best.get("sha256"):
        raise AuditError(f"{slot} best checkpoint changed after reconciliation")
    if Path(extra.get("best_checkpoint", "")).resolve() != best_path:
        raise AuditError(f"{slot} result names a different best checkpoint")
    config_path = _path(candidate["config"])
    config_sha256 = _launch_bound_file(launch, config_path, f"{slot} candidate config")
    model_source = _launch_bound_model_source(launch)
    trainer_path = str(registry["trainer"]["path"])
    trainer_sha = _launch_trainer_sha(launch, trainer_path)
    evaluation_config = config["development_evaluation"]
    semantic_exclusions = list(
        evaluation_config["validation_semantic_excluded_training_symbols"]
    )
    launch_semantics = _semantic_surface(
        _git_blob(str(launch["git_commit"]), trainer_path), semantic_exclusions
    )
    launch_evaluation_source = _model_evaluation_source(
        model_source["files"],
        list(evaluation_config["model_source_semantic_exceptions"]),
        lambda path: _git_blob(str(launch["git_commit"]), path),
    )
    return {
        "candidate_id": slot,
        "family": candidate["family"],
        "variant": candidate["variant"],
        "hypothesis": candidate["hypothesis"],
        "parameter_count": int(extra["parameter_count"]),
        "best_step": int(best_row["step"]),
        "best_windows_processed": int(best_row["windows_processed"]),
        "development_metrics": components,
        "checkpoint": {"path": _relative(best_path), "sha256": best["sha256"]},
        "config": {"path": _relative(config_path), "sha256": config_sha256},
        "deployment_fingerprint_sha256": _deployment_fingerprint(config_path),
        "launch_model_source": model_source,
        "launch_model_evaluation_source": launch_evaluation_source,
        "training": {
            "steps": steps,
            "windows_processed": windows,
            "stopped_for_plateau": bool(training.get("stopped_for_plateau")),
            "actual_gpu_hours_all_attempts": math.fsum(
                float(row.get("actual_gpu_hours") or 0.0) for row in attempts
            ),
            "attempt_count": len(attempts),
            "successful_attempt": int(attempt["attempt"]),
        },
        "provenance": {
            "result": {"path": _relative(output_path), "sha256": _sha256(output_path)},
            "launch": {"path": _relative(launch_path), "sha256": _sha256(launch_path)},
            "worker": {"path": _relative(worker_path), "sha256": _sha256(worker_path)},
            "git_commit": launch["git_commit"],
            "validation_implementation_sha256": trainer_sha,
            "validation_semantics_sha256": launch_semantics,
            "selection_split_manifest_sha256": extra["selection_split_manifest_sha256"],
            "validation_partition": "development",
            "confirmation_partition_accessed": False,
            "gift_eval_data_accessed": False,
        },
    }


def _incumbent_binding(config: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    authority = config["incumbent"]
    config_path = _path(authority["config"])
    checkpoint_path = _path(authority["checkpoint"])
    if _sha256(config_path) != authority["config_sha256"]:
        raise AuditError("predeclared incumbent config changed")
    if _sha256(checkpoint_path) != authority["checkpoint_sha256"]:
        raise AuditError("predeclared incumbent checkpoint changed")
    for source in authority["model_source_files"]:
        if _sha256(_path(source["path"])) != source["sha256"]:
            raise AuditError(f"predeclared incumbent model source changed: {source['path']}")
    targets = _load_yaml(_path(registry["protocol"]["target_config"]))
    declared = targets["selection"]["incumbent"]
    if authority["checkpoint_sha256"] != declared["checkpoint_sha256"]:
        raise AuditError("incumbent authority differs from the target protocol")
    return {
        "id": "incumbent",
        "method": declared["method"],
        "seed": int(declared["seed"]),
        "selected_step": int(declared["selected_step"]),
        "parameter_count": int(targets["model_size"]["current_reference_student_parameter_count"]),
        "checkpoint": {"path": authority["checkpoint"], "sha256": authority["checkpoint_sha256"]},
        "config": {"path": authority["config"], "sha256": authority["config_sha256"]},
        "predeclared_model_source_files": authority["model_source_files"],
    }


def _collect_screens(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    config, registry, ledger = _verify_protocol(config_path)
    states = ledger.get("candidates", {})
    if set(states) != set(registry["candidates"]):
        raise AuditError("registry and GPU ledger candidate slots differ")
    blocked = {
        slot: state.get("status")
        for slot, state in states.items()
        if state.get("status") in {"running", "launching", "unreconciled", "invalid"}
    }
    if blocked:
        raise AuditError(f"screen ledger is not terminal/reconciled: {blocked}")
    successful = [
        _candidate_record(
            slot, registry["candidates"][slot], states[slot], config, registry
        )
        for slot in sorted(states)
        if states[slot].get("status") == "succeeded"
    ]
    if not successful:
        raise AuditError("no successful recovery screen is available")
    if len(successful) > int(config["candidate_scope"]["maximum_substantive_screens"]):
        raise AuditError("successful screen count exceeds the frozen cap")
    incumbent = _incumbent_binding(config, registry)
    dispositions = {slot: str(states[slot].get("status")) for slot in sorted(states)}
    return config, registry, incumbent, successful, dispositions


def _evaluation_authority(
    config: dict[str, Any], incumbent: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluation = config["development_evaluation"]
    implementation_path = _path(evaluation["validation_implementation"])
    runner_path = _path(evaluation["evaluation_runner"])
    if _sha256(runner_path) != evaluation["evaluation_runner_sha256"]:
        raise AuditError("common development-evaluation runner changed")
    current_source = _source_tree(ROOT / "src/timesfm_lab")
    current_evaluation_source = _model_evaluation_source(
        current_source["files"],
        list(evaluation["model_source_semantic_exceptions"]),
        lambda path: _path(path).read_bytes(),
    )
    if current_evaluation_source["sha256"] != evaluation[
        "model_evaluation_source_sha256"
    ]:
        raise AuditError("common model/evaluation source changed after selection freeze")
    for row in candidates:
        if row["launch_model_evaluation_source"] != current_evaluation_source:
            raise AuditError(
                f"{row['candidate_id']} launch model/evaluation source is incompatible"
            )
    implementation_bytes = implementation_path.read_bytes()
    semantic_exclusions = list(
        evaluation["validation_semantic_excluded_training_symbols"]
    )
    current_semantics = _semantic_surface(implementation_bytes, semantic_exclusions)
    if current_semantics != evaluation["validation_semantics_sha256"]:
        raise AuditError("common validation semantics changed after selection freeze")
    launch_trainer_bindings = {}
    for row in candidates:
        launch_full = row["provenance"]["validation_implementation_sha256"]
        launch_semantics = row["provenance"]["validation_semantics_sha256"]
        if launch_semantics != current_semantics:
            raise AuditError(
                f"{row['candidate_id']} launch/common validation semantics differ"
            )
        launch_trainer_bindings[row["candidate_id"]] = {
            "launch_full_sha256": launch_full,
            "launch_validation_semantics_sha256": launch_semantics,
            "current_full_sha256_match": launch_full == _sha256(implementation_path),
        }
    configs = [incumbent["config"], *(row["config"] for row in candidates)]
    inference_values = [_load_yaml(_path(row["path"])).get("inference", {}) for row in configs]
    if len({_canonical_sha256(value) for value in inference_values}) != 1:
        raise AuditError("candidate deployment inference policies are not comparable")
    return {
        "validation_implementation": {
            "path": _relative(implementation_path),
            "sha256": _sha256(implementation_path),
            "excluded_training_only_symbols": semantic_exclusions,
            "semantics_sha256": current_semantics,
        },
        "evaluation_runner": {
            "path": _relative(runner_path),
            "sha256": _sha256(runner_path),
        },
        "model_source": current_source,
        "launch_compatible_model_evaluation_source": current_evaluation_source,
        "candidate_launch_validation_bindings": launch_trainer_bindings,
        "input_preprocessing": evaluation["input_preprocessing"],
        "validation_partition": "development",
        "batch_size_by_context": evaluation["batch_size_by_context"],
        "inference_policy_sha256": _canonical_sha256(inference_values[0]),
    }


def _development_bindings(
    incumbent: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    bindings = {
        "incumbent": {
            "checkpoint": incumbent["checkpoint"],
            "config": incumbent["config"],
            "original_model_source": incumbent["predeclared_model_source_files"],
        }
    }
    for row in candidates:
        bindings[row["candidate_id"]] = {
            "checkpoint": row["checkpoint"],
            "config": row["config"],
            "original_model_source": row["launch_model_source"],
            "original_model_evaluation_source": row[
                "launch_model_evaluation_source"
            ],
            "original_best_step": row["best_step"],
        }
    return bindings


def _attach_common_development(
    path: Path,
    config: dict[str, Any],
    registry: dict[str, Any],
    incumbent: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    if payload.get("schema_version") != 2 or payload.get("status") != "succeeded":
        raise AuditError("common development evaluation is not successful schema-v2 evidence")
    if payload.get("protocol_id") != config["protocol_id"]:
        raise AuditError("common development evaluation protocol mismatch")
    if payload.get("partition") != "development":
        raise AuditError("common evaluation did not use development")
    if payload.get("confirmation_partition_accessed") is not False:
        raise AuditError("common evaluation reports confirmation access")
    if payload.get("gift_eval_data_accessed") is not False:
        raise AuditError("common evaluation reports GIFT-Eval access")
    request_reference = payload.get("evaluation_request", {})
    request_path = _path(request_reference.get("path", ""))
    if _sha256(request_path) != request_reference.get("sha256"):
        raise AuditError("common development-evaluation request hash mismatch")
    request = _load_json(request_path)
    if request.get("status") != "awaiting_common_development_evaluation":
        raise AuditError("common development artifact references an invalid request")
    if request.get("confirmation_partition_accessed") is not False:
        raise AuditError("common development request reports confirmation access")
    if request.get("gift_eval_data_accessed") is not False:
        raise AuditError("common development request reports GIFT-Eval access")
    expected_bindings = _development_bindings(incumbent, candidates)
    if request.get("model_bindings") != expected_bindings:
        raise AuditError("development-evaluation request model bindings changed")
    authority = _evaluation_authority(config, incumbent, candidates)
    if request.get("evaluation_authority") != authority:
        raise AuditError("development-evaluation authority changed")
    if payload.get("evaluation_authority") != authority:
        raise AuditError("development runtime used a different evaluation authority")
    data_authority = _data_authority(config, registry)
    if request.get("data_authority") != data_authority:
        raise AuditError("development-evaluation request data authority changed")
    if payload.get("data_authority") != data_authority:
        raise AuditError("development runtime used a different data authority")
    models = payload.get("models", {})
    if set(models) != set(expected_bindings):
        raise AuditError("common development evaluation model set mismatch")
    attached = []
    by_id = {row["candidate_id"]: row for row in candidates}
    for model_id, binding in expected_bindings.items():
        model = models[model_id]
        if (
            model.get("checkpoint") != binding["checkpoint"]
            or model.get("config") != binding["config"]
        ):
            raise AuditError(f"{model_id} common-development checkpoint/config mismatch")
        metrics = _metric_components(model["validation"], f"{model_id}.common_development")
        if model_id == "incumbent":
            if int(model.get("parameter_count", -1)) != incumbent["parameter_count"]:
                raise AuditError("incumbent parameter count changed in common evaluation")
            incumbent = {
                **incumbent,
                "development_metrics": metrics,
                "deployment_fingerprint_sha256": model["deployment_fingerprint_sha256"],
                "model_source": authority["model_source"],
                "inference_implementation": authority["validation_implementation"],
                "provenance": {"path": _relative(path), "sha256": _sha256(path)},
            }
        else:
            original = by_id[model_id]
            if int(model.get("parameter_count", -1)) != original["parameter_count"]:
                raise AuditError(f"{model_id} parameter count changed in common evaluation")
            attached.append(
                {
                    **original,
                    "screen_best_development_metrics": original["development_metrics"],
                    "development_metrics": metrics,
                    "deployment_fingerprint_sha256": model[
                        "deployment_fingerprint_sha256"
                    ],
                    "model_source": authority["model_source"],
                    "inference_implementation": authority["validation_implementation"],
                }
            )
    return incumbent, attached


def _collect(
    config_path: Path, development_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    config, registry, incumbent, candidates, dispositions = _collect_screens(config_path)
    incumbent, candidates = _attach_common_development(
        development_path, config, registry, incumbent, candidates
    )
    return config, registry, incumbent, candidates, dispositions


def _data_authority(config: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    corpus = registry["corpus"]
    plan_path = _path(corpus["plan"])
    audit_path = _path(corpus["cache_audit"])
    split_path = _path(config["selection_split_manifest"])
    if _sha256(plan_path) != corpus["plan_sha256"]:
        raise AuditError("corpus plan changed after registry freeze")
    if _sha256(audit_path) != corpus["cache_audit_sha256"]:
        raise AuditError("cache audit changed after registry freeze")
    if _sha256(split_path) != config["selection_split_manifest_sha256"]:
        raise AuditError("selection split changed after selection-config freeze")
    return {
        "dataset_revision": corpus["dataset_revision"],
        "plan": {"path": corpus["plan"], "sha256": corpus["plan_sha256"]},
        "cache_audit": {
            "path": corpus["cache_audit"],
            "sha256": corpus["cache_audit_sha256"],
        },
        "data_root": corpus["data_root"],
        "cache_root": corpus["cache_root"],
        "selection_split_manifest": {
            "path": config["selection_split_manifest"],
            "sha256": config["selection_split_manifest_sha256"],
            "protocol_id": config["target_authority"]["protocol_id"],
        },
    }


def _partition_index_sha256(dataset: str, indices: np.ndarray) -> str:
    digest = hashlib.sha256(dataset.encode("utf-8") + b"\0")
    digest.update(np.sort(indices).astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _input_sha256(contexts: list[np.ndarray], identities: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for (row, end), context in zip(identities, contexts, strict=True):
        digest.update(f"{row}:{end}".encode())
        contiguous = np.ascontiguousarray(context)
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _partition_identity_sha256(
    dataset: str,
    rows: np.ndarray,
    ends: np.ndarray,
    indices: np.ndarray,
    context: int,
    horizon: int,
) -> str:
    chosen_rows = rows[indices].astype("<i8", copy=False)
    chosen_ends = ends[indices].astype("<i8", copy=False)
    order = np.lexsort((chosen_ends, chosen_rows))
    encoded = dataset.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<qqQ", context, horizon, len(indices)))
    pairs = np.column_stack((chosen_rows[order], chosen_ends[order])).astype(
        "<i8", copy=False
    )
    digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


def _development_indices(
    dataset: str,
    rows: np.ndarray,
    ends: np.ndarray,
    context: int,
    horizon: int,
    split: dict[str, Any],
    split_entry: dict[str, Any],
) -> np.ndarray:
    outer, _ = split_cache_indices(
        rows,
        ends,
        context_length=context,
        horizon=horizon,
        validation_fraction=float(split["outer_split"]["validation_fraction"]),
        seed=int(split["outer_split"]["seed"]),
        mode=str(split["outer_split"]["mode"]),
    )
    outer_validation = np.asarray(outer["validation"], dtype=np.int64)
    validation_rows = rows[outer_validation]
    validation_ends = ends[outer_validation]
    mode = "held_out_series" if len(np.unique(validation_rows)) > 1 else "blocked_time"
    try:
        inner, _ = split_cache_indices(
            validation_rows,
            validation_ends,
            context_length=context,
            horizon=horizon,
            validation_fraction=float(split["nested_split"]["confirmation_fraction"]),
            seed=int(split["nested_split"]["seed"]),
            mode=mode,
        )
        development = outer_validation[np.asarray(inner["training"], dtype=np.int64)]
    except ValueError as error:
        if mode != "blocked_time" or "left no training windows" not in str(error):
            raise
        development = outer_validation
    expected = split_entry["partitions"]["development"]
    actual = {
        "count": len(development),
        "cache_index_sha256": _partition_index_sha256(dataset, development),
        "identity_sha256": _partition_identity_sha256(
            dataset, rows, ends, development, context, horizon
        ),
    }
    if actual != expected:
        raise AuditError(f"{dataset} development identities do not reconstruct exactly")
    return development


def _frozen_latency_workloads(
    config: dict[str, Any], registry: dict[str, Any]
) -> list[dict[str, Any]]:
    from datasets import load_from_disk  # type: ignore[import-untyped]

    authority = _data_authority(config, registry)
    plan = _load_json(_path(authority["plan"]["path"]))
    split = _load_json(_path(authority["selection_split_manifest"]["path"]))
    plan_items = {str(row["dataset"]): row for row in plan["datasets"]}
    split_items = {str(row["dataset"]): row for row in split["datasets"]}
    workload_path = _path(config["latency_evidence"]["workload_config"])
    workload_config = _load_yaml(workload_path)
    cache_root = _path(authority["cache_root"])
    data_root = _path(authority["data_root"])
    workloads = []
    for spec in workload_config["workloads"]:
        dataset = str(spec["dataset"])
        item = plan_items[dataset]
        rows_parts = []
        ends_parts = []
        for shard_path in sorted((cache_root / dataset).glob("shard-*.npz")):
            with np.load(shard_path) as shard:
                rows_parts.append(np.asarray(shard["row_index"], dtype=np.int64))
                ends_parts.append(np.asarray(shard["context_end"], dtype=np.int64))
        if not rows_parts:
            raise AuditError(f"no cache shards for latency workload {dataset}")
        rows = np.concatenate(rows_parts)
        ends = np.concatenate(ends_parts)
        context = int(item["context"])
        horizon = int(item["horizon"])
        development = _development_indices(
            dataset,
            rows,
            ends,
            context,
            horizon,
            split,
            split_items[dataset],
        )
        batch = int(spec["batch"])
        chosen = development[:batch]
        if len(chosen) != batch:
            raise AuditError(f"{dataset} has fewer than {batch} development windows")
        variate_counts = [int(value) for value in item["actual_variate_counts"]]
        if len(variate_counts) != 1:
            raise AuditError(f"{dataset} does not have one fixed variate count")
        identity_pairs = [
            (int(rows[index]), int(ends[index])) for index in chosen
        ]
        identities = [
            {"row_index": row, "context_end": end} for row, end in identity_pairs
        ]
        source = load_from_disk(str(data_root / dataset), keep_in_memory=False)
        contexts = [
            np.atleast_2d(np.asarray(source[row]["target"], dtype=np.float32))[
                :, end - context : end
            ]
            for row, end in identity_pairs
        ]
        input_shape = [batch, variate_counts[0], context]
        output_shape = [batch, variate_counts[0], horizon, 9]
        if any(list(value.shape) != input_shape[1:] for value in contexts):
            raise AuditError(f"{dataset} raw latency inputs do not match the frozen shape")
        workloads.append(
            {
                "name": str(spec["name"]),
                "dataset": dataset,
                "cache_indices": chosen.tolist(),
                "identities": identities,
                "batch": batch,
                "variates": variate_counts[0],
                "context": context,
                "horizon": horizon,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "shape_sha256": _canonical_sha256(
                    {"input_shape": input_shape, "output_shape": output_shape}
                ),
                "identity_sha256": _canonical_sha256(identities),
                "input_sha256": _input_sha256(contexts, identity_pairs),
            }
        )
    return workloads


def _latency_bindings(
    incumbent: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        row.get("candidate_id", "incumbent"): {
            "checkpoint": row["checkpoint"],
            "config": row["config"],
            "deployment_fingerprint_sha256": row["deployment_fingerprint_sha256"],
            "model_source": row["model_source"],
            "inference_implementation": row["inference_implementation"],
        }
        for row in [incumbent, *candidates]
    }


def _latency_implementation(config: dict[str, Any]) -> dict[str, str]:
    policy = config["latency_evidence"]
    path = _path(policy["implementation"])
    observed = _sha256(path)
    if observed != policy["implementation_sha256"]:
        raise AuditError("development latency runner changed after selection freeze")
    return {"path": _relative(path), "sha256": observed}


def _latencies(
    path: Path,
    config: dict[str, Any],
    registry: dict[str, Any],
    incumbent: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, float]:
    evidence = _load_json(path)
    policy = config["latency_evidence"]
    if evidence.get("schema_version") != int(policy["schema_version"]):
        raise AuditError("latency evidence schema mismatch")
    if (
        evidence.get("status") != "succeeded"
        or evidence.get("protocol_id") != config["protocol_id"]
    ):
        raise AuditError("latency evidence status/protocol mismatch")
    if evidence.get("partition") != policy["partition"]:
        raise AuditError("screen latency did not use the development partition")
    if evidence.get("confirmation_partition_accessed") is not False:
        raise AuditError("latency evidence reports confirmation access")
    if evidence.get("gift_eval_data_accessed") is not False:
        raise AuditError("screen latency evidence reports GIFT-Eval access")
    expected_implementation = _latency_implementation(config)
    if evidence.get("latency_implementation") != expected_implementation:
        raise AuditError("latency runtime implementation binding mismatch")
    request_reference = evidence.get("latency_request", {})
    request_path = _path(request_reference.get("path", ""))
    if _sha256(request_path) != request_reference.get("sha256"):
        raise AuditError("latency request hash mismatch")
    request = _load_json(request_path)
    if request.get("status") != "awaiting_development_latency_evidence":
        raise AuditError("latency evidence does not reference a prepared request")
    if request.get("protocol_id") != config["protocol_id"]:
        raise AuditError("latency request protocol mismatch")
    if request.get("confirmation_partition_accessed") is not False:
        raise AuditError("latency request reports confirmation access")
    if request.get("gift_eval_data_accessed") is not False:
        raise AuditError("latency request reports GIFT-Eval access")
    if request.get("latency_implementation") != expected_implementation:
        raise AuditError("latency request implementation binding mismatch")
    measurement = evidence.get("measurement", {})
    frozen_workload_config = _load_yaml(
        _path(config["latency_evidence"]["workload_config"])
    )
    expected_warmups = int(frozen_workload_config["warmup_calls_per_workload"])
    expected_samples = int(frozen_workload_config["measured_calls_per_workload"])
    if int(measurement.get("warmup_calls_per_workload", -1)) != expected_warmups:
        raise AuditError("latency evidence warmup count differs from the frozen workload")
    if int(measurement.get("measured_calls_per_workload", -1)) != expected_samples:
        raise AuditError("latency evidence sample count differs from the frozen workload")
    if measurement.get("synchronize_each_sample") is not True:
        raise AuditError("latency evidence did not synchronize every sample")
    if measurement.get("single_gpu_no_contention") is not True:
        raise AuditError("latency evidence was not isolated on one GPU")
    workload_manifest = evidence.get("workload_manifest", {})
    workload_path = _path(workload_manifest.get("path", ""))
    if _sha256(workload_path) != workload_manifest.get("sha256"):
        raise AuditError("latency workload manifest hash mismatch")
    required_workload_path = _path(policy["workload_config"])
    if workload_path != required_workload_path:
        raise AuditError("latency evidence used a different development workload config")
    expected_bindings = _latency_bindings(incumbent, candidates)
    if request.get("model_bindings") != expected_bindings:
        raise AuditError("latency request model bindings differ from audited checkpoints")
    expected_data_authority = _data_authority(config, registry)
    if request.get("data_authority") != expected_data_authority:
        raise AuditError("latency request data authority differs from the registry")
    if evidence.get("data_authority") != expected_data_authority:
        raise AuditError("latency runtime data authority differs from the registry")
    expected_workloads = _frozen_latency_workloads(config, registry)
    if request.get("workloads") != expected_workloads:
        raise AuditError("latency request workload identities/shapes changed")
    observed_inputs = evidence.get("inputs", [])
    if not isinstance(observed_inputs, list) or len(observed_inputs) != len(expected_workloads):
        raise AuditError("latency evidence input count mismatch")
    binding_fields = tuple(expected_workloads[0])
    for expected, observed in zip(expected_workloads, observed_inputs, strict=True):
        if {name: observed.get(name) for name in binding_fields} != expected:
            raise AuditError(f"latency runtime input binding mismatch for {expected['name']}")
    models = evidence.get("models", {})
    if set(models) != set(expected_bindings):
        raise AuditError(
            "latency evidence model set differs from successful screens plus incumbent"
        )
    aggregate: dict[str, float] = {}
    workload_names: set[str] | None = None
    output_shapes = {row["name"]: row["output_shape"] for row in expected_workloads}
    for model_id, binding in expected_bindings.items():
        model = models[model_id]
        if model.get("checkpoint") != binding["checkpoint"]:
            raise AuditError(f"{model_id} latency checkpoint binding mismatch")
        if model.get("config") != binding["config"]:
            raise AuditError(f"{model_id} latency config binding mismatch")
        if model.get("deployment_fingerprint_sha256") != binding[
            "deployment_fingerprint_sha256"
        ]:
            raise AuditError(f"{model_id} latency deployment fingerprint mismatch")
        if model.get("model_source") != binding["model_source"]:
            raise AuditError(f"{model_id} latency model-source binding mismatch")
        if model.get("inference_implementation") != binding["inference_implementation"]:
            raise AuditError(f"{model_id} latency inference-code binding mismatch")
        workloads = model.get("workloads", {})
        if not isinstance(workloads, dict) or not workloads:
            raise AuditError(f"{model_id} has no latency workloads")
        names = set(workloads)
        if names != set(output_shapes):
            raise AuditError(f"{model_id} latency workload set differs from the request")
        if workload_names is None:
            workload_names = names
        elif names != workload_names:
            raise AuditError("latency workload names differ across models")
        medians = []
        for name, row in workloads.items():
            if row.get("output_shape") != output_shapes[name]:
                raise AuditError(f"{model_id}.{name} output shape is not [B,V,H,9]")
            samples = [
                _finite_positive(value, f"{model_id}.{name}.sample")
                for value in row.get("end_to_end_latency_ms_samples", [])
            ]
            if len(samples) != expected_samples:
                raise AuditError(f"{model_id}.{name} measured-call count changed")
            median = statistics.median(samples)
            declared = _finite_positive(
                row.get("p50_end_to_end_latency_ms"), f"{model_id}.{name}.p50"
            )
            if not math.isclose(median, declared, rel_tol=1e-12, abs_tol=0.0):
                raise AuditError(f"{model_id}.{name} p50 does not recompute from raw samples")
            medians.append(median)
        recomputed = math.exp(math.fsum(math.log(value) for value in medians) / len(medians))
        declared_aggregate = _finite_positive(
            model.get("aggregate_end_to_end_latency_ms"), f"{model_id}.aggregate_latency"
        )
        if not math.isclose(recomputed, declared_aggregate, rel_tol=1e-12, abs_tol=0.0):
            raise AuditError(f"{model_id} aggregate latency does not recompute")
        aggregate[model_id] = recomputed
    return aggregate


def _score(
    config: dict[str, Any],
    incumbent: dict[str, Any],
    candidates: list[dict[str, Any]],
    latencies: dict[str, float],
) -> list[dict[str, Any]]:
    weights = config["score"]["weights"]
    incumbent_values = {
        **incumbent["development_metrics"],
        "aggregate_end_to_end_latency_ms": latencies["incumbent"],
    }
    ranking = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        values = {
            **candidate["development_metrics"],
            "aggregate_end_to_end_latency_ms": latencies[candidate_id],
        }
        ratios = {name: values[name] / incumbent_values[name] for name in weights}
        score = math.exp(
            math.fsum(float(weights[name]) * math.log(ratios[name]) for name in weights)
        )
        ranking.append(
            {
                **candidate,
                "aggregate_end_to_end_latency_ms": latencies[candidate_id],
                "ratios_to_incumbent": ratios,
                "screening_score": score,
            }
        )
    ranking.sort(
        key=lambda row: (
            row["screening_score"],
            row["development_metrics"]["overall_dev_mwql_like"],
            row["parameter_count"],
            row["candidate_id"],
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def _pair_status(candidates: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    pair = list(config["candidate_scope"]["compact_attribution_pair"])
    present = {row["candidate_id"] for row in candidates}
    return {
        "required_pair": pair,
        "successful_members": [slot for slot in pair if slot in present],
        "complete": all(slot in present for slot in pair),
        "policy": (
            "a compact-decoder finalist is ineligible unless both GT and KD screens succeeded"
        ),
    }


def _command_prepare_evaluation(args: argparse.Namespace) -> int:
    config, registry, incumbent, candidates, dispositions = _collect_screens(args.config)
    payload = {
        "schema_version": 1,
        "status": "awaiting_common_development_evaluation",
        "protocol_id": config["protocol_id"],
        "created_at_utc": _now(),
        "partition": "development",
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "screen_dispositions": dispositions,
        "model_bindings": _development_bindings(incumbent, candidates),
        "data_authority": _data_authority(config, registry),
        "evaluation_authority": _evaluation_authority(config, incumbent, candidates),
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


def _command_prepare(args: argparse.Namespace) -> int:
    config, registry, incumbent, candidates, dispositions = _collect(
        args.config, args.development_evaluation
    )
    workload_path = _path(config["latency_evidence"]["workload_config"])
    workloads = _frozen_latency_workloads(config, registry)
    payload = {
        "schema_version": 1,
        "status": "awaiting_development_latency_evidence",
        "protocol_id": config["protocol_id"],
        "created_at_utc": _now(),
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "screen_dispositions": dispositions,
        "model_bindings": _latency_bindings(incumbent, candidates),
        "latency_implementation": _latency_implementation(config),
        "data_authority": _data_authority(config, registry),
        "workload_manifest": {
            "path": _relative(workload_path),
            "sha256": _sha256(workload_path),
        },
        "workloads": workloads,
        "required_evidence": {
            "partition": "development",
            "raw_samples_field": "models.<id>.workloads.<name>.end_to_end_latency_ms_samples",
            "aggregate": "geometric mean of per-workload p50 end-to-end latency",
            **config["latency_evidence"],
        },
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


def _build_audit(
    config_path: Path, development_path: Path, latency_path: Path
) -> dict[str, Any]:
    config, registry, incumbent, candidates, dispositions = _collect(
        config_path, development_path
    )
    latencies = _latencies(latency_path, config, registry, incumbent, candidates)
    ranking = _score(config, incumbent, candidates, latencies)
    pair = _pair_status(candidates, config)
    return {
        "schema_version": 1,
        "status": "succeeded",
        "protocol_id": config["protocol_id"],
        "selection_data_scope": "GiftEvalPretrain development only",
        "confirmation_partition_accessed": False,
        "gift_eval_metrics_accessed": False,
        "metric_semantics": config["development_metrics"],
        "score": config["score"],
        "screen_dispositions": dispositions,
        "incumbent": {
            **incumbent,
            "aggregate_end_to_end_latency_ms": latencies["incumbent"],
        },
        "candidate_ranking": ranking,
        "compact_attribution_control": pair,
        "sources": {
            "selection_config": {
                "path": _relative(config_path),
                "sha256": _sha256(config_path),
            },
            "development_evaluation": {
                "path": _relative(development_path),
                "sha256": _sha256(development_path),
            },
            "latency_evidence": {
                "path": _relative(latency_path),
                "sha256": _sha256(latency_path),
            },
        },
        "eligible_to_freeze": True,
    }


def _command_audit(args: argparse.Namespace) -> int:
    payload = _build_audit(
        args.config, args.development_evaluation, args.latency_evidence
    )
    payload["audited_at_utc"] = _now()
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


def _command_freeze(args: argparse.Namespace) -> int:
    audit = _load_json(args.audit)
    if audit.get("schema_version") != 1 or audit.get("status") != "succeeded":
        raise AuditError("screen audit is not successful schema-v1 evidence")
    if audit.get("eligible_to_freeze") is not True:
        raise AuditError("screen audit is not eligible to freeze")
    config = _load_yaml(args.config)
    if audit.get("protocol_id") != config["protocol_id"]:
        raise AuditError("screen audit protocol mismatch")
    if audit.get("confirmation_partition_accessed") is not False:
        raise AuditError("screen audit reports confirmation access")
    if audit.get("gift_eval_metrics_accessed") is not False:
        raise AuditError("screen audit reports GIFT-Eval metric access")
    sources = audit.get("sources", {})
    required_sources = {"selection_config", "development_evaluation", "latency_evidence"}
    if set(sources) != required_sources:
        raise AuditError("screen audit does not contain the exact required source set")
    if sources["selection_config"].get("path") != _relative(args.config):
        raise AuditError("screen audit is bound to a different selection config")
    for source in sources.values():
        path = _path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise AuditError(f"selection source changed: {path}")
    recomputed = _build_audit(
        args.config,
        _path(sources["development_evaluation"]["path"]),
        _path(sources["latency_evidence"]["path"]),
    )
    observed_without_timestamp = dict(audit)
    observed_without_timestamp.pop("audited_at_utc", None)
    if observed_without_timestamp != recomputed:
        raise AuditError("screen audit contents do not match recomputed source evidence")
    count = int(args.count)
    if not args.count_reason.strip():
        raise AuditError("--count-reason must be non-empty")
    maximum = int(config["candidate_scope"]["maximum_finalists"])
    ranking = audit.get("candidate_ranking", [])
    if count < 1 or count > maximum or count > len(ranking):
        raise AuditError(f"finalist count must be between 1 and {min(maximum, len(ranking))}")
    selected = ranking[:count]
    compact_ids = set(config["candidate_scope"]["compact_attribution_pair"])
    if (
        any(row["candidate_id"] in compact_ids for row in selected)
        and audit.get("compact_attribution_control", {}).get("complete") is not True
    ):
        raise AuditError("compact finalist requires successful matched GT and KD screens")
    payload = {
        "schema_version": 1,
        "status": "finalists_frozen",
        "protocol_id": config["protocol_id"],
        "frozen_at_utc": _now(),
        "screen_audit": {"path": _relative(args.audit), "sha256": _sha256(args.audit)},
        "selection_rule": config["score"],
        "finalist_count": count,
        "finalist_count_reason": args.count_reason.strip(),
        "finalists": [
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "variant": row["variant"],
                "screening_score": row["screening_score"],
                "checkpoint": row["checkpoint"],
                "config": row["config"],
                "deployment_fingerprint_sha256": row["deployment_fingerprint_sha256"],
                "model_source": row["model_source"],
                "inference_implementation": row["inference_implementation"],
            }
            for row in selected
        ],
        "confirmation_embargo": {
            "held_through_screen_selection": True,
            "confirmation_partition_accessed_during_selection": False,
            "next_gate": (
                "freeze the complete full-training recipe/seed/checkpoint roster before "
                "the first confirmation-target read"
            ),
            "confirmation_may_not_change": [
                "recipe",
                "training budget",
                "development-selected checkpoint step",
            ],
        },
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_evaluation = subparsers.add_parser(
        "prepare-evaluation", help="bind every selected checkpoint for one common dev pass"
    )
    prepare_evaluation.add_argument("--output", type=Path, required=True)
    prepare_evaluation.set_defaults(function=_command_prepare_evaluation)

    prepare = subparsers.add_parser("prepare", help="emit exact latency checkpoint bindings")
    prepare.add_argument("--development-evaluation", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=_command_prepare)

    audit = subparsers.add_parser("audit", help="audit and rank all successful screens")
    audit.add_argument("--development-evaluation", type=Path, required=True)
    audit.add_argument("--latency-evidence", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(function=_command_audit)

    freeze = subparsers.add_parser("freeze", help="immutably freeze the top one or two recipes")
    freeze.add_argument("--audit", type=Path, required=True)
    freeze.add_argument("--count", type=int, required=True)
    freeze.add_argument("--count-reason", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(function=_command_freeze)

    args = parser.parse_args()
    args.config = args.config.resolve()
    for name in ("development_evaluation", "latency_evidence", "audit", "output"):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    try:
        return int(args.function(args))
    except (AuditError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
