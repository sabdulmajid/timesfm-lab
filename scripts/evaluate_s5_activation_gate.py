#!/usr/bin/env python3
"""Evaluate the frozen v1.2 S5 activation gate from the common step-5000 resumes.

This is a CPU-only metadata audit.  It does not evaluate a model, open benchmark
data, or inspect the confirmation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = Path("configs/performance_recovery/candidates.yaml")
AUDIT = Path(
    "results/reproduction/distillation/performance-recovery-s3-s4-path-audit.json"
)
PLAN = Path("results/reproduction/distillation/production-1m-corpus-plan.json")
TARGETS = Path("configs/performance_recovery/targets.yaml")
CACHE_AUDIT = Path("results/reproduction/distillation/production-1m-cache-audit.json")
SELECTION_SPLIT = Path(
    "results/reproduction/distillation/performance-recovery-selection-split.json"
)
COMPACT_CONFIG = Path("configs/distillation/performance_recovery_compact_screen.yaml")
LEDGER = Path(
    "results/reproduction/distillation/performance-recovery-gpu-hour-ledger-at-s5-gate.json"
)
DEFAULT_OUTPUT = Path(
    "results/reproduction/distillation/performance-recovery-s5-activation-gate.json"
)

# These byte hashes are the pre-metric authorities frozen by protocol v1.2.
EXPECTED_HASHES = {
    "registry": "d4e39f9fdac0818e19c901a882429ec5570c691c27480bc114a4b8a8950abd30",
    "audit": "a1b7901dfa4c447a906e1cc636ea1714d81e4696b68f1c543ec8791f80318f2c",
    "plan": "2cd2967fa0ef4e1f6e7a18d050d7af1b7533458abab96e52b45a4c0caff0828c",
    "targets": "c66f07211515942a0661731e3d295dc256e394c26566dec608e8a2740404d616",
    "cache_audit": "a471d51a48140acf66f8ccffd6920615eda4562474174595ea683b541014449e",
    "selection_split": "9d3e06b328b76baaab558c18717b7336961f07e81261989349c0f4e20c899cd9",
    "compact_config": "a6a9d4c26be88fc0c65bc89075a4458518f019941a7e32b13483fc108fda8749",
    "gate_ledger": "5281ad7cd80e43503c6976b79f0650854f2984f6d6feea099fbc96a1332a408f",
}
TARGET_SHA256 = EXPECTED_HASHES["targets"]
SELECTION_SPLIT_SHA256 = EXPECTED_HASHES["selection_split"]
PREDECESSOR_REGISTRY_SHA256 = (
    "ffef27cada4e3a072fed7e2a7971b7684c94429bb02d69a73dd8c1383e3dbeeb"
)
OUTER_TRAINING_IDENTITY_SHA256 = (
    "24cad944c659cfe437f54b0050dac19fa9698ff45e43ce31907ac8caf824b9f4"
)
LAUNCH_COMMIT = "b3e2fc66f4769a61348374ce5ac62a444e4aab36"
LAUNCH_TRAINER_SHA256 = (
    "a6cf968c3a366bd14aa7a30e163a847d1e3237303a87b2f33f1da368022e3a31"
)

GATE_STEP = 5_000
LOGICAL_BATCH_WINDOWS = 256
GATE_THRESHOLD = 1.03
METRICS = ("student_pinball", "student_normalized_median_mae")
UNDERWEIGHTED_DOMAINS = (
    "Econ/Fin",
    "Energy",
    "Healthcare",
    "Sales",
    "Web/CloudOps",
)
REFERENCE_DOMAINS = ("Nature", "Transport")
SLOTS = {
    "S3": {
        "variant": "compact_gt",
        "checkpoint": Path(
            "checkpoints/performance-recovery/screens/S3/milestones/"
            "student-compact_gt-resume-step5000.pt"
        ),
        "checkpoint_sha256": (
            "7516cd3fac9d1aafe96e33fa5b0de7e3eb713c73a4f8b32fe8101c7230113cc4"
        ),
        "launch": Path(
            "results/reproduction/distillation/performance-recovery-launch-S3-attempt02.json"
        ),
        "launch_sha256": (
            "377f9838f15ce410d212bcd10ee5faa833797d9a6329f4a27394e2088660e284"
        ),
        "objective": {
            "ground_truth": 1.0,
            "multivariate_kd": 0.0,
            "univariate_kd": 0.0,
            "cvrd": 0.0,
        },
    },
    "S4": {
        "variant": "compact_kd4",
        "checkpoint": Path(
            "checkpoints/performance-recovery/screens/S4/milestones/"
            "student-compact_kd4-resume-step5000.pt"
        ),
        "checkpoint_sha256": (
            "5aa7aec95f1ed7009cf420c0b33cb3773f2ac87db90176cfa917f8093ff5680e"
        ),
        "launch": Path(
            "results/reproduction/distillation/performance-recovery-launch-S4-attempt02.json"
        ),
        "launch_sha256": (
            "a0f5bb2458586aad5e3b39db11295b7dfec249799ef293c6495c3fe2ca384bc0"
        ),
        "objective": {
            "ground_truth": 1.0,
            "multivariate_kd": 4.0,
            "univariate_kd": 0.0,
            "cvrd": 0.0,
        },
    },
}


class GateError(RuntimeError):
    """A frozen gate invariant was not satisfied."""


def _repo_path(path: Path) -> Path:
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise GateError(f"path escapes repository: {resolved}") from error
    return resolved


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be an integer")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GateError(f"{label} must be finite and positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{label} must be finite and positive") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise GateError(f"{label} must be finite and positive, got {parsed}")
    return parsed


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GateError(f"{label} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{label} must be finite and nonnegative") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise GateError(f"{label} must be finite and nonnegative")
    return parsed


def _geometric_mean(values: list[float], label: str) -> float:
    if not values:
        raise GateError(f"{label} has no values")
    return math.exp(math.fsum(math.log(_positive(value, label)) for value in values) / len(values))


def _require_close(observed: Any, expected: float, label: str) -> float:
    value = _positive(observed, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0):
        raise GateError(f"{label} does not recompute: observed={value}, expected={expected}")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a YAML mapping")
    return value


def _read_hashed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise GateError(f"{label} changed while it was read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _launch_argument(command: list[Any], flag: str) -> str:
    if command.count(flag) != 1:
        raise GateError(f"launch command does not contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise GateError(f"launch command has no value for {flag}")
    return str(command[index + 1])


def _validate_launches(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    protocol = _mapping(registry["protocol"], "registry.protocol")
    allowlist = {
        (str(row["candidate_id"]), _integer(row["attempt"], "launch attempt")): row
        for row in protocol.get("grandfathered_launch_records", [])
    }
    candidates = _mapping(registry["candidates"], "registry.candidates")
    trainer = _mapping(registry["trainer"], "registry.trainer")
    summaries: dict[str, dict[str, Any]] = {}
    for slot, specification in SLOTS.items():
        path = _repo_path(specification["launch"])
        launch, digest = _read_hashed_json(path, f"{slot} launch record")
        if digest != specification["launch_sha256"]:
            raise GateError(f"{slot} attempt-2 launch-record hash changed")
        allowlisted = _mapping(allowlist.get((slot, 2)), f"registry allowlist for {slot}")
        if allowlisted.get("path") != str(specification["launch"]):
            raise GateError(f"{slot} attempt-2 launch path is not grandfathered")
        if allowlisted.get("sha256") != digest:
            raise GateError(f"{slot} attempt-2 launch hash is not grandfathered")
        candidate = _mapping(candidates[slot], f"registry candidate {slot}")
        if (
            launch.get("schema_version") != 1
            or launch.get("candidate_id") != slot
            or launch.get("variant") != specification["variant"]
            or launch.get("protocol_id") != "timesfm3-performance-recovery-v1.1"
            or launch.get("git_commit") != LAUNCH_COMMIT
            or launch.get("selection_partition") != "development"
            or launch.get("gift_evaluation") is not False
            or launch.get("resumed") is not False
            or launch.get("resume_checkpoint_sha256") is not None
            or launch.get("initialization") != candidate.get("initialization")
        ):
            raise GateError(f"{slot} attempt-2 launch identity changed")
        if int(launch.get("step_ceiling", -1)) != 50_000:
            raise GateError(f"{slot} attempt-2 launch step ceiling changed")

        command = launch.get("command")
        if not isinstance(command, list) or len(command) < 4:
            raise GateError(f"{slot} launch command is malformed")
        if command[:4] != [
            trainer["python"],
            trainer["path"],
            str(COMPACT_CONFIG),
            str(PLAN),
        ]:
            raise GateError(f"{slot} launch executable/config/plan changed")
        expected_arguments = {
            "--variant": specification["variant"],
            "--training-seed": "42",
            "--split-seed": "42",
            "--checkpoint-dir": str(candidate["checkpoint_dir"]),
            "--output": str(candidate["output"]),
            "--max-steps": "50000",
            "--selection-split-manifest": str(SELECTION_SPLIT),
            "--validation-partition": "development",
        }
        for flag, expected in expected_arguments.items():
            if _launch_argument(command, flag) != expected:
                raise GateError(f"{slot} launch argument {flag} changed")
        if "--resume" in command or "--initialize-from" in command or "--distributed" in command:
            raise GateError(f"{slot} attempt-2 launch was not fresh single-GPU training")

        rows = launch.get("input_hashes")
        if not isinstance(rows, list) or not rows:
            raise GateError(f"{slot} launch has no input hashes")
        input_hashes: dict[str, str] = {}
        for raw_row in rows:
            row = _mapping(raw_row, f"{slot} launch input")
            relative = str(row.get("path", ""))
            expected = str(row.get("sha256", ""))
            if (
                not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or len(expected) != 64
                or relative in input_hashes
            ):
                raise GateError(f"{slot} launch contains an invalid input-hash row")
            observed = hashlib.sha256(_git("show", f"{LAUNCH_COMMIT}:{relative}")).hexdigest()
            if observed != expected:
                raise GateError(f"{slot} launch input differs from its Git snapshot: {relative}")
            input_hashes[relative] = expected
        required = {
            str(REGISTRY): PREDECESSOR_REGISTRY_SHA256,
            str(TARGETS): EXPECTED_HASHES["targets"],
            str(PLAN): EXPECTED_HASHES["plan"],
            str(CACHE_AUDIT): EXPECTED_HASHES["cache_audit"],
            str(SELECTION_SPLIT): EXPECTED_HASHES["selection_split"],
            str(COMPACT_CONFIG): EXPECTED_HASHES["compact_config"],
            str(trainer["path"]): LAUNCH_TRAINER_SHA256,
        }
        if any(input_hashes.get(name) != expected for name, expected in required.items()):
            raise GateError(f"{slot} launch does not bind every required frozen input")
        summaries[slot] = {
            "path": str(specification["launch"]),
            "sha256": digest,
            "git_commit": LAUNCH_COMMIT,
            "command": command,
            "initialization": launch["initialization"],
            "input_hashes": rows,
        }
    resolved = _git("rev-parse", "--verify", f"{LAUNCH_COMMIT}^{{commit}}").decode().strip()
    if resolved != LAUNCH_COMMIT:
        raise GateError("launch Git commit does not resolve exactly")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LAUNCH_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode:
        raise GateError("launch Git commit is not an ancestor of HEAD")
    return summaries


def _validate_ledger(launches: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    path = _repo_path(LEDGER)
    ledger, digest = _read_hashed_json(path, "GPU-hour ledger")
    if digest != EXPECTED_HASHES["gate_ledger"]:
        raise GateError("frozen gate-time GPU-hour ledger hash changed")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("protocol_id") != "timesfm3-performance-recovery-v1.1"
        or ledger.get("registry_sha256") != PREDECESSOR_REGISTRY_SHA256
        or float(ledger.get("hard_cap_physical_gpu_hours", math.nan)) != 120.0
    ):
        raise GateError("live GPU-hour ledger authority changed")
    states = _mapping(ledger.get("candidates"), "ledger.candidates")
    gpu_ids: set[int] = set()
    retained: dict[str, Any] = {}
    for slot in SLOTS:
        state = _mapping(states.get(slot), f"ledger.{slot}")
        attempts = state.get("attempts")
        if state.get("status") != "running" or not isinstance(attempts, list) or len(attempts) != 2:
            raise GateError(f"ledger {slot} is not the running second attempt")
        latest = _mapping(attempts[-1], f"ledger.{slot}.attempt2")
        if (
            _integer(latest.get("attempt"), f"ledger {slot} attempt") != 2
            or latest.get("status") != "running"
            or latest.get("launch_record") != launches[slot]["path"]
        ):
            raise GateError(f"ledger {slot} attempt-2 binding changed")
        gpu = _integer(latest.get("physical_gpu"), f"ledger {slot} physical GPU")
        if gpu in gpu_ids:
            raise GateError("S3 and S4 ledger attempts share a physical GPU")
        gpu_ids.add(gpu)
        retained[slot] = state
    s5 = _mapping(states.get("S5"), "ledger.S5")
    if s5.get("status") != "declared" or s5.get("attempts") != []:
        raise GateError("S5 was already launched before its activation gate")
    retained["S5"] = s5
    return {
        "path": str(LEDGER),
        "sha256_at_gate_evaluation": digest,
        "protocol_id": ledger["protocol_id"],
        "registry_sha256": ledger["registry_sha256"],
        "hard_cap_physical_gpu_hours": ledger["hard_cap_physical_gpu_hours"],
        "accounting": ledger.get("accounting"),
        "candidate_states": retained,
    }, digest


def _validate_authorities() -> tuple[dict[str, Any], dict[str, str], int]:
    paths = {
        "registry": _repo_path(REGISTRY),
        "audit": _repo_path(AUDIT),
        "plan": _repo_path(PLAN),
        "targets": _repo_path(TARGETS),
        "cache_audit": _repo_path(CACHE_AUDIT),
        "selection_split": _repo_path(SELECTION_SPLIT),
        "compact_config": _repo_path(COMPACT_CONFIG),
    }
    inputs: dict[str, str] = {}
    for name, path in paths.items():
        observed = _sha256(path)
        if observed != EXPECTED_HASHES[name]:
            raise GateError(
                f"frozen {name} hash mismatch: observed={observed}, "
                f"expected={EXPECTED_HASHES[name]}"
            )
        inputs[name] = observed

    registry = _load_yaml(paths["registry"], "registry")
    audit = _load_json(paths["audit"], "audit")
    plan = _load_json(paths["plan"], "plan")
    targets = _load_yaml(paths["targets"], "targets")
    cache_audit = _load_json(paths["cache_audit"], "cache audit")
    split = _load_json(paths["selection_split"], "selection split")
    compact_config = _load_yaml(paths["compact_config"], "compact config")
    if registry.get("schema_version") != 1:
        raise GateError("unexpected registry schema")
    protocol = _mapping(registry.get("protocol"), "registry.protocol")
    if protocol.get("id") != "timesfm3-performance-recovery-v1.2":
        raise GateError("registry is not the frozen v1.2 protocol")
    target = _mapping(protocol.get("target_authority"), "registry target authority")
    if target.get("sha256") != TARGET_SHA256 or target.get("policy") != "unchanged_by_v1.2":
        raise GateError("registry target authority changed")
    if targets.get("schema_version") != 1 or targets.get("protocol_id") != target.get(
        "protocol_id"
    ):
        raise GateError("target file is not the registry-declared authority")

    corpus = _mapping(registry.get("corpus"), "registry.corpus")
    if corpus.get("plan") != str(PLAN) or corpus.get("plan_sha256") != EXPECTED_HASHES["plan"]:
        raise GateError("registry corpus-plan binding changed")
    selection = _mapping(registry.get("selection_split"), "registry.selection_split")
    if selection.get("manifest_sha256") != SELECTION_SPLIT_SHA256:
        raise GateError("registry selection-split binding changed")
    if corpus.get("cache_audit") != str(CACHE_AUDIT) or corpus.get(
        "cache_audit_sha256"
    ) != EXPECTED_HASHES["cache_audit"]:
        raise GateError("registry cache-audit binding changed")
    if (
        split.get("schema_version") != 1
        or split.get("protocol_id") != target.get("protocol_id")
        or split.get("status") != "frozen_uninspected"
        or split.get("target_accessed") is not False
        or split.get("teacher_output_accessed") is not False
        or _mapping(split.get("source"), "split.source").get("plan_sha256")
        != EXPECTED_HASHES["plan"]
    ):
        raise GateError("selection split is not frozen development-only authority")
    split_outer = _mapping(
        _mapping(split.get("totals"), "split.totals").get("outer_training"),
        "split outer-training total",
    )
    if (
        split_outer.get("identity_sha256") != OUTER_TRAINING_IDENTITY_SHA256
        or _integer(split_outer.get("count"), "split outer-training count") != 927_346
    ):
        raise GateError("selection split outer-training identity changed")

    candidates = _mapping(registry.get("candidates"), "registry.candidates")
    s5 = _mapping(candidates.get("S5"), "registry.candidates.S5")
    expected_base = {
        "checkpoint_step": GATE_STEP,
        "candidates": ["S3", "S4"],
        "metric": "balanced_forecast_error",
        "direction": "lower",
        "tie_breaker": "S3",
        "preserve_selected_objective_exactly": True,
    }
    if s5.get("base_selection") != expected_base or s5.get("launchable") is not False:
        raise GateError("registry S5 base selection or launch state changed")
    if s5.get("single_changed_factor") != "per_window_domain_balanced_training_reduction":
        raise GateError("registry S5 changed-factor declaration changed")

    weight_source = _mapping(s5.get("domain_weights_source"), "S5 domain weights")
    if (
        weight_source.get("outer_training_identity_sha256")
        != OUTER_TRAINING_IDENTITY_SHA256
        or float(s5.get("maximum_estimated_physical_gpu_hours", math.nan)) != 29.1
    ):
        raise GateError("S5 training identity or cost ceiling changed")
    counts = _mapping(
        weight_source.get("counts_include_zero_target_windows"),
        "S5 outer-training domain counts",
    )
    expected_domains = set(UNDERWEIGHTED_DOMAINS) | set(REFERENCE_DOMAINS)
    if set(counts) != expected_domains:
        raise GateError("S5 domain-count names differ from the seven frozen domains")
    outer_training_windows = sum(
        _integer(counts[domain], f"S5 count for {domain}") for domain in expected_domains
    )
    if outer_training_windows != _integer(
        split_outer.get("count"), "split outer-training count"
    ):
        raise GateError("registry and selection-split training-window totals differ")

    training = _mapping(compact_config.get("training"), "compact config training")
    expected_training = {
        "max_steps": 50_000,
        "logical_batch_size_windows": LOGICAL_BATCH_WINDOWS,
        "validate_at_start": True,
        "validation_every_steps": GATE_STEP,
        "checkpoint_every_steps": GATE_STEP,
        "validation_selection_metric": "balanced_forecast_error",
    }
    if any(training.get(key) != expected for key, expected in expected_training.items()):
        raise GateError("compact config no longer yields the exact first-common gate records")
    loss_weights = _mapping(training.get("loss_weights"), "compact config loss weights")
    for slot, specification in SLOTS.items():
        if loss_weights.get(specification["variant"]) != specification["objective"]:
            raise GateError(f"compact config objective changed for {slot}")

    if audit.get("schema_version") != 1 or audit.get("status") != "completed":
        raise GateError("production-path audit is not completed schema-v1 evidence")
    fingerprints = _mapping(audit.get("fingerprints"), "audit.fingerprints")
    if fingerprints.get(str(PLAN)) != EXPECTED_HASHES["plan"]:
        raise GateError("audit corpus-plan binding changed")
    gradient_mass = _mapping(
        audit.get("exact_training_gradient_mass"), "audit exact training gradient mass"
    )
    audit_total = _integer(
        _mapping(gradient_mass.get("totals"), "audit gradient totals").get("windows"),
        "audit outer-training windows",
    )
    if audit_total != outer_training_windows:
        raise GateError("registry and audit outer-training window totals differ")
    audit_domains = _mapping(gradient_mass.get("by_domain"), "audit gradient domains")
    if set(audit_domains) != expected_domains:
        raise GateError("audit and registry domain names differ")
    for domain in expected_domains:
        audit_windows = _integer(
            _mapping(audit_domains[domain], f"audit domain {domain}").get("windows"),
            f"audit windows for {domain}",
        )
        if audit_windows != counts[domain]:
            raise GateError(f"registry/audit outer-training count differs for {domain}")
    optimization = _mapping(audit.get("optimization_evidence"), "audit optimization evidence")
    step34 = _mapping(
        optimization.get("exact_fixed_step34_batch_at_seeded_initialization"),
        "audit step-34 evidence",
    )
    if float(step34.get("gradient_clip_threshold", math.nan)) != 1.0:
        raise GateError("audit gradient-clip threshold changed")

    if plan.get("dataset_revision") != corpus.get("dataset_revision"):
        raise GateError("plan and registry dataset revisions differ")
    cache_rows = cache_audit.get("datasets")
    if (
        cache_audit.get("dataset_revision") != corpus.get("dataset_revision")
        or not isinstance(cache_rows, list)
        or len(cache_rows) != int(corpus.get("datasets", -1))
    ):
        raise GateError("cache audit does not match the frozen corpus authority")
    plan_rows = plan.get("datasets")
    if not isinstance(plan_rows, list) or len(plan_rows) != int(corpus.get("datasets", -1)):
        raise GateError("plan does not contain the registry-declared dataset count")
    dataset_domains: dict[str, str] = {}
    true_mv: set[str] = set()
    requested_windows = 0
    for index, raw_row in enumerate(plan_rows):
        row = _mapping(raw_row, f"plan.datasets[{index}]")
        dataset = str(row.get("dataset", ""))
        domain = str(row.get("domain", ""))
        if not dataset or dataset in dataset_domains or domain not in expected_domains:
            raise GateError(f"invalid or duplicate plan dataset/domain at index {index}")
        dataset_domains[dataset] = domain
        requested_windows += _integer(row.get("requested_windows"), f"{dataset}.requested_windows")
        if row.get("view_class") == "true_multivariate":
            true_mv.add(dataset)
        elif row.get("view_class") != "univariate":
            raise GateError(f"{dataset} has an unknown view class")
    if requested_windows != int(corpus.get("windows", -1)) or len(true_mv) != 11:
        raise GateError("plan window or true-multivariate count changed")
    cache_domains = {
        str(_mapping(row, "cache-audit dataset").get("dataset")): str(
            _mapping(row, "cache-audit dataset").get("domain")
        )
        for row in cache_rows
    }
    if cache_domains != dataset_domains:
        raise GateError("cache-audit dataset/domain mapping differs from the corpus plan")
    authority = {
        "domains": dataset_domains,
        "true_mv": true_mv,
        "registry": registry,
        "domain_weight_source": dict(weight_source),
        "maximum_estimated_physical_gpu_hours": s5["maximum_estimated_physical_gpu_hours"],
    }
    return authority, inputs, outer_training_windows


def _validate_metrics(
    value: Any,
    *,
    label: str,
    dataset_domains: dict[str, str],
    true_mv: set[str],
) -> dict[str, Any]:
    validation = _mapping(value, label)
    by_dataset = _mapping(validation.get("by_dataset"), f"{label}.by_dataset")
    if set(by_dataset) != set(dataset_domains):
        missing = sorted(set(dataset_domains) - set(by_dataset))
        extra = sorted(set(by_dataset) - set(dataset_domains))
        raise GateError(f"{label} dataset set mismatch: missing={missing}, extra={extra}")

    metrics: dict[str, dict[str, float | int]] = {}
    for dataset in sorted(dataset_domains):
        row = _mapping(by_dataset[dataset], f"{label}.{dataset}")
        metrics[dataset] = {
            metric: _positive(row.get(metric), f"{label}.{dataset}.{metric}")
            for metric in METRICS
        }
        metrics[dataset]["windows"] = _integer(row.get("windows"), f"{label}.{dataset}.windows")
        metrics[dataset]["observed_targets"] = _integer(
            row.get("observed_targets"), f"{label}.{dataset}.observed_targets"
        )
        if metrics[dataset]["windows"] <= 0 or metrics[dataset]["observed_targets"] <= 0:
            raise GateError(f"{label}.{dataset} has an empty development record")

    balanced = _mapping(validation.get("balanced"), f"{label}.balanced")
    if _integer(balanced.get("dataset_count"), f"{label}.dataset_count") != len(metrics):
        raise GateError(f"{label} balanced dataset count changed")
    if _integer(balanced.get("true_mv_dataset_count"), f"{label}.true_mv_dataset_count") != len(
        true_mv
    ):
        raise GateError(f"{label} balanced true-MV dataset count changed")
    if balanced.get("aggregation") != "unweighted geometric mean across datasets":
        raise GateError(f"{label} balanced aggregation changed")

    overall_pinball = _geometric_mean(
        [float(metrics[name]["student_pinball"]) for name in metrics],
        f"{label} overall pinball",
    )
    overall_mae = _geometric_mean(
        [float(metrics[name]["student_normalized_median_mae"]) for name in metrics],
        f"{label} overall median MAE",
    )
    true_mv_pinball = _geometric_mean(
        [float(metrics[name]["student_pinball"]) for name in sorted(true_mv)],
        f"{label} true-MV pinball",
    )
    true_mv_mae = _geometric_mean(
        [float(metrics[name]["student_normalized_median_mae"]) for name in sorted(true_mv)],
        f"{label} true-MV median MAE",
    )
    balanced_values = {
        "student_pinball": _require_close(
            balanced.get("student_pinball"), overall_pinball, f"{label}.student_pinball"
        ),
        "student_normalized_median_mae": _require_close(
        balanced.get("student_normalized_median_mae"),
        overall_mae,
        f"{label}.student_normalized_median_mae",
        ),
        "true_mv_student_pinball": _require_close(
        balanced.get("true_mv_student_pinball"),
        true_mv_pinball,
        f"{label}.true_mv_student_pinball",
        ),
        "true_mv_student_normalized_median_mae": _require_close(
        balanced.get("true_mv_student_normalized_median_mae"),
        true_mv_mae,
        f"{label}.true_mv_student_normalized_median_mae",
        ),
    }
    expected_score = math.exp(
        math.fsum(
            (
                math.log(overall_mae) / 3,
                math.log(overall_pinball) / 3,
                math.log(true_mv_mae) / 6,
                math.log(true_mv_pinball) / 6,
            )
        )
    )
    score = _require_close(
        balanced.get("forecast_error"), expected_score, f"{label}.balanced.forecast_error"
    )
    balanced_values["forecast_error"] = score
    identity = {
        name: {
            "windows": metrics[name]["windows"],
            "observed_targets": metrics[name]["observed_targets"],
        }
        for name in metrics
    }
    return {
        "forecast_error": score,
        "balanced": balanced_values,
        "by_dataset": metrics,
        "identity": identity,
    }


def _checkpoint_snapshot(
    slot: str,
    path: Path,
    *,
    expected_sha256: str,
    dataset_domains: dict[str, str],
    true_mv: set[str],
    outer_training_windows: int,
) -> dict[str, Any]:
    if path.is_symlink():
        raise GateError(f"{slot} resume checkpoint must not be a symlink")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise GateError(f"{slot} resume checkpoint is not a regular file")
    digest = _sha256(path)
    if digest != expected_sha256:
        raise GateError(
            f"{slot} immutable step-5000 checkpoint hash mismatch: "
            f"observed={digest}, expected={expected_sha256}"
        )
    hashed = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        hashed.st_dev,
        hashed.st_ino,
        hashed.st_size,
        hashed.st_mtime_ns,
    ):
        raise GateError(f"{slot} resume checkpoint changed while it was hashed")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise GateError(f"{slot} resume checkpoint changed while it was loaded")
    state = _mapping(state, f"{slot} resume checkpoint")
    if not _mapping(state.get("model"), f"{slot}.model"):
        raise GateError(f"{slot} resume checkpoint has no model state")
    if not _mapping(state.get("optimizer"), f"{slot}.optimizer"):
        raise GateError(f"{slot} resume checkpoint has no optimizer state")

    total_batches = math.ceil(outer_training_windows / LOGICAL_BATCH_WINDOWS)
    expected_epoch, expected_offset = divmod(GATE_STEP, total_batches)
    expected_windows = expected_epoch * outer_training_windows + min(
        expected_offset * LOGICAL_BATCH_WINDOWS, outer_training_windows
    )
    expected_tail = outer_training_windows % LOGICAL_BATCH_WINDOWS or LOGICAL_BATCH_WINDOWS
    exact = {
        "step": GATE_STEP,
        "epoch": expected_epoch,
        "batch_offset": expected_offset,
        "trained_windows": expected_windows,
        "logical_batch_size_windows": LOGICAL_BATCH_WINDOWS,
        "training_seed": 42,
        "split_seed": 42,
    }
    for key, expected in exact.items():
        if _integer(state.get(key), f"{slot}.{key}") != expected:
            raise GateError(f"{slot}.{key} differs from frozen step-5000 state: {state.get(key)}")
    if state.get("selection_split_manifest_sha256") != SELECTION_SPLIT_SHA256:
        raise GateError(f"{slot} selection-split hash mismatch")
    if state.get("validation_partition") != "development":
        raise GateError(f"{slot} resume did not use the development partition")
    sequence = str(state.get("training_sequence_sha256", ""))
    if len(sequence) != 64 or sequence == bytes(32).hex():
        raise GateError(f"{slot} has an invalid training-sequence hash")
    try:
        bytes.fromhex(sequence)
    except ValueError as error:
        raise GateError(f"{slot} has a non-hex training-sequence hash") from error

    optimizer_sum = _integer(
        state.get("optimizer_batch_windows_sum"), f"{slot}.optimizer_batch_windows_sum"
    )
    physical_sum = _integer(
        state.get("physical_microbatch_windows_sum"),
        f"{slot}.physical_microbatch_windows_sum",
    )
    if optimizer_sum != expected_windows or physical_sum != expected_windows:
        raise GateError(f"{slot} processed-window counters are inconsistent")
    optimizer_min = _integer(
        state.get("optimizer_batch_windows_min"), f"{slot}.optimizer_batch_windows_min"
    )
    if optimizer_min != expected_tail:
        raise GateError(f"{slot} optimizer-batch minimum changed")
    optimizer_max = _integer(
        state.get("optimizer_batch_windows_max"), f"{slot}.optimizer_batch_windows_max"
    )
    if optimizer_max != LOGICAL_BATCH_WINDOWS:
        raise GateError(f"{slot} optimizer-batch maximum changed")

    curve = state.get("learning_curve")
    if not isinstance(curve, list) or len(curve) != 2:
        raise GateError(f"{slot} must contain exactly the step-0 and step-5000 records")
    curve_steps = [
        _integer(_mapping(row, f"{slot}.curve").get("step"), f"{slot}.curve.step")
        for row in curve
    ]
    if curve_steps != [0, GATE_STEP]:
        raise GateError(f"{slot} learning curve is not exactly [0, 5000]")

    snapshots: dict[int, dict[str, Any]] = {}
    for expected_step, raw_row in zip((0, GATE_STEP), curve, strict=True):
        row = _mapping(raw_row, f"{slot}.step{expected_step}")
        expected_row_windows = 0 if expected_step == 0 else expected_windows
        expected_row_epoch = 0 if expected_step == 0 else expected_epoch
        row_exact = {
            "step": expected_step,
            "epoch": expected_row_epoch,
            "windows_processed": expected_row_windows,
            "optimizer_batches_processed": expected_step,
        }
        for key, expected in row_exact.items():
            if _integer(row.get(key), f"{slot}.step{expected_step}.{key}") != expected:
                raise GateError(f"{slot} step-{expected_step} {key} is not exact")
        observed_targets = _integer(
            row.get("observed_targets_processed"),
            f"{slot}.step{expected_step}.observed_targets_processed",
        )
        physical_microbatches = _integer(
            row.get("physical_microbatches_processed"),
            f"{slot}.step{expected_step}.physical_microbatches_processed",
        )
        if expected_step == 0 and (observed_targets != 0 or physical_microbatches != 0):
            raise GateError(f"{slot} step-zero counters are not zero")
        metrics = _validate_metrics(
            row.get("validation"),
            label=f"{slot}.step{expected_step}.validation",
            dataset_domains=dataset_domains,
            true_mv=true_mv,
        )
        snapshots[expected_step] = {
            "step": expected_step,
            "epoch": expected_row_epoch,
            "windows_processed": expected_row_windows,
            "observed_targets_processed": observed_targets,
            "optimizer_batches_processed": expected_step,
            "physical_microbatches_processed": physical_microbatches,
            **metrics,
        }

    if snapshots[GATE_STEP]["observed_targets_processed"] != _integer(
        state.get("observed_targets_processed"), f"{slot}.observed_targets_processed"
    ):
        raise GateError(f"{slot} observed-target counter differs from its step-5000 record")
    if snapshots[GATE_STEP]["physical_microbatches_processed"] != _integer(
        state.get("physical_microbatches_processed"), f"{slot}.physical_microbatches_processed"
    ):
        raise GateError(f"{slot} microbatch counter differs from its step-5000 record")
    train_weight_sum = _nonnegative(state.get("train_weight_sum"), f"{slot}.train_weight_sum")
    if train_weight_sum != snapshots[GATE_STEP]["observed_targets_processed"]:
        raise GateError(f"{slot} training weight differs from its observed-target count")
    best_score = _positive(state.get("best_score"), f"{slot}.best_score")
    expected_best = min(
        snapshots[0]["forecast_error"], snapshots[GATE_STEP]["forecast_error"]
    )
    if not math.isclose(best_score, expected_best, rel_tol=1e-12, abs_tol=0.0):
        raise GateError(f"{slot} best score is inconsistent with its two exact records")

    gradient_sum = _nonnegative(state.get("gradient_norm_sum"), f"{slot}.gradient_norm_sum")
    clipped_steps = _integer(state.get("gradient_clip_count"), f"{slot}.gradient_clip_count")
    if not 0 <= clipped_steps <= GATE_STEP:
        raise GateError(f"{slot} gradient clip count is out of range")
    trajectory = {
        "step": GATE_STEP,
        "epoch": expected_epoch,
        "batch_offset": expected_offset,
        "windows_processed": expected_windows,
        "observed_targets_processed": snapshots[GATE_STEP]["observed_targets_processed"],
        "physical_microbatches_processed": snapshots[GATE_STEP][
            "physical_microbatches_processed"
        ],
        "physical_microbatch_windows_sum": physical_sum,
        "physical_microbatch_windows_min": _integer(
            state.get("physical_microbatch_windows_min"),
            f"{slot}.physical_microbatch_windows_min",
        ),
        "physical_microbatch_windows_max": _integer(
            state.get("physical_microbatch_windows_max"),
            f"{slot}.physical_microbatch_windows_max",
        ),
        "optimizer_batch_windows_sum": optimizer_sum,
        "optimizer_batch_windows_min": optimizer_min,
        "optimizer_batch_windows_max": optimizer_max,
        "train_weight_sum": train_weight_sum,
        "training_sequence_sha256": sequence,
    }
    return {
        "path": _relative(path),
        "sha256": digest,
        "size_bytes": before.st_size,
        "trajectory": trajectory,
        "steps": snapshots,
        "optimization": {
            "gradient_clip_threshold": 1.0,
            "clipped_steps": clipped_steps,
            "clip_fraction": clipped_steps / GATE_STEP,
            "mean_preclip_gradient_norm": gradient_sum / GATE_STEP,
        },
    }


def _domain_ratios(
    selected: dict[str, Any], dataset_domains: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    step0 = selected["steps"][0]["by_dataset"]
    step5000 = selected["steps"][GATE_STEP]["by_dataset"]
    per_dataset: dict[str, Any] = {}
    domain_values: dict[str, dict[str, list[float]]] = {
        domain: {metric: [] for metric in METRICS}
        for domain in (*UNDERWEIGHTED_DOMAINS, *REFERENCE_DOMAINS)
    }
    for dataset in sorted(dataset_domains):
        domain = dataset_domains[dataset]
        row: dict[str, Any] = {"domain": domain}
        for metric in METRICS:
            before = float(step0[dataset][metric])
            after = float(step5000[dataset][metric])
            ratio = after / before
            _positive(ratio, f"{dataset}.{metric} ratio")
            row[metric] = {"step0": before, "step5000": after, "ratio": ratio}
            domain_values[domain][metric].append(ratio)
        per_dataset[dataset] = row

    by_domain: dict[str, Any] = {}
    for domain in (*UNDERWEIGHTED_DOMAINS, *REFERENCE_DOMAINS):
        metric_ratios = {
            metric: _geometric_mean(domain_values[domain][metric], f"{domain}.{metric}")
            for metric in METRICS
        }
        all_ratios = [
            ratio for metric in METRICS for ratio in domain_values[domain][metric]
        ]
        by_domain[domain] = {
            "dataset_count": len(domain_values[domain][METRICS[0]]),
            "metric_ratios": metric_ratios,
            "combined_ratio": _geometric_mean(all_ratios, f"{domain} combined ratio"),
        }
    return per_dataset, by_domain


def _group_ratios(by_domain: dict[str, Any]) -> dict[str, Any]:
    underweighted = _geometric_mean(
        [float(by_domain[domain]["combined_ratio"]) for domain in UNDERWEIGHTED_DOMAINS],
        "underweighted-domain group ratio",
    )
    reference = _geometric_mean(
        [float(by_domain[domain]["combined_ratio"]) for domain in REFERENCE_DOMAINS],
        "Nature+Transport group ratio",
    )
    return {
        "underweighted": {
            "domains": list(UNDERWEIGHTED_DOMAINS),
            "geometric_mean_ratio": underweighted,
        },
        "nature_and_transport": {
            "domains": list(REFERENCE_DOMAINS),
            "geometric_mean_ratio": reference,
        },
        "underweighted_over_nature_and_transport": underweighted / reference,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
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
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _evaluate(output: Path) -> dict[str, Any]:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")
    authority, authority_hashes, outer_training_windows = _validate_authorities()
    registry = authority["registry"]
    launches = _validate_launches(registry)
    ledger, _ = _validate_ledger(launches)
    candidates = _mapping(registry["candidates"], "registry.candidates")
    snapshots: dict[str, dict[str, Any]] = {}
    for slot, specification in SLOTS.items():
        candidate = _mapping(candidates.get(slot), f"registry candidate {slot}")
        if candidate.get("variant") != specification["variant"]:
            raise GateError(f"{slot} variant changed")
        checkpoint = _repo_path(specification["checkpoint"])
        expected_parent = _repo_path(Path(str(candidate["checkpoint_dir"])) / "milestones")
        if checkpoint.parent != expected_parent:
            raise GateError(f"{slot} milestone checkpoint is outside its registry directory")
        snapshots[slot] = _checkpoint_snapshot(
            slot,
            checkpoint,
            expected_sha256=str(specification["checkpoint_sha256"]),
            dataset_domains=authority["domains"],
            true_mv=authority["true_mv"],
            outer_training_windows=outer_training_windows,
        )

    identities = {
        slot: {
            step: snapshot["steps"][step]["identity"] for step in (0, GATE_STEP)
        }
        for slot, snapshot in snapshots.items()
    }
    reference_identity = identities["S3"][0]
    if any(
        identity != reference_identity
        for rows in identities.values()
        for identity in rows.values()
    ):
        raise GateError("S3/S4 step-0/step-5000 development windows are not identical")
    if snapshots["S3"]["trajectory"] != snapshots["S4"]["trajectory"]:
        raise GateError("S3/S4 step-5000 training windows, order, or counters differ")

    scores = {
        slot: float(snapshot["steps"][GATE_STEP]["forecast_error"])
        for slot, snapshot in snapshots.items()
    }
    selected_slot = "S3" if scores["S3"] <= scores["S4"] else "S4"
    tie = scores["S3"] == scores["S4"]
    improvements = {
        slot: {
            "step0": float(snapshot["steps"][0]["forecast_error"]),
            "step5000": float(snapshot["steps"][GATE_STEP]["forecast_error"]),
            "step5000_over_step0": float(snapshot["steps"][GATE_STEP]["forecast_error"])
            / float(snapshot["steps"][0]["forecast_error"]),
            "improved": float(snapshot["steps"][GATE_STEP]["forecast_error"])
            < float(snapshot["steps"][0]["forecast_error"]),
        }
        for slot, snapshot in snapshots.items()
    }
    improvement_passes = any(bool(row["improved"]) for row in improvements.values())

    ratio_analyses: dict[str, dict[str, Any]] = {}
    for slot, snapshot in snapshots.items():
        per_dataset, by_domain = _domain_ratios(snapshot, authority["domains"])
        ratio_analyses[slot] = {
            "per_dataset": per_dataset,
            "by_domain": by_domain,
            "groups": _group_ratios(by_domain),
        }
    selected_groups = ratio_analyses[selected_slot]["groups"]
    skew_ratio = float(selected_groups["underweighted_over_nature_and_transport"])
    domain_gate_passes = skew_ratio >= GATE_THRESHOLD
    activate = improvement_passes and domain_gate_passes

    evaluator = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "protocol_id": "timesfm3-performance-recovery-v1.2",
        "gate_id": "S5",
        "status": "completed",
        "evaluated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "scope": "development-only resume-checkpoint metadata; no model execution or GPU work",
        "inputs": {
            "evaluator": {"path": _relative(evaluator), "sha256": _sha256(evaluator)},
            "frozen_authorities": {
                "registry": {
                    "path": str(REGISTRY),
                    "sha256": authority_hashes["registry"],
                },
                "production_path_audit": {
                    "path": str(AUDIT),
                    "sha256": authority_hashes["audit"],
                },
                "production_corpus_plan": {
                    "path": str(PLAN),
                    "sha256": authority_hashes["plan"],
                },
                "target_authority": {
                    "path": str(TARGETS),
                    "sha256": authority_hashes["targets"],
                },
                "production_cache_audit": {
                    "path": str(CACHE_AUDIT),
                    "sha256": authority_hashes["cache_audit"],
                },
                "selection_split": {
                    "path": str(SELECTION_SPLIT),
                    "sha256": authority_hashes["selection_split"],
                },
                "compact_candidate_config": {
                    "path": str(COMPACT_CONFIG),
                    "sha256": authority_hashes["compact_config"],
                },
            },
            "grandfathered_attempt02_launch_records": launches,
            "frozen_gate_time_gpu_hour_ledger": ledger,
            "resume_checkpoints": {
                slot: {
                    "path": snapshot["path"],
                    "sha256": snapshot["sha256"],
                    "size_bytes": snapshot["size_bytes"],
                }
                for slot, snapshot in snapshots.items()
            },
        },
        "frozen_gate": {
            "checkpoint_step": GATE_STEP,
            "base_metric": "balanced_forecast_error",
            "direction": "lower",
            "tie_breaker": "S3",
            "dataset_metrics": list(METRICS),
            "domain_aggregation": (
                "geometric mean of every per-dataset step5000/step0 ratio for both metrics"
            ),
            "group_aggregation": "unweighted geometric mean across domain ratios",
            "underweighted_domains": list(UNDERWEIGHTED_DOMAINS),
            "reference_domains": list(REFERENCE_DOMAINS),
            "minimum_underweighted_over_reference_ratio": GATE_THRESHOLD,
            "requires_any_compact_overall_improvement": True,
            "candidate_objectives": {
                slot: {
                    "variant": specification["variant"],
                    "loss_weights": specification["objective"],
                }
                for slot, specification in SLOTS.items()
            },
            "s5_domain_weights_source": authority["domain_weight_source"],
            "s5_maximum_estimated_physical_gpu_hours": authority[
                "maximum_estimated_physical_gpu_hours"
            ],
        },
        "checkpoint_audit": {
            slot: {
                "variant": SLOTS[slot]["variant"],
                "trajectory": snapshot["trajectory"],
                "records": {
                    f"step{step}": {
                        "step": snapshot["steps"][step]["step"],
                        "epoch": snapshot["steps"][step]["epoch"],
                        "windows_processed": snapshot["steps"][step][
                            "windows_processed"
                        ],
                        "observed_targets_processed": snapshot["steps"][step][
                            "observed_targets_processed"
                        ],
                        "optimizer_batches_processed": snapshot["steps"][step][
                            "optimizer_batches_processed"
                        ],
                        "physical_microbatches_processed": snapshot["steps"][step][
                            "physical_microbatches_processed"
                        ],
                        "balanced": snapshot["steps"][step]["balanced"],
                        "by_dataset": snapshot["steps"][step]["by_dataset"],
                    }
                    for step in (0, GATE_STEP)
                },
                "optimization": snapshot["optimization"],
            }
            for slot, snapshot in snapshots.items()
        },
        "base_selection": {
            "step5000_balanced_forecast_error": scores,
            "exact_tie": tie,
            "selected_candidate": selected_slot,
            "selected_variant": SLOTS[selected_slot]["variant"],
        },
        "overall_compact_improvement": {**improvements, "passes": improvement_passes},
        "candidate_metrics": {
            slot: {
                "overall_compact_improvement": improvements[slot],
                "raw_ratios": ratio_analyses[slot],
            }
            for slot in snapshots
        },
        "decision": {
            "selected_candidate": selected_slot,
            "selected_variant": SLOTS[selected_slot]["variant"],
            "selected_objective": SLOTS[selected_slot]["objective"],
            "selected_candidate_group_ratios": selected_groups,
            "domain_skew_ratio": skew_ratio,
            "domain_skew_threshold": GATE_THRESHOLD,
            "domain_skew_passes": domain_gate_passes,
            "overall_compact_improvement_passes": improvement_passes,
            "activate_s5": activate,
            "outcome": "activate_S5" if activate else "leave_S5_unlaunched",
        },
        "data_access": {
            "development_metrics_read_from_resume_checkpoints": True,
            "confirmation_partition_accessed": False,
            "gift_eval_data_accessed": False,
            "models_executed": False,
            "gpu_used": False,
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        output = _repo_path(args.output)
        payload = _evaluate(output)
    except Exception as error:
        parser.exit(2, f"error: {error}\n")
    decision = payload["decision"]
    print(
        f"wrote {_relative(output)}: {decision['outcome']} "
        f"(selected={payload['base_selection']['selected_candidate']}, "
        f"skew_ratio={decision['domain_skew_ratio']:.9f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
