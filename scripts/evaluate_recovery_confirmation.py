#!/usr/bin/env python3
"""Authorize and execute the single frozen-finalist confirmation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import train_production_student as trainer
import yaml

from timesfm_lab.config import load_config
from timesfm_lab.models import build_student

ROOT = Path(__file__).resolve().parents[1]
GIT_COMMON_DIR = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
).resolve()
SELECTION_CONFIG = ROOT / "configs/performance_recovery/screen_selection.yaml"
DERIVATIVE_ALLOWLIST = ROOT / "configs/performance_recovery/finalist_derivatives.yaml"
SELECTION_SPLIT = (
    ROOT / "results/reproduction/distillation/performance-recovery-selection-split.json"
)
AUTHORIZATION = (
    ROOT / "results/reproduction/distillation/performance-recovery-confirmation-authorization.json"
)
BURN_RECORD_NAME = "timesfm-lab-performance-recovery-confirmation-access-burn.json"
BURN_RECORD = GIT_COMMON_DIR / BURN_RECORD_NAME
BURN_LOCATOR = {"scope": "git_common_dir", "name": BURN_RECORD_NAME}
RESULT = (
    ROOT / "results/reproduction/distillation/performance-recovery-confirmation-evaluation.json"
)
EXPECTED_PROTOCOL = "timesfm3-performance-recovery-v1.2"
EXPECTED_SPLIT_SHA256 = "9d3e06b328b76baaab558c18717b7336961f07e81261989349c0f4e20c899cd9"
EXPECTED_DERIVATIVE_ALLOWLIST_SHA256 = (
    "a10a72f7e2334fc0f36308e1e396259c4506d794206fe54eb41a374ffc6e3b78"
)
EXPECTED_DATASET_REVISION = "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874"
MAXIMUM_ROSTER_MODELS = 6
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
CONFIRMATION_POLICY = {
    "evaluate_entire_roster_once": True,
    "confirmation_may_change_recipe": False,
    "confirmation_may_change_training_budget": False,
    "confirmation_may_select_checkpoint": False,
    "confirmation_may_change_learning_rate": False,
    "confirmation_may_trigger_early_stopping": False,
}
ROSTER_KEYS = {
    "schema_version",
    "status",
    "protocol_id",
    "frozen_at_utc",
    "screen_selection",
    "derivative_allowlist",
    "confirmation_policy",
    "finalists",
}
FINALIST_KEYS = {
    "finalist_id",
    "source_candidate_id",
    "source_screen_finalist_sha256",
    "full_training_authority_sha256",
    "training_recipe_sha256",
    "training_seed",
    "training_budget_steps",
    "development_selected_checkpoint_step",
    "config",
    "checkpoint",
    "development_selection_evidence",
    "training_result",
    "training_code_commit",
}
DEVELOPMENT_SELECTION_KEYS = {
    "schema_version",
    "status",
    "protocol_id",
    "finalist_id",
    "config",
    "checkpoint",
    "training_result",
    "training_seed",
    "selected_checkpoint_step",
    "selection_partition",
    "selection_metric",
    "selection_direction",
    "confirmation_partition_accessed",
    "gift_eval_data_accessed",
}
AUTHORIZATION_KEYS = {
    "schema_version",
    "status",
    "protocol_id",
    "authorized_at_utc",
    "frozen_roster",
    "frozen_screen_selection",
    "derivative_allowlist",
    "selection_config",
    "candidate_registry",
    "selection_split",
    "corpus_plan",
    "cache_audit",
    "data_root",
    "cache_root",
    "implementation",
    "policy",
    "burn_record",
    "result",
    "authorization_must_be_committed_before_use",
    "authorization_payload_sha256",
}


class ConfirmationError(RuntimeError):
    """A confirmation-embargo invariant was not satisfied."""


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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _path(value: str) -> Path:
    if Path(value).is_absolute():
        raise ConfirmationError(f"repository artifact path must be relative: {value}")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ConfirmationError(f"artifact escapes repository root: {value}") from error
    return path


def _recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ConfirmationError(f"expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ConfirmationError(f"expected YAML mapping: {path}")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _require_binding(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ConfirmationError(f"{label} must contain exactly path and sha256")
    expected = value["sha256"]
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise ConfirmationError(f"{label} has an invalid SHA-256")
    path = _path(str(value["path"]))
    if not path.is_file() or _sha256(path) != expected:
        raise ConfirmationError(f"{label} file/hash binding failed")
    return path


def _require_tracked_clean(path: Path, label: str) -> None:
    relative = _relative(path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode:
        raise ConfirmationError(f"{label} is not frozen in git: {relative}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ConfirmationError(f"{label} differs from HEAD: {relative}")


def _git_blob_sha256(commit: str, path: str) -> str:
    if not GIT_COMMIT.fullmatch(commit):
        raise ConfirmationError(f"invalid training commit: {commit!r}")
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode or resolved.stdout.strip() != commit:
        raise ConfirmationError(f"training commit does not resolve exactly: {commit}")
    safe_path = _path(path)
    relative = _relative(safe_path)
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if blob.returncode:
        raise ConfirmationError(f"training commit lacks required artifact: {relative}")
    return hashlib.sha256(blob.stdout).hexdigest()


def _source_tree() -> dict[str, Any]:
    files = [
        {"path": _relative(path), "sha256": _sha256(path)}
        for path in sorted((ROOT / "src/timesfm_lab").rglob("*.py"))
    ]
    return {"files": files, "sha256": _canonical_sha256(files)}


def _deployment_fingerprint(config_path: Path) -> str:
    config = _load_yaml(config_path)
    return _canonical_sha256(
        {"student": config["student"], "inference": config.get("inference", {})}
    )


def _validate_derivative_allowlist() -> dict[str, Any]:
    _require_tracked_clean(DERIVATIVE_ALLOWLIST, "finalist-derivative allowlist")
    if _sha256(DERIVATIVE_ALLOWLIST) != EXPECTED_DERIVATIVE_ALLOWLIST_SHA256:
        raise ConfirmationError("finalist-derivative allowlist hash changed")
    allowlist = _load_yaml(DERIVATIVE_ALLOWLIST)
    if (
        allowlist.get("schema_version") != 1
        or allowlist.get("status") != "frozen_predeclared_before_full_training"
        or allowlist.get("protocol_id") != EXPECTED_PROTOCOL
    ):
        raise ConfirmationError("finalist-derivative allowlist is not frozen")
    authorities = allowlist.get("authorities", {})
    expected = {
        "screen_selection_config": SELECTION_CONFIG,
        "candidate_registry": ROOT / "configs/performance_recovery/candidates.yaml",
        "corpus_plan": ROOT
        / "results/reproduction/distillation/production-1m-corpus-plan.json",
        "cache_audit": ROOT
        / "results/reproduction/distillation/production-1m-cache-audit.json",
        "selection_split": SELECTION_SPLIT,
    }
    for name, path in expected.items():
        binding = authorities.get(name)
        if binding != _binding(path):
            raise ConfirmationError(f"finalist-derivative {name} authority changed")
    if (
        authorities.get("screen_selection", {}).get("path")
        != "results/reproduction/distillation/performance-recovery-screen-finalists.json"
        or authorities.get("data_root") != "data/gift-pretrain-production"
        or authorities.get("cache_root") != "teacher_cache/production-1m"
        or authorities.get("dataset_revision") != EXPECTED_DATASET_REVISION
    ):
        raise ConfirmationError("finalist-derivative data/selection authority changed")
    registry = _load_yaml(expected["candidate_registry"])
    derivatives = allowlist.get("derivatives", {})
    if set(derivatives) != {"S1", "S2", "S3", "S4", "S5"}:
        raise ConfirmationError("finalist-derivative candidate set changed")
    for candidate_id, derivative in derivatives.items():
        candidate = registry["candidates"][candidate_id]
        config_path = _require_binding(
            derivative["config"], f"{candidate_id} derivative config"
        )
        config = _load_yaml(config_path)
        candidate_variant = candidate.get("variant")
        candidate_config = candidate.get("config")
        if candidate_id == "S5" and candidate_config is None:
            candidate_config = (
                "configs/distillation/performance_recovery_compact_s5_domain_balanced.yaml"
            )
        variant_matches = (
            derivative.get("variant") == candidate_variant
            if candidate_variant is not None
            else candidate_id == "S5"
            and derivative.get("variant") in config["training"]["loss_weights"]
        )
        if (
            derivative.get("source_candidate_id") != candidate_id
            or not variant_matches
            or derivative["config"]["path"] != candidate_config
            or derivative.get("deployment_fingerprint_sha256")
            != _deployment_fingerprint(config_path)
            or not isinstance(
                derivative.get("candidate_registry_initialization_allowed"), bool
            )
        ):
            raise ConfirmationError(f"{candidate_id} derivative differs from its registry recipe")
    return allowlist


def _validate_selected_derivative(
    selected: dict[str, Any], candidate_id: str, allowlist: dict[str, Any]
) -> str:
    derivative = allowlist["derivatives"][candidate_id]
    if (
        selected.get("candidate_id") != candidate_id
        or selected.get("variant") != derivative["variant"]
        or selected.get("config") != derivative["config"]
        or selected.get("deployment_fingerprint_sha256")
        != derivative["deployment_fingerprint_sha256"]
    ):
        raise ConfirmationError(
            f"{candidate_id}: selected deployment is not its predeclared derivative base"
        )
    return _canonical_sha256(selected)


def _atomic_json_no_clobber(path: Path, value: dict[str, Any]) -> None:
    """Create complete JSON atomically and fsync both file and directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ConfirmationError(f"refusing to overwrite one-shot artifact: {path}") from error
        linked = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
        if linked:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


def _code_binding() -> dict[str, Any]:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_production_student.py",
        *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
    ]
    for path in files:
        _require_tracked_clean(path, "confirmation implementation")
    bindings = [_binding(path) for path in files]
    return {"files": bindings, "sha256": _canonical_sha256(bindings)}


def _validate_screen_selection(path: Path) -> dict[str, Any]:
    _require_tracked_clean(path, "frozen screen selection")
    selection = _load_json(path)
    required = {
        "schema_version",
        "status",
        "protocol_id",
        "frozen_at_utc",
        "screen_audit",
        "selection_rule",
        "finalist_count",
        "finalist_count_reason",
        "finalists",
        "confirmation_embargo",
    }
    if set(selection) != required:
        raise ConfirmationError("screen selection has an unexpected schema")
    if (
        selection["schema_version"] != 1
        or selection["status"] != "finalists_frozen"
        or selection["protocol_id"] != EXPECTED_PROTOCOL
    ):
        raise ConfirmationError("screen selection is not a frozen v1.2 finalist selection")
    embargo = selection["confirmation_embargo"]
    if (
        not isinstance(embargo, dict)
        or embargo.get("held_through_screen_selection") is not True
        or embargo.get("confirmation_partition_accessed_during_selection") is not False
        or set(embargo.get("confirmation_may_not_change", []))
        != {"recipe", "training budget", "development-selected checkpoint step"}
    ):
        raise ConfirmationError("screen selection does not preserve the confirmation embargo")

    audit_path = _require_binding(selection["screen_audit"], "screen audit")
    audit = _load_json(audit_path)
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "succeeded"
        or audit.get("protocol_id") != EXPECTED_PROTOCOL
        or audit.get("eligible_to_freeze") is not True
        or audit.get("confirmation_partition_accessed") is not False
        or audit.get("gift_eval_metrics_accessed") is not False
    ):
        raise ConfirmationError("screen audit is not eligible target-blind evidence")
    sources = audit.get("sources", {})
    if set(sources) != {"selection_config", "development_evaluation", "latency_evidence"}:
        raise ConfirmationError("screen audit lacks the exact selection sources")
    source_paths = {
        name: _require_binding(binding, f"screen audit {name}") for name, binding in sources.items()
    }
    if source_paths["selection_config"] != SELECTION_CONFIG.resolve():
        raise ConfirmationError("screen audit is bound to another selection config")
    selection_config = _load_yaml(SELECTION_CONFIG)
    if selection["selection_rule"] != selection_config["score"]:
        raise ConfirmationError("screen selection rule changed")
    if not str(selection["finalist_count_reason"]).strip():
        raise ConfirmationError("screen selection lacks its finalist-count reason")
    count = int(selection["finalist_count"])
    ranking = audit["candidate_ranking"]
    if count < 1 or count > 2 or count > len(ranking):
        raise ConfirmationError("screen selection contains an invalid finalist count")
    expected_finalists = [
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
        for row in ranking[:count]
    ]
    if selection["finalists"] != expected_finalists:
        raise ConfirmationError("screen finalists differ from the audited ranking")
    current_model_source = _source_tree()
    trainer_path = (ROOT / "scripts/train_production_student.py").resolve()
    trainer_binding = _binding(trainer_path)
    for row in selection["finalists"]:
        config_path = _require_binding(
            row["config"], f"{row['candidate_id']} selected config"
        )
        _require_tracked_clean(config_path, f"{row['candidate_id']} selected config")
        if _deployment_fingerprint(config_path) != row["deployment_fingerprint_sha256"]:
            raise ConfirmationError(
                f"{row['candidate_id']} selected deployment fingerprint changed"
            )
        if row["model_source"] != current_model_source:
            raise ConfirmationError(f"{row['candidate_id']} selected model source changed")
        inference = row["inference_implementation"]
        if (
            not isinstance(inference, dict)
            or inference.get("path") != trainer_binding["path"]
            or inference.get("sha256") != trainer_binding["sha256"]
        ):
            raise ConfirmationError(
                f"{row['candidate_id']} selected inference implementation changed"
            )
    return selection


def _domain_weight_configuration_sha256(config: dict[str, Any]) -> str | None:
    training = config["training"]
    if str(training.get("loss_reduction", "observed_target_element")) != (
        "per_window_domain_balanced"
    ):
        return None
    return _canonical_sha256(
        {
            "domain_weight_source": training.get("domain_weight_source"),
            "domain_weights": {
                str(key): float(value)
                for key, value in training.get("domain_weights", {}).items()
            },
            "outer_training_counts": {
                str(key): int(value)
                for key, value in training.get(
                    "domain_weight_outer_training_counts", {}
                ).items()
            },
            "denominator": training.get("domain_weight_denominator"),
            "zero_target_window_policy": training.get("zero_target_window_policy"),
        }
    )


def _validate_git_artifacts(
    artifacts: Any,
    commit: str,
    expected_paths: set[str],
) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise ConfirmationError("full-training Git artifact list is absent")
    observed: dict[str, str] = {}
    for binding in artifacts:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ConfirmationError("full-training Git artifact binding is malformed")
        path = str(binding["path"])
        digest = str(binding["sha256"])
        if path in observed or not SHA256.fullmatch(digest):
            raise ConfirmationError("full-training Git artifact is duplicate/invalid")
        if _git_blob_sha256(commit, path) != digest:
            raise ConfirmationError(f"training Git blob hash mismatch: {path}")
        observed[path] = digest
    if set(observed) != expected_paths:
        missing = sorted(expected_paths - set(observed))
        extra = sorted(set(observed) - expected_paths)
        raise ConfirmationError(
            f"full-training Git artifact set differs; missing={missing}, extra={extra}"
        )


def _validate_initialization(
    launch: dict[str, Any],
    selected: dict[str, Any],
    derivative: dict[str, Any],
    registry_candidate: dict[str, Any],
    config: dict[str, Any],
    training_seed: int,
) -> None:
    origin = launch["initialization_origin_fingerprint"]
    if (
        not isinstance(origin, dict)
        or set(origin)
        != {
            "schema_version",
            "kind",
            "random_initialization_sha256",
            "initialization_sha256",
            "initialization_checkpoint",
            "initialization_checkpoint_sha256",
        }
        or origin["schema_version"] != 1
        or launch["initialization_origin_sha256"] != _canonical_sha256(origin)
    ):
        raise ConfirmationError("full-training initialization fingerprint is malformed")

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(training_seed)
        initial_model = build_student(config["student"])
        random_sha256 = trainer._state_sha256(initial_model)
    finally:
        torch.random.set_rng_state(rng_state)
    if origin["random_initialization_sha256"] != random_sha256:
        raise ConfirmationError("random initialization cannot be reproduced from the seed")

    mode = str(launch["initialization_mode"])
    declared = registry_candidate.get("initialization", {})
    if mode == "candidate_registry_initialization" and declared.get("kind") == "seeded_random":
        if (
            derivative["candidate_registry_initialization_allowed"] is not True
            or origin["kind"] != "seeded_random"
            or origin["initialization_checkpoint"] is not None
            or origin["initialization_checkpoint_sha256"] is not None
            or origin["initialization_sha256"] != random_sha256
        ):
            raise ConfirmationError("seeded finalist initialization differs from its policy")
        if training_seed == 42 and origin["initialization_sha256"] != declared["state_sha256"]:
            raise ConfirmationError("seed-42 initialization differs from the candidate registry")
        return

    if mode == "selected_screen_checkpoint_weights_only":
        expected_checkpoint = selected["checkpoint"]
    elif (
        mode == "candidate_registry_initialization"
        and derivative["candidate_registry_initialization_allowed"] is True
        and declared.get("kind") == "checkpoint"
    ):
        expected_checkpoint = {
            "path": declared["path"], "sha256": declared["file_sha256"]
        }
    else:
        raise ConfirmationError("full-training initialization mode is not allowlisted")
    checkpoint_path = _recorded_path(str(origin["initialization_checkpoint"]))
    if (
        checkpoint_path != _path(expected_checkpoint["path"])
        or origin["initialization_checkpoint_sha256"] != expected_checkpoint["sha256"]
        or _sha256(checkpoint_path) != expected_checkpoint["sha256"]
        or origin["kind"] != "checkpoint"
    ):
        raise ConfirmationError("full-training initialization checkpoint changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    initial_model.load_state_dict(state)
    if trainer._state_sha256(initial_model) != origin["initialization_sha256"]:
        raise ConfirmationError("loaded initialization state hash does not reproduce")


def _validate_full_training_authority(
    row: dict[str, Any],
    result: dict[str, Any],
    selected: dict[str, Any],
    selection_binding: dict[str, str],
    allowlist: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    extra = result["extra"]
    authority = extra.get("full_training_authority")
    claimed_authority_sha256 = extra.get("full_training_authority_sha256")
    if (
        not isinstance(authority, dict)
        or set(authority)
        != {
            "schema_version",
            "status",
            "launch",
            "launch_sha256",
            "best_checkpoint",
            "final_checkpoint",
        }
        or authority["schema_version"] != 1
        or authority["status"] != "completed_full_training_authority"
        or claimed_authority_sha256 != _canonical_sha256(authority)
        or row["full_training_authority_sha256"] != claimed_authority_sha256
    ):
        raise ConfirmationError("completed full-training authority failed its hash/schema")
    launch = authority["launch"]
    if (
        not isinstance(launch, dict)
        or set(launch)
        != {
            "schema_version",
            "status",
            "protocol_id",
            "source_candidate_id",
            "variant",
            "derivative_allowlist",
            "screen_selection",
            "training_commit",
            "repository_git_artifacts",
            "config",
            "corpus",
            "selection_split",
            "training_recipe_fingerprint",
            "training_recipe_sha256",
            "initialization_mode",
            "initialization_origin_fingerprint",
            "initialization_origin_sha256",
        }
        or launch["schema_version"] != 1
        or launch["status"] != "frozen_full_training_launch_authority"
        or launch["protocol_id"] != EXPECTED_PROTOCOL
        or authority["launch_sha256"] != _canonical_sha256(launch)
        or extra.get("full_training_launch_authority") != launch
        or extra.get("full_training_launch_authority_sha256")
        != authority["launch_sha256"]
    ):
        raise ConfirmationError("full-training launch authority failed its hash/schema")

    source_candidate = str(row["source_candidate_id"])
    derivative = allowlist["derivatives"][source_candidate]
    selected_deployment = {
        name: selected[name]
        for name in (
            "candidate_id",
            "variant",
            "config",
            "checkpoint",
            "deployment_fingerprint_sha256",
            "model_source",
            "inference_implementation",
        )
    }
    if (
        row["source_screen_finalist_sha256"] != _canonical_sha256(selected)
        or launch["source_candidate_id"] != source_candidate
        or launch["variant"] != selected["variant"]
        or launch["variant"] != derivative["variant"]
        or launch["config"] != derivative["config"]
        or launch["config"] != selected["config"]
        or row["config"] != derivative["config"]
        or launch["screen_selection"]
        != {
            "path": selection_binding["path"],
            "sha256": selection_binding["sha256"],
            "selected_finalist_sha256": _canonical_sha256(selected),
            "selected_deployment": selected_deployment,
        }
        or launch["derivative_allowlist"]
        != {
            "path": _relative(DERIVATIVE_ALLOWLIST),
            "sha256": EXPECTED_DERIVATIVE_ALLOWLIST_SHA256,
            "entry": derivative,
        }
        or selected["deployment_fingerprint_sha256"]
        != derivative["deployment_fingerprint_sha256"]
    ):
        raise ConfirmationError("full training is not an exact selected-recipe derivative")

    commit = str(row["training_code_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode:
        raise ConfirmationError("full-training commit is not an ancestor of confirmation HEAD")
    authorities = allowlist["authorities"]
    expected_corpus = {
        "plan": authorities["corpus_plan"],
        "cache_audit": authorities["cache_audit"],
        "data_root": authorities["data_root"],
        "cache_root": authorities["cache_root"],
        "dataset_revision": authorities["dataset_revision"],
    }
    if (
        launch["training_commit"] != commit
        or result["git_commit"] != commit
        or launch["corpus"] != expected_corpus
        or launch["selection_split"] != authorities["selection_split"]
    ):
        raise ConfirmationError("full-training commit/corpus/split authority changed")
    required_git_paths = {
        _relative(DERIVATIVE_ALLOWLIST),
        selection_binding["path"],
        derivative["config"]["path"],
        authorities["corpus_plan"]["path"],
        authorities["cache_audit"]["path"],
        authorities["selection_split"]["path"],
        authorities["screen_selection_config"]["path"],
        authorities["candidate_registry"]["path"],
        selected["inference_implementation"]["path"],
        *(binding["path"] for binding in selected["model_source"]["files"]),
    }
    _validate_git_artifacts(launch["repository_git_artifacts"], commit, required_git_paths)
    git_bindings = {row["path"]: row["sha256"] for row in launch["repository_git_artifacts"]}
    expected_git_hashes = {
        selection_binding["path"]: selection_binding["sha256"],
        _relative(DERIVATIVE_ALLOWLIST): EXPECTED_DERIVATIVE_ALLOWLIST_SHA256,
        derivative["config"]["path"]: derivative["config"]["sha256"],
        authorities["corpus_plan"]["path"]: authorities["corpus_plan"]["sha256"],
        authorities["cache_audit"]["path"]: authorities["cache_audit"]["sha256"],
        authorities["selection_split"]["path"]: authorities["selection_split"]["sha256"],
        authorities["screen_selection_config"]["path"]: authorities[
            "screen_selection_config"
        ]["sha256"],
        authorities["candidate_registry"]["path"]: authorities["candidate_registry"][
            "sha256"
        ],
        selected["inference_implementation"]["path"]: selected[
            "inference_implementation"
        ]["sha256"],
        **{
            binding["path"]: binding["sha256"]
            for binding in selected["model_source"]["files"]
        },
    }
    if any(git_bindings[path] != digest for path, digest in expected_git_hashes.items()):
        raise ConfirmationError("training commit does not contain the selected source authority")

    training = config["training"]
    training_source_sha256 = {
        selected["inference_implementation"]["path"]: selected[
            "inference_implementation"
        ]["sha256"],
        **{
            binding["path"]: binding["sha256"]
            for binding in selected["model_source"]["files"]
        },
    }
    expected_recipe = {
        "schema_version": 1,
        "config_sha256": derivative["config"]["sha256"],
        "corpus_plan_sha256": authorities["corpus_plan"]["sha256"],
        "selection_split_manifest_sha256": authorities["selection_split"]["sha256"],
        "training_source_sha256": training_source_sha256,
        "variant": derivative["variant"],
        "loss_weights": training["loss_weights"][derivative["variant"]],
        "loss_reduction": str(training.get("loss_reduction", "observed_target_element")),
        "domain_weight_configuration_sha256": _domain_weight_configuration_sha256(config),
        "training_seed": row["training_seed"],
        "split_seed": int(allowlist["allowed_runtime_differences"]["split_seed"]),
        "validation_partition": "development",
        "logical_batch_size_windows": training.get("logical_batch_size_windows"),
        "maximum_steps": row["training_budget_steps"],
        "distributed": False,
        "early_stopping_enabled": True,
    }
    expected_recipe_sha256 = _canonical_sha256(expected_recipe)
    if (
        launch["training_recipe_fingerprint"] != expected_recipe
        or launch["training_recipe_sha256"] != expected_recipe_sha256
        or extra.get("training_recipe_fingerprint") != expected_recipe
        or extra.get("training_recipe_sha256") != expected_recipe_sha256
        or row["training_recipe_sha256"] != expected_recipe_sha256
        or extra.get("training_source_sha256") != training_source_sha256
    ):
        raise ConfirmationError("full-training recipe fingerprint does not recompute")

    registry = _load_yaml(_path(authorities["candidate_registry"]["path"]))
    registry_candidate = registry["candidates"][source_candidate]
    _validate_initialization(
        launch,
        selected,
        derivative,
        registry_candidate,
        config,
        int(row["training_seed"]),
    )
    if (
        launch["initialization_origin_fingerprint"]
        != extra.get("initialization_origin_fingerprint")
        or launch["initialization_origin_sha256"]
        != extra.get("initialization_origin_sha256")
    ):
        raise ConfirmationError("result initialization differs from its launch authority")

    best_binding = authority["best_checkpoint"]
    final_binding = authority["final_checkpoint"]
    best_path = _require_binding(best_binding, "full-training best checkpoint")
    _require_binding(final_binding, "full-training final checkpoint")
    if (
        best_path != checkpoint_path
        or row["checkpoint"] != best_binding
        or extra.get("best_checkpoint_sha256") != best_binding["sha256"]
        or extra.get("final_checkpoint_sha256") != final_binding["sha256"]
        or _recorded_path(str(extra.get("best_checkpoint"))) != best_path
        or _recorded_path(str(extra.get("final_checkpoint")))
        != _path(final_binding["path"])
    ):
        raise ConfirmationError("full-training checkpoint hashes/paths do not reproduce")


def _validate_roster(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_tracked_clean(path, "full-training finalist roster")
    roster = _load_json(path)
    if set(roster) != ROSTER_KEYS:
        raise ConfirmationError("full-training finalist roster has an unexpected schema")
    if (
        roster["schema_version"] != 2
        or roster["status"] != "full_training_finalist_roster_frozen"
        or roster["protocol_id"] != EXPECTED_PROTOCOL
        or roster["confirmation_policy"] != CONFIRMATION_POLICY
    ):
        raise ConfirmationError("full-training finalist roster is not frozen/authorized safely")
    allowlist = _validate_derivative_allowlist()
    if roster["derivative_allowlist"] != _binding(DERIVATIVE_ALLOWLIST):
        raise ConfirmationError("roster is bound to another finalist-derivative allowlist")
    selection_path = _require_binding(roster["screen_selection"], "frozen screen selection")
    selection = _validate_screen_selection(selection_path)
    selected_by_id = {
        str(selected["candidate_id"]): selected for selected in selection["finalists"]
    }
    selected_ids = set(selected_by_id)
    finalists = roster["finalists"]
    if not isinstance(finalists, list) or not finalists or len(finalists) > MAXIMUM_ROSTER_MODELS:
        raise ConfirmationError("full-training finalist roster must contain 1-6 models")
    model_ids = set()
    candidate_seeds = set()
    checkpoint_hashes = set()
    represented_candidates = set()
    for row in finalists:
        if not isinstance(row, dict) or set(row) != FINALIST_KEYS:
            raise ConfirmationError("full-training finalist entry has an unexpected schema")
        model_id = str(row["finalist_id"])
        if not model_id or model_id in model_ids:
            raise ConfirmationError("full-training finalist ids must be nonempty and unique")
        model_ids.add(model_id)
        source_candidate = str(row["source_candidate_id"])
        if source_candidate not in selected_ids:
            raise ConfirmationError(f"{model_id}: source candidate was not frozen by selection")
        selected = selected_by_id[source_candidate]
        selected_sha256 = _validate_selected_derivative(
            selected, source_candidate, allowlist
        )
        if row["source_screen_finalist_sha256"] != selected_sha256:
            raise ConfirmationError(
                f"{model_id}: source finalist fingerprint/recipe differs from the allowlist"
            )
        represented_candidates.add(source_candidate)
        recipe_sha256 = row["training_recipe_sha256"]
        authority_sha256 = row["full_training_authority_sha256"]
        if (
            not isinstance(recipe_sha256, str)
            or not SHA256.fullmatch(recipe_sha256)
            or not isinstance(authority_sha256, str)
            or not SHA256.fullmatch(authority_sha256)
        ):
            raise ConfirmationError(f"{model_id}: training recipe SHA-256 is invalid")
        seed = row["training_seed"]
        budget = row["training_budget_steps"]
        checkpoint_step = row["development_selected_checkpoint_step"]
        runtime_policy = allowlist["allowed_runtime_differences"]
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(budget, int)
            or isinstance(budget, bool)
            or not isinstance(checkpoint_step, int)
            or isinstance(checkpoint_step, bool)
            or budget <= 0
            or checkpoint_step < 0
            or checkpoint_step > budget
            or seed not in runtime_policy["training_seed"]
            or budget not in runtime_policy["maximum_steps"]
        ):
            raise ConfirmationError(f"{model_id}: seed/budget/checkpoint step is invalid")
        candidate_seed = (source_candidate, seed)
        checkpoint_hash = row.get("checkpoint", {}).get("sha256")
        if candidate_seed in candidate_seeds or checkpoint_hash in checkpoint_hashes:
            raise ConfirmationError(f"{model_id}: duplicate candidate/seed or checkpoint")
        candidate_seeds.add(candidate_seed)
        checkpoint_hashes.add(checkpoint_hash)
        commit = row["training_code_commit"]
        if not isinstance(commit, str) or not GIT_COMMIT.fullmatch(commit):
            raise ConfirmationError(f"{model_id}: training code commit is invalid")
        config_path = _require_binding(row["config"], f"{model_id} config")
        checkpoint_path = _require_binding(row["checkpoint"], f"{model_id} checkpoint")
        selection_evidence_path = _require_binding(
            row["development_selection_evidence"],
            f"{model_id} development selection evidence",
        )
        result_path = _require_binding(row["training_result"], f"{model_id} training result")
        config = load_config(config_path)
        result = _load_json(result_path)
        selection_evidence = _load_json(selection_evidence_path)
        training = result.get("extra", {}).get("training", {})
        if config.get("dataset_revision") != EXPECTED_DATASET_REVISION:
            raise ConfirmationError(f"{model_id}: config dataset revision changed")
        if (
            result.get("status") != "succeeded"
            or result.get("git_commit") != commit
            or result.get("dataset_revision") != EXPECTED_DATASET_REVISION
            or result.get("extra", {}).get("training_seed") != seed
            or result.get("extra", {}).get("validation_split_seed")
            != int(allowlist["allowed_runtime_differences"]["split_seed"])
            or result.get("extra", {}).get("validation_partition") != "development"
            or result.get("extra", {}).get("confirmation_partition_accessed") is not False
            or result.get("extra", {}).get("gift_eval_data_accessed") is not False
            or result.get("extra", {}).get("training_recipe_sha256") != recipe_sha256
            or training.get("maximum_steps") != budget
            or not isinstance(training.get("steps"), int)
            or training["steps"] < checkpoint_step
            or training["steps"] > budget
        ):
            raise ConfirmationError(f"{model_id}: training result provenance is incomplete")
        best_checkpoint = _recorded_path(str(result["extra"]["best_checkpoint"]))
        if best_checkpoint != checkpoint_path:
            raise ConfirmationError(f"{model_id}: roster checkpoint is not the recorded best")
        _validate_full_training_authority(
            row,
            result,
            selected,
            roster["screen_selection"],
            allowlist,
            config,
            checkpoint_path,
        )
        curve = result["extra"].get("learning_curve")
        if not isinstance(curve, list) or not curve:
            raise ConfirmationError(f"{model_id}: training result lacks a learning curve")
        curve_steps = [int(curve_row["step"]) for curve_row in curve]
        if (
            curve_steps != sorted(set(curve_steps))
            or curve_steps[-1] != int(training["steps"])
        ):
            raise ConfirmationError(f"{model_id}: DEVELOPMENT learning curve is incomplete")
        selected_curve_row = min(
            curve,
            key=lambda curve_row: (
                trainer._validation_score(curve_row["validation"], config["training"]),
                int(curve_row["step"]),
            ),
        )
        if int(selected_curve_row["step"]) != checkpoint_step:
            raise ConfirmationError(f"{model_id}: checkpoint was not DEVELOPMENT-selected")
        if (
            set(selection_evidence) != DEVELOPMENT_SELECTION_KEYS
            or selection_evidence["schema_version"] != 1
            or selection_evidence["status"] != "development_checkpoint_frozen"
            or selection_evidence["protocol_id"] != EXPECTED_PROTOCOL
            or selection_evidence["finalist_id"] != model_id
            or selection_evidence["config"] != row["config"]
            or selection_evidence["checkpoint"] != row["checkpoint"]
            or selection_evidence["training_result"] != row["training_result"]
            or selection_evidence["training_seed"] != seed
            or selection_evidence["selected_checkpoint_step"] != checkpoint_step
            or selection_evidence["selection_partition"] != "development"
            or selection_evidence["selection_metric"]
            != config["training"]["validation_selection_metric"]
            or selection_evidence["selection_direction"] != "lower"
            or selection_evidence["confirmation_partition_accessed"] is not False
            or selection_evidence["gift_eval_data_accessed"] is not False
        ):
            raise ConfirmationError(f"{model_id}: DEVELOPMENT selection evidence is invalid")
    if represented_candidates != selected_ids:
        raise ConfirmationError("full-training roster omits a frozen screen finalist")
    return roster, selection


def _validate_selection_split() -> dict[str, Any]:
    if _sha256(SELECTION_SPLIT) != EXPECTED_SPLIT_SHA256:
        raise ConfirmationError("frozen development/confirmation split hash changed")
    split = _load_json(SELECTION_SPLIT)
    if (
        split.get("schema_version") != 1
        or split.get("protocol_id") != "timesfm3-performance-recovery-v1.1"
        or split.get("status") != "frozen_uninspected"
        or split.get("target_accessed") is not False
        or split.get("teacher_output_accessed") is not False
    ):
        raise ConfirmationError("selection split is not the frozen target-blind authority")
    if len(split.get("datasets", [])) != 77 or any(
        not isinstance(row.get("confirmation_eligible"), bool) for row in split["datasets"]
    ):
        raise ConfirmationError("selection split must explicitly classify all 77 datasets")
    eligible = [row for row in split["datasets"] if row.get("confirmation_eligible") is True]
    excluded = [row for row in split["datasets"] if row.get("confirmation_eligible") is False]
    if (
        len(eligible) != 68
        or sum(int(row["partitions"]["confirmation"]["count"]) for row in eligible) != 50_318
        or len(excluded) != 9
    ):
        raise ConfirmationError("frozen confirmation scope is not 68 datasets/50318 windows")
    for row in excluded:
        if (
            int(row["partitions"]["confirmation"]["count"]) != 0
            or not str(row.get("inner_split_report", {}).get("reason", "")).strip()
        ):
            raise ConfirmationError("zero-window exclusion lacks its manifest reason")
    return split


def _authorization_payload(roster_path: Path) -> dict[str, Any]:
    if AUTHORIZATION.exists() or BURN_RECORD.exists() or RESULT.exists():
        raise ConfirmationError("confirmation authorization/access/result already exists")
    roster, _ = _validate_roster(roster_path)
    _validate_selection_split()
    selection_config = _load_yaml(SELECTION_CONFIG)
    registry_path = _path(selection_config["registry"])
    registry = _load_yaml(registry_path)
    corpus = registry["corpus"]
    if _sha256(registry_path) != selection_config["registry_sha256"]:
        raise ConfirmationError("candidate registry changed after selection freeze")
    if corpus["dataset_revision"] != EXPECTED_DATASET_REVISION:
        raise ConfirmationError("candidate registry dataset revision changed")
    if selection_config["selection_split_manifest_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ConfirmationError("selection config split authority changed")
    plan_path = _path(corpus["plan"])
    cache_audit_path = _path(corpus["cache_audit"])
    if _sha256(plan_path) != corpus["plan_sha256"]:
        raise ConfirmationError("production corpus plan changed")
    if _sha256(cache_audit_path) != corpus["cache_audit_sha256"]:
        raise ConfirmationError("production cache audit changed")
    code = _code_binding()
    payload = {
        "schema_version": 1,
        "status": "confirmation_access_authorized",
        "protocol_id": EXPECTED_PROTOCOL,
        "authorized_at_utc": _now(),
        "frozen_roster": _binding(roster_path),
        "frozen_screen_selection": roster["screen_selection"],
        "derivative_allowlist": _binding(DERIVATIVE_ALLOWLIST),
        "selection_config": _binding(SELECTION_CONFIG),
        "candidate_registry": _binding(registry_path),
        "selection_split": _binding(SELECTION_SPLIT),
        "corpus_plan": _binding(plan_path),
        "cache_audit": _binding(cache_audit_path),
        "data_root": str(corpus["data_root"]),
        "cache_root": str(corpus["cache_root"]),
        "implementation": code,
        "policy": CONFIRMATION_POLICY,
        "burn_record": BURN_LOCATOR,
        "result": _relative(RESULT),
        "authorization_must_be_committed_before_use": True,
    }
    payload["authorization_payload_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_authorization() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not AUTHORIZATION.is_file():
        raise ConfirmationError("committed confirmation authorization is absent")
    _require_tracked_clean(AUTHORIZATION, "confirmation authorization")
    authorization = _load_json(AUTHORIZATION)
    if set(authorization) != AUTHORIZATION_KEYS:
        raise ConfirmationError("confirmation authorization has an unexpected schema")
    claimed = authorization.pop("authorization_payload_sha256", None)
    if claimed != _canonical_sha256(authorization):
        raise ConfirmationError("confirmation authorization payload hash failed")
    authorization["authorization_payload_sha256"] = claimed
    if (
        authorization.get("schema_version") != 1
        or authorization.get("status") != "confirmation_access_authorized"
        or authorization.get("protocol_id") != EXPECTED_PROTOCOL
        or authorization.get("policy") != CONFIRMATION_POLICY
        or authorization.get("burn_record") != BURN_LOCATOR
        or authorization.get("result") != _relative(RESULT)
        or authorization.get("authorization_must_be_committed_before_use") is not True
    ):
        raise ConfirmationError("confirmation authorization schema/policy is invalid")
    if authorization.get("implementation") != _code_binding():
        raise ConfirmationError("confirmation implementation changed after authorization")
    roster_path = _require_binding(authorization["frozen_roster"], "frozen roster")
    roster, selection = _validate_roster(roster_path)
    if authorization["frozen_screen_selection"] != roster["screen_selection"]:
        raise ConfirmationError("authorization is bound to a different screen selection")
    if authorization["derivative_allowlist"] != _binding(DERIVATIVE_ALLOWLIST):
        raise ConfirmationError("authorization is bound to another derivative allowlist")
    if authorization["selection_split"] != _binding(SELECTION_SPLIT):
        raise ConfirmationError("authorization is bound to another selection split")
    _validate_selection_split()
    if authorization["selection_config"] != _binding(SELECTION_CONFIG):
        raise ConfirmationError("authorization is bound to another selection config")
    _require_binding(authorization["corpus_plan"], "corpus plan")
    _require_binding(authorization["cache_audit"], "cache audit")
    selection_config = _load_yaml(SELECTION_CONFIG)
    registry_path = _path(selection_config["registry"])
    if authorization["candidate_registry"] != _binding(registry_path):
        raise ConfirmationError("authorization is bound to another candidate registry")
    registry = _load_yaml(registry_path)
    corpus = registry["corpus"]
    if authorization["corpus_plan"] != _binding(_path(corpus["plan"])):
        raise ConfirmationError("authorization is bound to another corpus plan")
    if authorization["cache_audit"] != _binding(_path(corpus["cache_audit"])):
        raise ConfirmationError("authorization is bound to another cache audit")
    expected_roots = (str(corpus["data_root"]), str(corpus["cache_root"]))
    if (authorization["data_root"], authorization["cache_root"]) != expected_roots:
        raise ConfirmationError("authorization data/cache roots changed")
    return authorization, roster, selection


def _assert_no_training_processes() -> None:
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"train_production_student.py" in command:
            raise ConfirmationError(
                f"training process {entry.name} is active; confirmation access is forbidden"
            )


def _load_models(roster: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], Any]]:
    loaded = []
    for row in roster["finalists"]:
        config_path = _path(row["config"]["path"])
        checkpoint_path = _path(row["checkpoint"]["path"])
        config = load_config(config_path)
        if config.get("training", {}).get("precision") != "bfloat16":
            raise ConfirmationError(f"{row['finalist_id']}: confirmation requires frozen BF16")
        model = build_student(config["student"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise ConfirmationError(f"{row['finalist_id']}: checkpoint is not a state dict")
        model.load_state_dict(state, strict=True)
        model.eval()
        loaded.append((row, config, model))
    return loaded


def _load_confirmation_corpora(
    authorization: dict[str, Any], split: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    plan = _load_json(_path(authorization["corpus_plan"]["path"]))
    entries = {str(row["dataset"]): row for row in split["datasets"]}
    selection_config = _load_yaml(SELECTION_CONFIG)
    batch_sizes = selection_config["development_evaluation"]["batch_size_by_context"]
    corpora = [
        trainer._load_corpus(
            item,
            data_root=_path(authorization["data_root"]),
            cache_root=_path(authorization["cache_root"]),
            validation_fraction=float(split["outer_split"]["validation_fraction"]),
            validation_mode=str(split["outer_split"]["mode"]),
            seed=int(split["outer_split"]["seed"]),
            batch_sizes=batch_sizes,
            selection_manifest=split,
            selection_entry=entries[str(item["dataset"])],
            validation_partition="confirmation",
        )
        for item in plan["datasets"]
    ]
    eligible, excluded = [], []
    eligible_observed_targets = 0
    for corpus in corpora:
        entry = entries[corpus.name]
        declared = int(entry["partitions"]["confirmation"]["count"])
        is_eligible = entry["confirmation_eligible"]
        if not isinstance(is_eligible, bool) or is_eligible != (declared > 0):
            raise ConfirmationError(f"{corpus.name}: eligibility/count mismatch")
        if len(corpus.validation_indices) != declared:
            raise ConfirmationError(f"{corpus.name}: materialized confirmation count mismatch")
        if is_eligible:
            observed = trainer._observed_target_count(corpus, corpus.validation_indices)
            if observed <= 0:
                raise ConfirmationError(f"{corpus.name}: confirmation has no observed targets")
            eligible_observed_targets += observed
            eligible.append(corpus)
        else:
            excluded.append(
                {
                    "dataset": corpus.name,
                    "reason": str(entry["inner_split_report"]["reason"]),
                    "declared_confirmation_windows": declared,
                }
            )
    windows = sum(len(corpus.validation_indices) for corpus in eligible)
    if len(eligible) != 68 or windows != 50_318 or len(excluded) != 9:
        raise ConfirmationError("materialized confirmation scope is not exactly 68/50318/9")
    scope = {
        "eligible_dataset_count": len(eligible),
        "eligible_window_count": windows,
        "eligible_observed_target_count": eligible_observed_targets,
        "excluded_dataset_count": len(excluded),
        "excluded_datasets": excluded,
        "aggregation": "eligible datasets only; no backfill",
    }
    return eligible, scope


def _command_authorize(args: argparse.Namespace) -> int:
    payload = _authorization_payload(args.roster.resolve())
    _atomic_json_no_clobber(AUTHORIZATION, payload)
    print(AUTHORIZATION)
    print("Commit this authorization unchanged before running evaluate.")
    return 0


def _command_schema(_: argparse.Namespace) -> int:
    schema = {
        "top_level_exact_keys": sorted(ROSTER_KEYS),
        "schema_version": 2,
        "required_status": "full_training_finalist_roster_frozen",
        "required_protocol_id": EXPECTED_PROTOCOL,
        "screen_selection": {"path": "repository-relative", "sha256": "lowercase SHA-256"},
        "derivative_allowlist": _binding(DERIVATIVE_ALLOWLIST),
        "confirmation_policy_exact_value": CONFIRMATION_POLICY,
        "finalist_entry_exact_keys": sorted(FINALIST_KEYS),
        "development_selection_evidence_exact_keys": sorted(DEVELOPMENT_SELECTION_KEYS),
        "artifact_binding_fields": ["path", "sha256"],
        "maximum_roster_models": MAXIMUM_ROSTER_MODELS,
        "constraints": [
            "every frozen screen finalist is represented",
            "each source_screen_finalist_sha256 binds the entire selected deployment row",
            "config/variant/model source/inference code have no allowed derivative",
            "training commit contains the frozen selection and derivative allowlist",
            "config/data/split/cache/code/initialization/checkpoint hashes are recomputed",
            "candidate/seed pairs and checkpoint hashes are unique",
            "training result is successful DEVELOPMENT-only evidence",
            "checkpoint selection step does not exceed the frozen training budget",
        ],
    }
    print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


def _command_provenance_probes(_: argparse.Namespace) -> int:
    """Exercise provenance tamper gates without reading data or using a GPU."""

    allowlist = _validate_derivative_allowlist()
    candidate_id = "S3"
    derivative = allowlist["derivatives"][candidate_id]
    selected = {
        "candidate_id": candidate_id,
        "variant": derivative["variant"],
        "config": derivative["config"],
        "deployment_fingerprint_sha256": derivative["deployment_fingerprint_sha256"],
        "checkpoint": {"path": "unused", "sha256": "0" * 64},
        "model_source": _source_tree(),
        "inference_implementation": _binding(
            ROOT / "scripts/train_production_student.py"
        ),
    }
    _validate_selected_derivative(selected, candidate_id, allowlist)
    blocked = []
    for field, replacement in (
        ("variant", "relabelled_recipe"),
        ("config", allowlist["derivatives"]["S1"]["config"]),
        ("deployment_fingerprint_sha256", "f" * 64),
    ):
        tampered = json.loads(json.dumps(selected))
        tampered[field] = replacement
        try:
            _validate_selected_derivative(tampered, candidate_id, allowlist)
        except ConfirmationError:
            blocked.append(f"selected_{field}_tamper")
        else:
            raise ConfirmationError(f"selected {field} tamper was accepted")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binding = _binding(DERIVATIVE_ALLOWLIST)
    _validate_git_artifacts([binding], commit, {binding["path"]})
    bad_binding = {**binding, "sha256": "f" * 64}
    try:
        _validate_git_artifacts([bad_binding], commit, {binding["path"]})
    except ConfirmationError:
        blocked.append("training_git_blob_hash_tamper")
    else:
        raise ConfirmationError("training Git-blob hash tamper was accepted")
    print(
        json.dumps(
            {
                "status": "passed",
                "confirmation_or_gift_targets_accessed": False,
                "gpu_accessed": False,
                "blocked_tampers": blocked,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    if BURN_RECORD.exists() or RESULT.exists():
        raise ConfirmationError("confirmation was already attempted or completed")
    authorization, roster, _ = _validate_authorization()
    models = _load_models(roster)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ConfirmationError("confirmation evaluation requires one explicit CUDA device")
    torch.cuda.set_device(device)
    _assert_no_training_processes()

    authorization_sha256 = _sha256(AUTHORIZATION)
    burn = {
        "schema_version": 1,
        "status": "confirmation_access_burned",
        "protocol_id": EXPECTED_PROTOCOL,
        "burned_at_utc": _now(),
        "authorization": {
            "path": _relative(AUTHORIZATION),
            "sha256": authorization_sha256,
            "payload_sha256": authorization["authorization_payload_sha256"],
        },
        "frozen_roster": authorization["frozen_roster"],
        "frozen_screen_selection": authorization["frozen_screen_selection"],
        "derivative_allowlist": authorization["derivative_allowlist"],
        "selection_split": authorization["selection_split"],
        "expected_scope": {
            "eligible_dataset_count": 68,
            "eligible_window_count": 50_318,
            "excluded_dataset_count": 9,
        },
        "operation": "single evaluation-only pass over the complete frozen roster",
        "storage": BURN_LOCATOR,
        "result": _relative(RESULT),
        "retry_permitted": False,
    }
    burn["burn_payload_sha256"] = _canonical_sha256(burn)
    # This durable one-shot record is created before _load_corpus can read any
    # source target values. It also closes training before the process scan, so
    # a new trainer cannot enter between the scan and the burn. A crash or an
    # already-active trainer after this point consumes confirmation safely.
    _atomic_json_no_clobber(BURN_RECORD, burn)
    _assert_no_training_processes()

    split = _validate_selection_split()
    corpora, scope = _load_confirmation_corpora(authorization, split)
    evaluations = []
    for row, config, model in models:
        before = trainer._state_sha256(model)
        model.to(device)
        metrics = trainer._validate(
            model,
            corpora,
            device,
            str(config["training"]["input_preprocessing"]),
            dict(config["inference"]),
        )
        model.cpu()
        after = trainer._state_sha256(model)
        if before != after:
            raise ConfirmationError(f"{row['finalist_id']}: evaluation mutated model state")
        evaluations.append(
            {
                "finalist_id": row["finalist_id"],
                "source_candidate_id": row["source_candidate_id"],
                "source_screen_finalist_sha256": row[
                    "source_screen_finalist_sha256"
                ],
                "full_training_authority_sha256": row[
                    "full_training_authority_sha256"
                ],
                "training_seed": row["training_seed"],
                "config": row["config"],
                "checkpoint": row["checkpoint"],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "state_sha256_before_and_after": before,
                "metrics": metrics,
            }
        )
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "status": "confirmation_evaluation_completed",
        "protocol_id": EXPECTED_PROTOCOL,
        "completed_at_utc": _now(),
        "authorization": _binding(AUTHORIZATION),
        "burn_record": {**BURN_LOCATOR, "sha256": _sha256(BURN_RECORD)},
        "frozen_roster": authorization["frozen_roster"],
        "frozen_screen_selection": authorization["frozen_screen_selection"],
        "scope": scope,
        "evaluation_count": len(evaluations),
        "evaluations_in_frozen_roster_order": evaluations,
        "selection_or_ranking_performed": False,
        "training_or_checkpoint_write_performed": False,
        "learning_rate_or_early_stopping_accessed": False,
        "gift_eval_data_accessed": False,
        "confirmation_partition_accessed": True,
        "device": str(device),
    }
    result["result_payload_sha256"] = _canonical_sha256(result)
    _atomic_json_no_clobber(RESULT, result)
    print(RESULT)
    return 0


def main() -> int:
    os.environ.pop("GIFT_EVAL", None)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema = subparsers.add_parser("roster-schema", help="print the exact frozen-roster contract")
    schema.set_defaults(function=_command_schema)
    probes = subparsers.add_parser(
        "provenance-probes", help="run data-free/GPU-free provenance tamper probes"
    )
    probes.set_defaults(function=_command_provenance_probes)
    authorize = subparsers.add_parser(
        "authorize", help="bind a committed full-training finalist roster"
    )
    authorize.add_argument("--roster", type=Path, required=True)
    authorize.set_defaults(function=_command_authorize)
    evaluate = subparsers.add_parser("evaluate", help="burn and execute the sole confirmation pass")
    evaluate.add_argument("--device", required=True, help="one explicit CUDA device, e.g. cuda:0")
    evaluate.set_defaults(function=_command_evaluate)
    args = parser.parse_args()
    try:
        return int(args.function(args))
    except (
        ConfirmationError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
