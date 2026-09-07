#!/usr/bin/env python3
"""Create and validate append-only full-training finalist attempt lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_ROOT = ROOT / "results/reproduction/distillation/finalist-training-lineage"
PROTOCOL = "timesfm3-performance-recovery-v1.2"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class LineageError(RuntimeError):
    """A full-training attempt is not part of the immutable external lineage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as error:
        raise LineageError(f"path escapes repository: {path}") from error


def _path(value: str) -> Path:
    if Path(value).is_absolute():
        raise LineageError(f"lineage path must be repository-relative: {value}")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise LineageError(f"lineage path escapes repository: {value}") from error
    return path


def _binding(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _require_binding(binding: Any, label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise LineageError(f"{label} must contain exactly path and sha256")
    if not SHA256.fullmatch(str(binding["sha256"])):
        raise LineageError(f"{label} SHA-256 is invalid")
    path = _path(str(binding["path"]))
    if not path.is_file() or _sha256(path) != binding["sha256"]:
        raise LineageError(f"{label} file/hash binding failed")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise LineageError(f"expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise LineageError(f"expected YAML mapping: {path}")
    return value


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob_sha256(commit: str, path: str) -> str:
    if not COMMIT.fullmatch(commit):
        raise LineageError("lineage commit is invalid")
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if blob.returncode:
        raise LineageError(f"lineage commit lacks {path}")
    return hashlib.sha256(blob.stdout).hexdigest()


def _require_attempt_commit(path: Path, record: dict[str, Any], commit: str) -> None:
    relative = _relative(path)
    history = subprocess.run(
        ["git", "log", "--format=%H", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if history != [commit]:
        raise LineageError("attempt record is not an append-only Git addition")
    if _git_blob_sha256(commit, relative) != _sha256(path):
        raise LineageError("attempt record differs from its training commit")
    source = str(record["source_authority_commit"])
    parent = subprocess.run(
        ["git", "rev-parse", f"{commit}^"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if parent.returncode or parent.stdout.strip() != source:
        raise LineageError("attempt record was not committed directly over its authority")
    changed = subprocess.run(
        ["git", "diff", "--name-only", source, commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != [relative]:
        raise LineageError("attempt commit changed files beyond its append-only record")


def _require_completion_commit(path: Path, completion: dict[str, Any]) -> str:
    relative = _relative(path)
    history = subprocess.run(
        ["git", "log", "--format=%H", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(history) != 1:
        raise LineageError("completion is not an append-only Git addition")
    commit = history[0]
    if _git_blob_sha256(commit, relative) != _sha256(path):
        raise LineageError("completion changed after its append-only creation commit")
    parent = subprocess.run(
        ["git", "rev-parse", f"{commit}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if parent != completion["attempt_training_commit"]:
        raise LineageError("completion was not committed directly after its attempt")
    changed = subprocess.run(
        ["git", "diff", "--name-only", parent, commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != [relative]:
        raise LineageError("completion commit changed files beyond its append-only record")
    return commit


def _require_tracked_clean(path: Path, label: str) -> None:
    relative = _relative(path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked.returncode or dirty:
        raise LineageError(f"{label} is not committed and clean")


def _state_dict_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise LineageError("model state dict contains an unsupported entry")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(node: Any) -> None:
        if torch.is_tensor(node):
            tensor = node.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(node, dict):
            digest.update(b"dict\0")
            for key in sorted(node, key=lambda item: repr(item)):
                update(key)
                update(node[key])
        elif isinstance(node, (list, tuple)):
            digest.update(type(node).__name__.encode() + b"\0")
            for item in node:
                update(item)
        elif node is None or isinstance(node, (str, int, float, bool)):
            digest.update(type(node).__name__.encode() + b"\0")
            digest.update(json.dumps(node, sort_keys=True).encode())
        else:
            raise LineageError(f"unsupported checkpoint value: {type(node).__name__}")

    update(value)
    return digest.hexdigest()


def checkpoint_fingerprints(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model",
        "optimizer",
        "step",
        "learning_curve",
        "training_recipe_sha256",
        "initialization_origin_sha256",
        "training_sequence_sha256",
        "full_training_launch_authority",
        "full_training_launch_authority_sha256",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise LineageError("resume checkpoint lacks required state")
    launch = payload["full_training_launch_authority"]
    launch_sha256 = payload["full_training_launch_authority_sha256"]
    if (
        not isinstance(launch, dict)
        or not SHA256.fullmatch(str(launch_sha256))
        or launch_sha256 != _canonical_sha256(launch)
    ):
        raise LineageError("resume checkpoint launch authority is missing or corrupt")
    return {
        "checkpoint_sha256": _sha256(path),
        "step": int(payload["step"]),
        "model_state_sha256": _state_dict_sha256(payload["model"]),
        "optimizer_state_sha256": _tree_sha256(payload["optimizer"]),
        "learning_curve_sha256": _canonical_sha256(payload["learning_curve"]),
        "training_recipe_sha256": str(payload["training_recipe_sha256"]),
        "initialization_origin_sha256": str(payload["initialization_origin_sha256"]),
        "training_sequence_sha256": str(payload["training_sequence_sha256"]),
        "full_training_launch_authority_sha256": str(launch_sha256),
    }


def _validate_payload(record: dict[str, Any], keys: set[str], label: str) -> None:
    if set(record) != keys:
        raise LineageError(f"{label} has an unexpected schema")
    claimed = record.pop("payload_sha256")
    if claimed != _canonical_sha256(record):
        raise LineageError(f"{label} payload hash failed")
    record["payload_sha256"] = claimed


ATTEMPT_KEYS = {
    "schema_version",
    "status",
    "protocol_id",
    "created_at_utc",
    "source_authority_commit",
    "run_key",
    "attempt",
    "source_candidate_id",
    "variant",
    "training_seed",
    "training_budget_steps",
    "config",
    "screen_selection",
    "derivative_allowlist",
    "checkpoint_dir",
    "output",
    "root_initialization",
    "previous_completion",
    "resume_checkpoint",
    "command",
    "payload_sha256",
}
COMPLETION_KEYS = {
    "schema_version",
    "status",
    "protocol_id",
    "completed_at_utc",
    "run_key",
    "attempt",
    "attempt_record",
    "attempt_training_commit",
    "exit_code",
    "resume_checkpoint",
    "resume_checkpoint_fingerprints",
    "best_checkpoint",
    "result",
    "final_checkpoint",
    "payload_sha256",
}


def _validate_attempt_schema(attempt: dict[str, Any]) -> None:
    if (
        attempt["schema_version"] != 1
        or attempt["status"] != "prepared_append_only"
        or attempt["protocol_id"] != PROTOCOL
        or not COMMIT.fullmatch(str(attempt["source_authority_commit"]))
        or isinstance(attempt["attempt"], bool)
        or not isinstance(attempt["attempt"], int)
        or int(attempt["attempt"]) < 1
        or isinstance(attempt["training_seed"], bool)
        or not isinstance(attempt["training_seed"], int)
        or isinstance(attempt["training_budget_steps"], bool)
        or not isinstance(attempt["training_budget_steps"], int)
        or int(attempt["training_budget_steps"]) < 1
        or not isinstance(attempt["run_key"], str)
        or not attempt["run_key"]
        or not isinstance(attempt["command"], list)
        or not attempt["command"]
        or not all(isinstance(value, str) for value in attempt["command"])
    ):
        raise LineageError("attempt record status/protocol/types are invalid")
    for key in ("config", "screen_selection", "derivative_allowlist"):
        binding = attempt[key]
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not SHA256.fullmatch(str(binding["sha256"]))
        ):
            raise LineageError(f"attempt {key} binding is invalid")
        _path(str(binding["path"]))
    root = attempt["root_initialization"]
    if (
        not isinstance(root, dict)
        or set(root) != {"mode", "kind", "checkpoint"}
        or root["mode"]
        not in {
            "selected_screen_checkpoint_weights_only",
            "candidate_registry_initialization",
        }
        or root["kind"] not in {"checkpoint", "seeded_random"}
    ):
        raise LineageError("attempt root initialization is invalid")
    if root["kind"] == "checkpoint":
        binding = root["checkpoint"]
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not SHA256.fullmatch(str(binding["sha256"]))
        ):
            raise LineageError("attempt root checkpoint binding is invalid")
        _path(str(binding["path"]))
    elif root["checkpoint"] is not None:
        raise LineageError("seeded-random root must not contain a checkpoint")


def _run_artifact_paths(
    source_candidate_id: str, training_seed: int, attempt: int
) -> tuple[Path, Path, Path]:
    if not re.fullmatch(r"S[1-5]", source_candidate_id):
        raise LineageError("attempt candidate identifier is invalid")
    checkpoint_root = (
        ROOT
        / "checkpoints/performance-recovery/finalists"
        / source_candidate_id
        / f"seed{training_seed}"
    )
    checkpoint_dir = checkpoint_root / f"attempt{attempt:02d}"
    output = (
        ROOT
        / "results/reproduction/distillation"
        / (
            f"performance-recovery-finalist-{source_candidate_id}-seed{training_seed}"
            f"-attempt{attempt:02d}.json"
        )
    )
    return checkpoint_root, checkpoint_dir, output


def _launch_lineage_payload(
    attempt_path: Path,
    attempt: dict[str, Any],
    training_commit: str,
    chain: list[dict[str, str]],
) -> dict[str, Any]:
    if int(attempt["attempt"]) == 1:
        resume_fingerprints = None
        predecessor_best = None
    else:
        previous, _, _ = _validate_completion(attempt["previous_completion"])
        resume_fingerprints = previous["resume_checkpoint_fingerprints"]
        predecessor_best = previous["best_checkpoint"]
    return {
        "attempt_record": _binding(attempt_path),
        "attempt": int(attempt["attempt"]),
        "run_key": attempt["run_key"],
        "training_commit": training_commit,
        "chain_artifacts": chain,
        "resume_checkpoint_fingerprints": resume_fingerprints,
        "predecessor_best_checkpoint": predecessor_best,
        "root_initialization": attempt["root_initialization"],
    }


def _validate_launch_authority(
    *,
    launch: Any,
    launch_sha256: Any,
    origin: Any,
    origin_sha256: Any,
    training_recipe: Any,
    training_recipe_sha256: Any,
    attempt: dict[str, Any],
    launch_lineage: dict[str, Any],
) -> None:
    if (
        not isinstance(launch, dict)
        or launch_sha256 != _canonical_sha256(launch)
        or launch.get("schema_version") != 1
        or launch.get("status") != "frozen_full_training_launch_authority"
        or launch.get("protocol_id") != PROTOCOL
        or launch.get("source_candidate_id") != attempt["source_candidate_id"]
        or launch.get("variant") != attempt["variant"]
        or launch.get("training_commit") != launch_lineage["training_commit"]
        or launch.get("attempt_lineage") != launch_lineage
        or launch.get("config") != attempt["config"]
        or launch.get("initialization_mode") != attempt["root_initialization"]["mode"]
        or not isinstance(origin, dict)
        or origin_sha256 != _canonical_sha256(origin)
        or launch.get("initialization_origin_fingerprint") != origin
        or launch.get("initialization_origin_sha256") != origin_sha256
        or training_recipe_sha256 != launch.get("training_recipe_sha256")
        or training_recipe != launch.get("training_recipe_fingerprint")
    ):
        raise LineageError("resume checkpoint does not belong to its external attempt")
    derivative_binding = launch.get("derivative_allowlist", {})
    selection_binding = launch.get("screen_selection", {})
    if (
        {
            "path": derivative_binding.get("path"),
            "sha256": derivative_binding.get("sha256"),
        }
        != attempt["derivative_allowlist"]
        or {
            "path": selection_binding.get("path"),
            "sha256": selection_binding.get("sha256"),
        }
        != attempt["screen_selection"]
    ):
        raise LineageError("resume checkpoint selection authority differs from its attempt")
    root = attempt["root_initialization"]
    if root["kind"] == "checkpoint":
        expected_checkpoint = root["checkpoint"]
        if (
            origin.get("kind") != "checkpoint"
            or origin.get("initialization_checkpoint_sha256")
            != expected_checkpoint["sha256"]
            or Path(str(origin.get("initialization_checkpoint", ""))).resolve()
            != _path(expected_checkpoint["path"])
        ):
            raise LineageError("resume checkpoint initialization differs from lineage root")
    elif (
        root["kind"] != "seeded_random"
        or origin.get("kind") != "seeded_random"
        or origin.get("initialization_checkpoint") is not None
        or origin.get("initialization_checkpoint_sha256") is not None
    ):
        raise LineageError("resume checkpoint seeded initialization differs from lineage root")


def _validate_checkpoint_launch(
    checkpoint_path: Path,
    attempt: dict[str, Any],
    launch_lineage: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise LineageError("resume checkpoint payload is malformed")
    _validate_launch_authority(
        launch=payload.get("full_training_launch_authority"),
        launch_sha256=payload.get("full_training_launch_authority_sha256"),
        origin=payload.get("initialization_origin_fingerprint"),
        origin_sha256=payload.get("initialization_origin_sha256"),
        training_recipe=payload.get("training_recipe_fingerprint"),
        training_recipe_sha256=payload.get("training_recipe_sha256"),
        attempt=attempt,
        launch_lineage=launch_lineage,
    )
    return checkpoint_fingerprints(checkpoint_path)


def _validate_completion(
    binding: dict[str, str],
) -> tuple[dict[str, Any], Path, str]:
    path = _require_binding(binding, "preceding attempt completion")
    _require_tracked_clean(path, "preceding attempt completion")
    completion = _load_json(path)
    _validate_payload(completion, COMPLETION_KEYS, "attempt completion")
    if (
        completion["schema_version"] != 1
        or completion["protocol_id"] != PROTOCOL
        or completion["status"] not in {"interrupted_resumable", "succeeded"}
    ):
        raise LineageError("preceding completion status/protocol is invalid")
    completion_commit = _require_completion_commit(path, completion)
    attempt_path = _require_binding(completion["attempt_record"], "completed attempt record")
    attempt = _load_json(attempt_path)
    _validate_payload(attempt, ATTEMPT_KEYS, "completed attempt record")
    _validate_attempt_schema(attempt)
    _require_attempt_commit(
        attempt_path, attempt, str(completion["attempt_training_commit"])
    )
    if (
        attempt["run_key"] != completion["run_key"]
        or attempt["attempt"] != completion["attempt"]
    ):
        raise LineageError("completion disagrees with its attempt record")
    if completion["status"] == "interrupted_resumable":
        if int(completion["exit_code"]) == 0:
            raise LineageError("interrupted completion records a successful exit code")
        resume = _require_binding(completion["resume_checkpoint"], "resume checkpoint")
        if checkpoint_fingerprints(resume) != completion["resume_checkpoint_fingerprints"]:
            raise LineageError("resume model/optimizer/curve/origin differs from completion")
        _require_binding(completion["best_checkpoint"], "preceding best checkpoint")
        if completion["result"] is not None or completion["final_checkpoint"] is not None:
            raise LineageError("interrupted completion contains terminal artifacts")
    else:
        if int(completion["exit_code"]) != 0:
            raise LineageError("successful completion records a failing exit code")
        _require_binding(completion["best_checkpoint"], "successful best checkpoint")
        result_path = _require_binding(completion["result"], "successful training result")
        _require_binding(completion["final_checkpoint"], "successful final checkpoint")
        result = _load_json(result_path)
        if (
            result.get("status") != "succeeded"
            or result.get("git_commit") != completion["attempt_training_commit"]
        ):
            raise LineageError("successful completion result/commit is invalid")
        if completion["resume_checkpoint"] is not None:
            resume = _require_binding(completion["resume_checkpoint"], "terminal resume checkpoint")
            if checkpoint_fingerprints(resume) != completion["resume_checkpoint_fingerprints"]:
                raise LineageError("terminal resume checkpoint changed")
        elif completion["resume_checkpoint_fingerprints"] is not None:
            raise LineageError("terminal completion has fingerprints without a checkpoint")
    return completion, attempt_path, completion_commit


def _validate_chain(
    attempt_path: Path,
    attempt: dict[str, Any],
    training_commit: str,
) -> list[dict[str, str]]:
    _validate_attempt_schema(attempt)
    for key in ("config", "screen_selection", "derivative_allowlist"):
        _require_binding(attempt[key], f"attempt {key}")
    if attempt["root_initialization"]["kind"] == "checkpoint":
        _require_binding(
            attempt["root_initialization"]["checkpoint"],
            "attempt root initialization checkpoint",
        )
    _, expected_checkpoint_dir, expected_output = _run_artifact_paths(
        str(attempt["source_candidate_id"]),
        int(attempt["training_seed"]),
        int(attempt["attempt"]),
    )
    if (
        attempt["checkpoint_dir"] != _relative(expected_checkpoint_dir)
        or attempt["output"] != _relative(expected_output)
    ):
        raise LineageError("attempt does not use its immutable per-attempt output paths")
    _require_attempt_commit(attempt_path, attempt, training_commit)
    bindings = [_binding(attempt_path)]
    number = int(attempt["attempt"])
    previous = attempt["previous_completion"]
    if number == 1:
        if previous is not None or attempt["resume_checkpoint"] is not None:
            raise LineageError("first full-training attempt must not resume")
        return bindings
    if number <= 1 or previous is None:
        raise LineageError("later attempt lacks its immediately preceding completion")
    completion, previous_attempt_path, completion_commit = _validate_completion(previous)
    if completion_commit != attempt["source_authority_commit"]:
        raise LineageError(
            "predecessor completion was not the exact launch authority of this attempt"
        )
    if (
        completion["status"] != "interrupted_resumable"
        or int(completion["attempt"]) != number - 1
        or completion["run_key"] != attempt["run_key"]
        or attempt["resume_checkpoint"] != completion["resume_checkpoint"]
    ):
        raise LineageError("resume does not use the immediate resumable predecessor")
    previous_attempt = _load_json(previous_attempt_path)
    _validate_payload(previous_attempt, ATTEMPT_KEYS, "preceding attempt record")
    stable = {
        "run_key",
        "source_candidate_id",
        "variant",
        "training_seed",
        "training_budget_steps",
        "config",
        "screen_selection",
        "derivative_allowlist",
        "root_initialization",
    }
    if any(attempt[key] != previous_attempt[key] for key in stable):
        raise LineageError("resume lineage changed a frozen run attribute")
    previous_chain = _validate_chain(
        previous_attempt_path,
        previous_attempt,
        str(completion["attempt_training_commit"]),
    )
    previous_launch = _launch_lineage_payload(
        previous_attempt_path,
        previous_attempt,
        str(completion["attempt_training_commit"]),
        previous_chain,
    )
    _validate_checkpoint_launch(
        _require_binding(completion["resume_checkpoint"], "predecessor resume checkpoint"),
        previous_attempt,
        previous_launch,
    )
    bindings.extend([previous, *previous_chain])
    if len({binding["path"] for binding in bindings}) != len(bindings):
        raise LineageError("resume lineage contains a cycle or duplicate")
    return bindings


def validate_attempt_for_launch(
    *,
    attempt_path: Path,
    source_candidate_id: str,
    variant: str,
    training_seed: int,
    training_budget_steps: int,
    config: dict[str, str],
    screen_selection: dict[str, str],
    derivative_allowlist: dict[str, str],
    checkpoint_dir: Path,
    output: Path,
    allowed_initializations: list[dict[str, Any]],
    resume: Path | None,
) -> dict[str, Any]:
    _require_tracked_clean(attempt_path, "current full-training attempt record")
    attempt = _load_json(attempt_path)
    _validate_payload(attempt, ATTEMPT_KEYS, "current attempt record")
    _validate_attempt_schema(attempt)
    expected = {
        "source_candidate_id": source_candidate_id,
        "variant": variant,
        "training_seed": training_seed,
        "training_budget_steps": training_budget_steps,
        "config": config,
        "screen_selection": screen_selection,
        "derivative_allowlist": derivative_allowlist,
        "checkpoint_dir": _relative(checkpoint_dir),
        "output": _relative(output),
    }
    if any(attempt[key] != value for key, value in expected.items()):
        raise LineageError("attempt record differs from requested full-training launch")
    if attempt["root_initialization"] not in allowed_initializations:
        raise LineageError("attempt root initialization is not allowlisted")
    commit = _head()
    chain = _validate_chain(attempt_path, attempt, commit)
    checkpoint_root, expected_checkpoint_dir, expected_output = _run_artifact_paths(
        source_candidate_id, training_seed, int(attempt["attempt"])
    )
    if checkpoint_dir.resolve() != expected_checkpoint_dir or output.resolve() != expected_output:
        raise LineageError("attempt output paths are not the immutable per-attempt paths")
    existing = ([_relative(output)] if output.exists() else []) + (
        [_relative(path) for path in checkpoint_dir.iterdir()]
        if checkpoint_dir.exists()
        else []
    )
    if existing:
        raise LineageError(f"attempt refuses pre-existing output artifacts: {existing}")
    if int(attempt["attempt"]) == 1:
        if resume is not None:
            raise LineageError("resume is forbidden on the first full-training attempt")
        result_prefix = (
            f"performance-recovery-finalist-{source_candidate_id}-seed{training_seed}"
        )
        stale = (
            list(checkpoint_root.rglob("*")) if checkpoint_root.exists() else []
        ) + list(
            (ROOT / "results/reproduction/distillation").glob(
                f"{result_prefix}*.json"
            )
        )
        if stale:
            raise LineageError(
                f"first attempt refuses pre-existing run artifacts: {stale}"
            )
        resume_fingerprints = None
        predecessor_best = None
    else:
        expected_resume = _require_binding(
            attempt["resume_checkpoint"], "current attempt resume checkpoint"
        )
        if resume is None or resume.resolve() != expected_resume:
            raise LineageError("trainer resume path differs from external lineage")
        previous, previous_attempt_path, _ = _validate_completion(
            attempt["previous_completion"]
        )
        previous_attempt = _load_json(previous_attempt_path)
        _validate_payload(previous_attempt, ATTEMPT_KEYS, "predecessor attempt record")
        previous_chain = _validate_chain(
            previous_attempt_path,
            previous_attempt,
            str(previous["attempt_training_commit"]),
        )
        resume_fingerprints = _validate_checkpoint_launch(
            expected_resume,
            previous_attempt,
            _launch_lineage_payload(
                previous_attempt_path,
                previous_attempt,
                str(previous["attempt_training_commit"]),
                previous_chain,
            ),
        )
        if resume_fingerprints != previous["resume_checkpoint_fingerprints"]:
            raise LineageError("resume checkpoint content changed after predecessor completion")
        predecessor_best = previous["best_checkpoint"]
    launch = _launch_lineage_payload(attempt_path, attempt, commit, chain)
    if (
        launch["resume_checkpoint_fingerprints"] != resume_fingerprints
        or launch["predecessor_best_checkpoint"] != predecessor_best
    ):
        raise LineageError("attempt lineage reconstruction changed")
    return launch


def validate_completed_lineage(
    completion_binding: dict[str, str],
) -> dict[str, Any]:
    completion, attempt_path, _ = _validate_completion(completion_binding)
    attempt = _load_json(attempt_path)
    _validate_payload(attempt, ATTEMPT_KEYS, "final attempt record")
    chain = _validate_chain(
        attempt_path, attempt, str(completion["attempt_training_commit"])
    )
    launch_lineage = _launch_lineage_payload(
        attempt_path,
        attempt,
        str(completion["attempt_training_commit"]),
        chain,
    )
    if completion["resume_checkpoint"] is not None:
        observed = _validate_checkpoint_launch(
            _require_binding(completion["resume_checkpoint"], "terminal resume checkpoint"),
            attempt,
            launch_lineage,
        )
        if observed != completion["resume_checkpoint_fingerprints"]:
            raise LineageError("terminal checkpoint differs from its external completion")
    return {
        "completion": completion,
        "attempt": attempt,
        "chain_artifacts": chain,
        "launch_lineage": launch_lineage,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise LineageError(f"refusing to overwrite lineage artifact: {path}")
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
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _assert_clean_tracked_tree() -> None:
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise LineageError("tracked worktree must be clean before appending lineage")


def _command_prepare(args: argparse.Namespace) -> int:
    _assert_clean_tracked_tree()
    allowlist_path = ROOT / "configs/performance_recovery/finalist_derivatives.yaml"
    allowlist = _load_yaml(allowlist_path)
    authorities = allowlist["authorities"]
    selection_path = _path(authorities["screen_selection"]["path"])
    _require_tracked_clean(allowlist_path, "derivative allowlist")
    _require_tracked_clean(selection_path, "screen selection")
    selection = _load_json(selection_path)
    selected_rows = [
        row
        for row in selection.get("finalists", [])
        if row.get("candidate_id") == args.candidate
    ]
    if len(selected_rows) != 1:
        raise LineageError("candidate is not exactly once in the frozen selection")
    selected = selected_rows[0]
    derivative = allowlist["derivatives"][args.candidate]
    if (
        selected.get("variant") != derivative["variant"]
        or selected.get("config") != derivative["config"]
        or selected.get("deployment_fingerprint_sha256")
        != derivative["deployment_fingerprint_sha256"]
        or args.seed not in allowlist["allowed_runtime_differences"]["training_seed"]
        or args.max_steps not in allowlist["allowed_runtime_differences"]["maximum_steps"]
    ):
        raise LineageError("requested derivative is outside the frozen selection/allowlist")
    run_key = f"{args.candidate}-seed{args.seed}-steps{args.max_steps}"
    previous_binding = None
    resume_binding = None
    if args.previous_completion is None:
        attempt_number = 1
        if args.initialization_mode == "selected-screen":
            initialization = {
                "mode": "selected_screen_checkpoint_weights_only",
                "kind": "checkpoint",
                "checkpoint": selected["checkpoint"],
            }
        else:
            if derivative["candidate_registry_initialization_allowed"] is not True:
                raise LineageError("candidate-registry initialization is forbidden")
            registry = _load_yaml(_path(authorities["candidate_registry"]["path"]))
            declared = registry["candidates"][args.candidate]["initialization"]
            initialization = {
                "mode": "candidate_registry_initialization",
                "kind": declared["kind"],
                "checkpoint": (
                    {"path": declared["path"], "sha256": declared["file_sha256"]}
                    if declared["kind"] == "checkpoint"
                    else None
                ),
            }
        checkpoint_root, _, _ = _run_artifact_paths(
            args.candidate, args.seed, attempt_number
        )
        result_prefix = f"performance-recovery-finalist-{args.candidate}-seed{args.seed}"
        existing = (
            list(checkpoint_root.rglob("*")) if checkpoint_root.exists() else []
        ) + list((ROOT / "results/reproduction/distillation").glob(f"{result_prefix}*.json"))
        if existing:
            raise LineageError(f"first attempt refuses pre-existing artifacts: {existing}")
    else:
        previous_path = args.previous_completion.resolve()
        previous_binding = _binding(previous_path)
        completion, previous_attempt_path, completion_commit = _validate_completion(
            previous_binding
        )
        if completion_commit != _head():
            raise LineageError(
                "new attempt must be prepared directly from its predecessor completion commit"
            )
        previous_attempt = _load_json(previous_attempt_path)
        _validate_payload(previous_attempt, ATTEMPT_KEYS, "preceding attempt record")
        if (
            completion["status"] != "interrupted_resumable"
            or previous_attempt["run_key"] != run_key
            or previous_attempt["source_candidate_id"] != args.candidate
        ):
            raise LineageError("previous completion is not the immediate run predecessor")
        attempt_number = int(completion["attempt"]) + 1
        initialization = previous_attempt["root_initialization"]
        resume_binding = completion["resume_checkpoint"]
    _, checkpoint_dir, output = _run_artifact_paths(
        args.candidate, args.seed, attempt_number
    )
    existing_current = ([output] if output.exists() else []) + (
        list(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else []
    )
    if existing_current:
        raise LineageError(
            f"attempt refuses pre-existing output artifacts: {existing_current}"
        )
    attempt_path = LINEAGE_ROOT / f"{run_key}-attempt{attempt_number:02d}.json"
    if attempt_path.exists():
        raise LineageError("attempt number/path already exists")
    trainer = _load_yaml(_path(authorities["candidate_registry"]["path"]))["trainer"]
    command = [
        trainer["python"],
        trainer["path"],
        derivative["config"]["path"],
        authorities["corpus_plan"]["path"],
        "--variant",
        derivative["variant"],
        "--training-seed",
        str(args.seed),
        "--split-seed",
        str(allowlist["allowed_runtime_differences"]["split_seed"]),
        "--data-root",
        authorities["data_root"],
        "--cache-root",
        authorities["cache_root"],
        "--checkpoint-dir",
        _relative(checkpoint_dir),
        "--output",
        _relative(output),
        "--max-steps",
        str(args.max_steps),
        "--selection-split-manifest",
        authorities["selection_split"]["path"],
        "--validation-partition",
        "development",
        "--frozen-finalist-selection",
        _relative(selection_path),
        "--full-training-attempt-record",
        _relative(attempt_path),
    ]
    if resume_binding is not None:
        command.extend(("--resume", resume_binding["path"]))
    elif initialization["checkpoint"] is not None:
        command.extend(("--initialize-from", initialization["checkpoint"]["path"]))
    payload = {
        "schema_version": 1,
        "status": "prepared_append_only",
        "protocol_id": PROTOCOL,
        "created_at_utc": _now(),
        "source_authority_commit": _head(),
        "run_key": run_key,
        "attempt": attempt_number,
        "source_candidate_id": args.candidate,
        "variant": derivative["variant"],
        "training_seed": args.seed,
        "training_budget_steps": args.max_steps,
        "config": derivative["config"],
        "screen_selection": _binding(selection_path),
        "derivative_allowlist": _binding(allowlist_path),
        "checkpoint_dir": _relative(checkpoint_dir),
        "output": _relative(output),
        "root_initialization": initialization,
        "previous_completion": previous_binding,
        "resume_checkpoint": resume_binding,
        "command": command,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    _atomic_json(attempt_path, payload)
    print(attempt_path)
    print("Commit only this attempt record, then execute its exact command array.")
    print(json.dumps(command))
    return 0


def _command_finalize(args: argparse.Namespace) -> int:
    _assert_clean_tracked_tree()
    attempt_path = args.attempt_record.resolve()
    _require_tracked_clean(attempt_path, "completed attempt record")
    attempt = _load_json(attempt_path)
    _validate_payload(attempt, ATTEMPT_KEYS, "completed attempt record")
    _validate_attempt_schema(attempt)
    commit = _head()
    chain = _validate_chain(attempt_path, attempt, commit)
    launch_lineage = _launch_lineage_payload(
        attempt_path, attempt, commit, chain
    )
    checkpoint_dir = _path(attempt["checkpoint_dir"])
    variant = str(attempt["variant"])
    resume_path = checkpoint_dir / f"student-{variant}-resume.pt"
    best_path = checkpoint_dir / f"student-{variant}-best.pt"
    final_path = checkpoint_dir / f"student-{variant}-final.pt"
    result_path = _path(attempt["output"])
    if args.exit_code == 0:
        for path in (best_path, final_path, result_path):
            if not path.is_file():
                raise LineageError(f"successful attempt lacks {path}")
        status = "succeeded"
        result = _binding(result_path)
        final = _binding(final_path)
        result_payload = _load_json(result_path)
        extra = result_payload.get("extra", {})
        _validate_launch_authority(
            launch=extra.get("full_training_launch_authority"),
            launch_sha256=extra.get("full_training_launch_authority_sha256"),
            origin=extra.get("initialization_origin_fingerprint"),
            origin_sha256=extra.get("initialization_origin_sha256"),
            training_recipe=extra.get("training_recipe_fingerprint"),
            training_recipe_sha256=extra.get("training_recipe_sha256"),
            attempt=attempt,
            launch_lineage=launch_lineage,
        )
        completed_authority = extra.get("full_training_authority")
        expected_best = _binding(best_path)
        expected_final = _binding(final_path)
        if (
            result_payload.get("status") != "succeeded"
            or result_payload.get("git_commit") != commit
            or not isinstance(completed_authority, dict)
            or extra.get("full_training_authority_sha256")
            != _canonical_sha256(completed_authority)
            or completed_authority.get("status") != "completed_full_training_authority"
            or completed_authority.get("launch")
            != extra.get("full_training_launch_authority")
            or completed_authority.get("launch_sha256")
            != extra.get("full_training_launch_authority_sha256")
            or completed_authority.get("best_checkpoint") != expected_best
            or completed_authority.get("final_checkpoint") != expected_final
            or extra.get("best_checkpoint_sha256") != expected_best["sha256"]
            or extra.get("final_checkpoint_sha256") != expected_final["sha256"]
        ):
            raise LineageError("successful result does not belong to its external attempt")
    else:
        if not resume_path.is_file() or not best_path.is_file() or result_path.exists():
            raise LineageError("interrupted attempt is not externally resumable")
        status = "interrupted_resumable"
        result = None
        final = None
    resume_fingerprints = None
    if resume_path.is_file():
        resume_fingerprints = _validate_checkpoint_launch(
            resume_path, attempt, launch_lineage
        )
    completion_path = attempt_path.with_name(
        f"{attempt_path.stem}-completion.json"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "protocol_id": PROTOCOL,
        "completed_at_utc": _now(),
        "run_key": attempt["run_key"],
        "attempt": attempt["attempt"],
        "attempt_record": _binding(attempt_path),
        "attempt_training_commit": commit,
        "exit_code": args.exit_code,
        "resume_checkpoint": _binding(resume_path) if resume_path.is_file() else None,
        "resume_checkpoint_fingerprints": resume_fingerprints,
        "best_checkpoint": _binding(best_path),
        "result": result,
        "final_checkpoint": final,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    _atomic_json(completion_path, payload)
    print(completion_path)
    print("Commit this completion unchanged before preparing another attempt or confirmation.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="append a frozen launch attempt")
    prepare.add_argument("--candidate", choices=("S1", "S2", "S3", "S4", "S5"), required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--max-steps", type=int, required=True)
    prepare.add_argument(
        "--initialization-mode",
        choices=("selected-screen", "candidate-registry"),
        default="selected-screen",
    )
    prepare.add_argument("--previous-completion", type=Path)
    prepare.set_defaults(function=_command_prepare)
    finalize = subparsers.add_parser("finalize", help="append a terminal attempt completion")
    finalize.add_argument("--attempt-record", type=Path, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.set_defaults(function=_command_finalize)
    args = parser.parse_args()
    try:
        return int(args.function(args))
    except (LineageError, FileExistsError, KeyError, RuntimeError, ValueError) as error:
        print(f"blocked: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
