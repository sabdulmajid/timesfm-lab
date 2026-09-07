#!/usr/bin/env python3
"""Fail-closed launcher and GPU-hour ledger for bounded recovery screens."""

from __future__ import annotations

import argparse
import copy
import ctypes
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import torch
import yaml

from timesfm_lab.models import build_student

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/performance_recovery/candidates.yaml"
DEFAULT_LEDGER = (
    ROOT / "results/reproduction/distillation/performance-recovery-gpu-hour-ledger.json"
)
SLOTS = tuple(f"S{index}" for index in range(1, 7))
PREDECESSOR_PROTOCOL_ID = "timesfm3-performance-recovery-v1.1"
SUCCESSOR_PROTOCOL_ID = "timesfm3-performance-recovery-v1.2"
MIGRATION_EVIDENCE_TYPE = "performance_recovery_gpu_ledger_protocol_migration"
TERMINAL_ATTEMPT_STATUSES = frozenset({"failed", "invalid", "succeeded"})
S5_MAXIMUM_ESTIMATED_PHYSICAL_GPU_HOURS = 29.1
S5_INITIALIZATION_STATE_SHA256 = "ca585678134535639b287ec295347d5103a0f1e21229d44f60913f5495c8f530"
S5_AUTOTUNE_CATEGORY = "s5_exact_throughput_autotune"
S5_AUTOTUNE_CHILD_KIND = "s5_exact_throughput_autotune_child"
S5_AUTOTUNE_TERMINATION_GRACE_SECONDS = 10.0
S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS = 5.0


class GateError(RuntimeError):
    """A launch-safety invariant is not satisfied."""


class _AutotuneWorkerInterrupted(RuntimeError):
    """Internal control flow used to make managed-worker signals auditable."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"autotune worker received signal {signum}")
        self.signum = signum


def _s5_autotune_cleanup_budget_seconds() -> float:
    """Reserve TERM, KILL/reap, and terminal-evidence time inside the hard cap."""

    return (
        2.0 * S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
        + S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_document(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _root_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise GateError(f"path escapes repository: {path}") from error
    return path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _python_executable(value: str) -> Path:
    """Resolve the repository venv entry without following its system-Python symlink."""
    path = Path(value)
    path = (ROOT / path).absolute() if not path.is_absolute() else path.absolute()
    if os.path.commonpath((str(ROOT), str(path))) != str(ROOT) or not path.is_file():
        raise GateError(f"invalid repository Python entry: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise GateError(f"expected mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise GateError(f"expected object in {path}")
    return value


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise GateError(f"expected object in {path}")
    return value, hashlib.sha256(encoded).hexdigest()


def _load_torch_snapshot(path: Path, *, map_location: str = "cpu") -> tuple[Any, str, bytes]:
    """Hash and deserialize exactly one immutable byte snapshot."""

    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        torch.load(io.BytesIO(encoded), map_location=map_location, weights_only=False),
        digest,
        encoded,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json_document(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create immutable JSON, refusing to replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json_document(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise GateError(f"immutable evidence already exists: {path}") from error
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    """Atomically create immutable binary evidence without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise GateError(f"immutable evidence already exists: {path}") from error
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise GateError("JSON pointer must be empty or begin with '/'")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise GateError(f"JSON pointer traverses a scalar at {token!r}")
    return current


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise GateError(f"missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise GateError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _require_full_git_oid(value: Any, label: str) -> str:
    oid = str(value)
    if len(oid) not in {40, 64} or any(character not in "0123456789abcdef" for character in oid):
        raise GateError(f"{label} must be a full lowercase Git object ID")
    return oid


def _git_blob(commit: Any, path: Path) -> tuple[str, bytes]:
    commit_oid = _require_full_git_oid(commit, "reference Git commit")
    relative = _relative(path)
    blob_oid = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit_oid}:{relative}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    _require_full_git_oid(blob_oid, "reference Git blob")
    contents = subprocess.run(
        ["git", "cat-file", "blob", blob_oid],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return blob_oid, contents


def _require_git_ancestor(ancestor: Any, descendant: Any, label: str) -> None:
    ancestor_oid = _require_full_git_oid(ancestor, label)
    descendant_oid = _require_full_git_oid(descendant, "descendant Git commit")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_oid, descendant_oid],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 1:
        raise GateError(f"{label} is not an ancestor of {descendant_oid}")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _load_yaml_bytes(value: bytes, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(value)
    if not isinstance(payload, dict):
        raise GateError(f"expected mapping in {label}")
    return payload


def _require_paths_clean(paths: list[str]) -> None:
    paths = list(dict.fromkeys(paths))
    diff_commands = (
        ["git", "diff", "--quiet", "--", *paths],
        ["git", "diff", "--cached", "--quiet", "--", *paths],
    )
    for arguments in diff_commands:
        result = subprocess.run(arguments, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise GateError("launch-relevant tracked files have uncommitted changes")


def _require_relevant_tree_clean(registry_path: Path, registry: dict[str, Any]) -> None:
    paths = [
        registry["trainer"]["path"],
        _relative(Path(__file__)),
        "src/timesfm_lab",
        _relative(registry_path),
        registry["protocol"]["target_config"],
        registry["corpus"]["plan"],
        registry["corpus"]["cache_audit"],
        registry["selection_split"]["manifest"],
        "scripts/freeze_recovery_selection_split.py",
    ]
    paths.extend(
        candidate["config"]
        for candidate in registry["candidates"].values()
        if candidate.get("launchable")
    )
    for candidate in registry["candidates"].values():
        if not candidate.get("launchable"):
            continue
        activation = candidate.get("activation_evidence")
        if isinstance(activation, dict):
            paths.append(activation["path"])
        autotune = candidate.get("throughput_autotune")
        if isinstance(autotune, dict):
            paths.extend((autotune["implementation"], autotune["config"]))
    _require_paths_clean(paths)


def _verify_selection_reconstruction(registry: dict[str, Any]) -> None:
    expected = registry["selection_split"]["manifest_sha256"]
    with tempfile.TemporaryDirectory(prefix="timesfm-recovery-split-") as directory:
        output = Path(directory) / "split.json"
        command = [
            str(_python_executable(registry["trainer"]["python"])),
            str(ROOT / "scripts/freeze_recovery_selection_split.py"),
            "--plan",
            str(_root_path(registry["corpus"]["plan"])),
            "--cache-root",
            str(_root_path(registry["corpus"]["cache_root"])),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            raise GateError(f"selection split reconstruction failed: {result.stderr.strip()}")
        observed = _sha256(output)
        if observed != expected:
            raise GateError(
                f"reconstructed selection split mismatch: expected {expected}, observed {observed}"
            )


def _verify_model_initialization(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    model = build_student(config["student"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(candidate["parameter_count"]):
        raise GateError(
            "parameter count mismatch: "
            f"declared {candidate['parameter_count']}, observed {parameter_count}"
        )
    initialization = candidate["initialization"]
    if initialization["kind"] == "checkpoint":
        checkpoint = _root_path(initialization["path"])
        _verify_file(checkpoint, initialization["file_sha256"], "initialization checkpoint")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        expected = initialization["loaded_state_sha256"]
    elif initialization["kind"] == "seeded_random":
        expected = initialization["state_sha256"]
    else:
        raise GateError(f"unsupported initialization kind {initialization['kind']!r}")
    observed = _state_sha256(model)
    if observed != expected:
        raise GateError(
            f"initial model-state hash mismatch: expected {expected}, observed {observed}"
        )


def _registry_except_protocol_and_s5(registry: dict[str, Any]) -> dict[str, Any]:
    comparable = {key: value for key, value in registry.items() if key != "protocol"}
    candidates = dict(comparable.get("candidates", {}))
    candidates.pop("S5", None)
    comparable["candidates"] = candidates
    return comparable


def _verify_grandfathered_launch_records(
    registry_path: Path,
    protocol: dict[str, Any],
    *,
    predecessor_protocol_id: str,
    predecessor_registry_sha256: str,
    reference_git_commit: str,
    target_path: Path,
    target_sha256: str,
    deep: bool,
) -> list[dict[str, Any]]:
    records = protocol.get("grandfathered_launch_records")
    if not isinstance(records, list):
        raise GateError("v1.2 must declare its grandfathered launch records")
    expected_keys = {("S3", 1), ("S3", 2), ("S4", 1), ("S4", 2)}
    observed_keys: set[tuple[str, int]] = set()
    verified: list[dict[str, Any]] = []
    for entry in records:
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_id",
            "attempt",
            "path",
            "sha256",
        }:
            raise GateError("invalid grandfathered launch-record declaration")
        slot = str(entry["candidate_id"])
        attempt_number = int(entry["attempt"])
        key = (slot, attempt_number)
        if key in observed_keys:
            raise GateError("duplicate grandfathered launch record")
        observed_keys.add(key)
        launch_path = _root_path(entry["path"])
        _verify_file(launch_path, str(entry["sha256"]), f"{slot} launch attempt {attempt_number}")
        launch = _load_json(launch_path)
        if (
            launch.get("protocol_id") != predecessor_protocol_id
            or launch.get("candidate_id") != slot
        ):
            raise GateError(f"{slot} attempt {attempt_number} has invalid grandfathered identity")
        input_hashes = launch.get("input_hashes")
        if not isinstance(input_hashes, list):
            raise GateError(f"{slot} attempt {attempt_number} has invalid input hashes")
        hashes: dict[str, str] = {}
        for item in input_hashes:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise GateError(f"{slot} attempt {attempt_number} has invalid input hash entry")
            path = str(item["path"])
            if path in hashes:
                raise GateError(f"{slot} attempt {attempt_number} repeats an input path")
            hashes[path] = str(item["sha256"])
        required = {
            _relative(registry_path): predecessor_registry_sha256,
            _relative(target_path): target_sha256,
        }
        if any(hashes.get(path) != expected for path, expected in required.items()):
            raise GateError(f"{slot} attempt {attempt_number} changed its launch authority")
        launch_commit = _require_full_git_oid(
            launch.get("git_commit"), f"{slot} attempt {attempt_number} Git commit"
        )
        _require_git_ancestor(
            launch_commit,
            reference_git_commit,
            f"{slot} attempt {attempt_number} Git commit",
        )
        if deep:
            for relative, expected in hashes.items():
                _, contents = _git_blob(launch_commit, _root_path(relative))
                if hashlib.sha256(contents).hexdigest() != expected:
                    raise GateError(
                        f"{slot} attempt {attempt_number} input differs from its Git snapshot: "
                        f"{relative}"
                    )
        verified.append(
            {
                "candidate_id": slot,
                "attempt": attempt_number,
                "path": _relative(launch_path),
                "sha256": str(entry["sha256"]),
            }
        )
    if observed_keys != expected_keys:
        raise GateError("v1.2 grandfathering must contain exactly S3/S4 attempts 1 and 2")
    return verified


def _verify_protocol_authority(
    registry_path: Path,
    registry: dict[str, Any],
    targets: dict[str, Any],
    *,
    deep: bool,
) -> dict[str, Any] | None:
    """Verify a direct target binding or the one approved v1.1-to-v1.2 overlay."""

    protocol = registry["protocol"]
    protocol_id = str(protocol["id"])
    target_protocol_id = str(targets["protocol_id"])
    target_path = _root_path(protocol["target_config"])
    target_sha256 = _sha256(target_path)
    if protocol_id == target_protocol_id:
        return None
    if (target_protocol_id, protocol_id) != (
        PREDECESSOR_PROTOCOL_ID,
        SUCCESSOR_PROTOCOL_ID,
    ):
        raise GateError("registry/target protocol mismatch")

    expected_target_authority = {
        "protocol_id": PREDECESSOR_PROTOCOL_ID,
        "sha256": target_sha256,
        "policy": "unchanged_by_v1.2",
    }
    if protocol.get("target_authority") != expected_target_authority:
        raise GateError("v1.2 target authority is not the unchanged v1.1 target")
    supersedes = protocol.get("supersedes")
    if not isinstance(supersedes, dict) or set(supersedes) != {
        "protocol_id",
        "registry_sha256",
        "reference_git_commit",
        "changed_scope",
    }:
        raise GateError("v1.2 predecessor registry authority is incomplete")
    if (
        supersedes.get("protocol_id") != PREDECESSOR_PROTOCOL_ID
        or supersedes.get("changed_scope") != "S5_slot_only"
    ):
        raise GateError("v1.2 declares an unsupported predecessor transition")
    predecessor_registry_sha256 = str(supersedes["registry_sha256"])
    if len(predecessor_registry_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in predecessor_registry_sha256
    ):
        raise GateError("predecessor registry SHA-256 is invalid")
    reference_git_commit = _require_full_git_oid(
        supersedes["reference_git_commit"], "predecessor registry Git commit"
    )
    _require_git_ancestor(reference_git_commit, _git_commit(), "predecessor registry Git commit")

    predecessor_blob_oid, predecessor_bytes = _git_blob(reference_git_commit, registry_path)
    if hashlib.sha256(predecessor_bytes).hexdigest() != predecessor_registry_sha256:
        raise GateError("predecessor registry Git blob SHA-256 mismatch")
    predecessor = _load_yaml_bytes(predecessor_bytes, "predecessor registry Git blob")
    predecessor_protocol = predecessor.get("protocol", {})
    if (
        predecessor.get("schema_version") != 1
        or not isinstance(predecessor_protocol, dict)
        or predecessor_protocol.get("id") != PREDECESSOR_PROTOCOL_ID
        or predecessor_protocol.get("target_config") != protocol["target_config"]
    ):
        raise GateError("predecessor registry Git blob has the wrong authority")
    _, predecessor_target_bytes = _git_blob(reference_git_commit, target_path)
    if hashlib.sha256(predecessor_target_bytes).hexdigest() != target_sha256:
        raise GateError("v1.1 target authority changed in the v1.2 overlay")
    if _registry_except_protocol_and_s5(predecessor) != _registry_except_protocol_and_s5(registry):
        raise GateError("v1.2 changed registry scope outside protocol metadata and S5")
    if float(predecessor_protocol["hard_cap_physical_gpu_hours"]) != float(
        protocol["hard_cap_physical_gpu_hours"]
    ):
        raise GateError("v1.2 changed the predecessor GPU-hour cap")
    if int(predecessor_protocol["maximum_substantive_screens"]) != int(
        protocol["maximum_substantive_screens"]
    ):
        raise GateError("v1.2 changed the predecessor screen cap")

    grandfathered = _verify_grandfathered_launch_records(
        registry_path,
        protocol,
        predecessor_protocol_id=PREDECESSOR_PROTOCOL_ID,
        predecessor_registry_sha256=predecessor_registry_sha256,
        reference_git_commit=reference_git_commit,
        target_path=target_path,
        target_sha256=target_sha256,
        deep=deep,
    )
    return {
        "source_protocol_id": PREDECESSOR_PROTOCOL_ID,
        "source_registry_sha256": predecessor_registry_sha256,
        "source_registry_git_commit": reference_git_commit,
        "source_registry_git_blob_oid": predecessor_blob_oid,
        "source_registry": predecessor,
        "destination_protocol_id": SUCCESSOR_PROTOCOL_ID,
        "destination_registry_sha256": _sha256(registry_path),
        "target_path": _relative(target_path),
        "target_protocol_id": PREDECESSOR_PROTOCOL_ID,
        "target_sha256": target_sha256,
        "changed_scope": "S5_slot_only",
        "grandfathered_launch_records": grandfathered,
    }


def _verify_s5_activation(registry: dict[str, Any]) -> None:
    """Bind launchable S5 to the frozen development-only activation decision."""

    s5 = registry["candidates"]["S5"]
    if not s5.get("launchable"):
        return
    if not math.isclose(
        float(s5.get("maximum_estimated_physical_gpu_hours", math.nan)),
        S5_MAXIMUM_ESTIMATED_PHYSICAL_GPU_HOURS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise GateError("S5 maximum estimated GPU-hours changed from the frozen 29.1-hour gate")
    evidence = s5.get("activation_evidence")
    required_evidence_keys = {
        "status",
        "path",
        "sha256",
        "gate_time_registry_sha256",
        "outcome",
        "selected_candidate",
        "selected_variant",
        "checkpoint_step",
        "selected_checkpoint",
    }
    if not isinstance(evidence, dict) or set(evidence) != required_evidence_keys:
        raise GateError("launchable S5 has incomplete activation evidence")
    evidence_path = _root_path(evidence["path"])
    _verify_file(evidence_path, str(evidence["sha256"]), "S5 activation evidence")
    payload = _load_json(evidence_path)
    decision = payload.get("decision", {})
    selected_checkpoint = payload.get("inputs", {}).get("resume_checkpoints", {}).get("S3")
    if (
        evidence["status"] != "passed"
        or payload.get("status") != "completed"
        or payload.get("protocol_id") != registry["protocol"]["id"]
        or payload.get("gate_id") != "S5"
        or decision.get("activate_s5") is not True
        or decision.get("outcome") != evidence["outcome"]
        or decision.get("selected_candidate") != evidence["selected_candidate"]
        or decision.get("selected_variant") != evidence["selected_variant"]
        or int(payload.get("frozen_gate", {}).get("checkpoint_step", -1))
        != int(evidence["checkpoint_step"])
        or selected_checkpoint != evidence["selected_checkpoint"]
        or not math.isclose(
            float(
                payload.get("frozen_gate", {}).get(
                    "s5_maximum_estimated_physical_gpu_hours", math.nan
                )
            ),
            S5_MAXIMUM_ESTIMATED_PHYSICAL_GPU_HOURS,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise GateError("S5 activation declaration differs from its frozen gate evidence")
    gate_registry = payload.get("inputs", {}).get("frozen_authorities", {}).get("registry", {})
    if gate_registry.get("sha256") != evidence["gate_time_registry_sha256"]:
        raise GateError("S5 gate-time registry authority changed")

    base_slot = str(evidence["selected_candidate"])
    if base_slot != "S3" or s5["base_selection"]["checkpoint_step"] != int(
        evidence["checkpoint_step"]
    ):
        raise GateError("S5 does not preserve the frozen selected base")
    base = registry["candidates"][base_slot]
    if s5.get("initialization") != base.get("initialization"):
        raise GateError("S5 initialization differs from the selected base")
    if int(s5.get("parameter_count", -1)) != int(base.get("parameter_count", -2)):
        raise GateError("S5 parameter count differs from the selected base")

    s5_config = _load_yaml(_root_path(s5["config"]))
    base_config = _load_yaml(_root_path(base["config"]))
    if s5_config.get("student") != base_config.get("student"):
        raise GateError("S5 architecture differs from the selected base")
    if s5_config.get("inference") != base_config.get("inference"):
        raise GateError("S5 deployment path differs from the selected base")
    expected_initialization = {
        "kind": "seeded_random",
        "state_sha256": S5_INITIALIZATION_STATE_SHA256,
    }
    if (
        s5.get("initialization") != expected_initialization
        or s5_config.get("initialization") != expected_initialization
    ):
        raise GateError("S5 is not bound to the frozen seeded-random initialization")
    for key in (
        "seed",
        "model_revision",
        "dataset_revision",
        "hardware_snapshot",
        "model",
        "cache",
    ):
        if s5_config.get(key) != base_config.get(key):
            raise GateError(f"S5 changed selected-base config field {key}")
    allowed_training_changes = {
        "hypothesis",
        "loss_weights",
        "loss_reduction",
        "domain_weight_denominator",
        "zero_target_window_policy",
        "domain_weight_source",
        "domain_weight_outer_training_counts",
        "domain_weights",
    }
    shared_training = {
        key: value
        for key, value in s5_config["training"].items()
        if key not in allowed_training_changes
    }
    base_training = {
        key: value
        for key, value in base_config["training"].items()
        if key not in {"hypothesis", "loss_weights"}
    }
    if shared_training != base_training:
        raise GateError("S5 changed selected-base training semantics outside its reducer")
    objective = {
        "ground_truth": 1.0,
        "multivariate_kd": 0.0,
        "univariate_kd": 0.0,
        "cvrd": 0.0,
    }
    training = s5_config["training"]
    if (
        s5["variant"] != "compact_gt_domain_balanced"
        or training.get("loss_weights") != {s5["variant"]: objective}
        or training.get("loss_reduction") != "per_window_domain_balanced"
        or training.get("domain_weight_denominator") != "unweighted_valid_windows"
        or training.get("zero_target_window_policy")
        != "sequence_only_excluded_from_loss_and_denominator"
    ):
        raise GateError("S5 objective or reducer differs from the activated recipe")
    weight_source = s5["domain_weights_source"]
    if (
        training.get("domain_weights") != weight_source["weights"]
        or training.get("domain_weight_outer_training_counts")
        != weight_source["counts_include_zero_target_windows"]
        or training.get("domain_weight_source", {}).get("outer_training_identity_sha256")
        != weight_source["outer_training_identity_sha256"]
    ):
        raise GateError("S5 config differs from the frozen domain-weight authority")

    autotune = s5.get("throughput_autotune")
    required_autotune_keys = {
        "implementation",
        "implementation_sha256",
        "config",
        "config_sha256",
        "json_pointer",
        "semantics",
    }
    if not isinstance(autotune, dict) or set(autotune) != required_autotune_keys:
        raise GateError("launchable S5 lacks its exact-code throughput probe")
    _verify_file(
        _root_path(autotune["implementation"]),
        str(autotune["implementation_sha256"]),
        "S5 throughput implementation",
    )
    autotune_path = _root_path(autotune["config"])
    _verify_file(autotune_path, str(autotune["config_sha256"]), "S5 throughput config")
    probe_config = _load_yaml(autotune_path)
    if (
        probe_config.get("candidate_config") != s5["config"]
        or probe_config.get("probe", {}).get("variant") != s5["variant"]
        or probe_config.get("probe", {}).get("logical_batch_size_windows") != 256
        or probe_config.get("probe", {}).get("initialization_state_sha256")
        != S5_INITIALIZATION_STATE_SHA256
        or int(probe_config.get("probe", {}).get("warmup_optimizer_steps", -1)) != 4
        or int(probe_config.get("probe", {}).get("measured_optimizer_steps", -1)) != 32
        or int(probe_config.get("probe", {}).get("expected_replayed_windows", -1)) != 9216
        or int(probe_config.get("probe", {}).get("expected_measured_windows", -1)) != 8192
        or probe_config.get("output_template")
        != "results/reproduction/systems/compact-student-s5-domain-balanced-autotune-blackwell-"
        "attempt{attempt:02d}.json"
        or not math.isclose(
            float(probe_config.get("probe", {}).get("maximum_physical_gpu_hours", math.nan)),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise GateError("S5 throughput probe is not bound to its launch recipe")
    authority = probe_config.get("authority", {})
    if (
        authority.get("training_implementation") != registry["trainer"]["path"]
        or _root_path(authority.get("manager_implementation", "")) != Path(__file__).resolve()
        or authority.get("loss_implementation") != "src/timesfm_lab/distill/losses.py"
    ):
        raise GateError("S5 throughput probe names the wrong launch implementations")
    for key in ("training_implementation", "loss_implementation", "manager_implementation"):
        _verify_file(
            _root_path(authority[key]),
            str(authority[f"{key}_sha256"]),
            f"S5 throughput {key}",
        )
    required_additional = {
        "configs/performance_recovery/candidates.yaml",
        "configs/performance_recovery/screen_selection.yaml",
    }
    if set(authority.get("additional_relevant_files", [])) != required_additional:
        raise GateError("S5 throughput source map omits a registry authority")
    if "--expected-physical-gpu-uuid" not in _root_path(registry["trainer"]["path"]).read_text():
        raise GateError("S5 trainer lacks the physical-GPU identity gate")


def _verify_registry(registry_path: Path, *, deep: bool) -> dict[str, Any]:
    registry = _load_yaml(registry_path)
    protocol = registry["protocol"]
    targets = _load_yaml(_root_path(protocol["target_config"]))
    if registry["schema_version"] != 1:
        raise GateError("unsupported candidate-registry schema")
    _verify_protocol_authority(registry_path, registry, targets, deep=deep)
    target_cap = float(targets["compute_budget"]["hard_cap_gpu_hours"])
    if target_cap != float(protocol["hard_cap_physical_gpu_hours"]):
        raise GateError("registry changed the frozen GPU-hour cap")
    target_slots = tuple(item["id"] for item in targets["selection"]["candidate_slots"])
    if tuple(registry["candidates"]) != SLOTS or target_slots != SLOTS:
        raise GateError("candidate registry must contain exactly the frozen S1-S6 slots")
    if len(registry["candidates"]) > int(protocol["maximum_substantive_screens"]):
        raise GateError("candidate registry exceeds the six-screen cap")

    corpus = registry["corpus"]
    _verify_file(_root_path(corpus["plan"]), corpus["plan_sha256"], "corpus plan")
    _verify_file(_root_path(corpus["cache_audit"]), corpus["cache_audit_sha256"], "cache audit")
    audit = _load_json(_root_path(corpus["cache_audit"]))
    totals = audit.get("summary", audit.get("totals", {}))
    if (
        audit.get("status") != "succeeded"
        or totals.get("unique_windows") != corpus["unique_windows"]
    ):
        raise GateError("production cache audit is not a successful exact-uniqueness audit")
    if totals.get("duplicate_windows") != 0 or totals.get("datasets") != corpus["datasets"]:
        raise GateError("production cache audit does not match the declared corpus")
    if (
        not _root_path(corpus["data_root"]).is_dir()
        or not _root_path(corpus["cache_root"]).is_dir()
    ):
        raise GateError("production data/cache roots are unavailable")

    selection = registry["selection_split"]
    _verify_file(
        _root_path(selection["manifest"]), selection["manifest_sha256"], "selection manifest"
    )
    selection_payload = _load_json(_root_path(selection["manifest"]))
    if (
        selection_payload.get("status")
        != selection["confirmation_status_required_at_screen_launch"]
    ):
        raise GateError("confirmation manifest is no longer frozen and uninspected")
    if selection_payload.get("target_accessed") or selection_payload.get("teacher_output_accessed"):
        raise GateError("selection manifest reports prohibited target/teacher-output access")

    trainer = _root_path(registry["trainer"]["path"])
    trainer_text = trainer.read_text()
    required_arguments = registry["trainer"]["required_selection_arguments"]
    for key in ("manifest", "partition"):
        if required_arguments[key] not in trainer_text:
            raise GateError(
                f"trainer has not implemented required selection argument {required_arguments[key]}"
            )

    for slot, candidate in registry["candidates"].items():
        if not candidate.get("launchable"):
            continue
        config = _load_yaml(_root_path(candidate["config"]))
        training = config["training"]
        if candidate["variant"] not in training["loss_weights"]:
            raise GateError(f"{slot}: variant is absent from its declared config")
        if int(config["seed"]) != int(registry["screening"]["seed"]):
            raise GateError(f"{slot}: config seed differs from the frozen screening seed")
        if int(training["max_steps"]) != int(registry["screening"]["maximum_steps"]):
            raise GateError(f"{slot}: step ceiling differs from the registry")
        if int(training.get("logical_batch_size_windows", 0)) != 256:
            raise GateError(f"{slot}: fixed 256-window logical batch is required")
        if config["dataset_revision"] != corpus["dataset_revision"]:
            raise GateError(f"{slot}: dataset revision mismatch")
        if deep:
            _verify_model_initialization(candidate, config)
        measurement = candidate.get("throughput_measurement")
        if measurement:
            _validate_measurement(measurement, candidate, registry)
    _verify_s5_activation(registry)
    if deep:
        _verify_selection_reconstruction(registry)
        _require_relevant_tree_clean(registry_path, registry)
    return registry


def _migration_transition_payload(
    registry_path: Path, registry: dict[str, Any], transition: dict[str, Any]
) -> dict[str, Any]:
    cap = float(registry["protocol"]["hard_cap_physical_gpu_hours"])
    target = {
        "path": transition["target_path"],
        "protocol_id": transition["target_protocol_id"],
        "sha256": transition["target_sha256"],
    }
    return {
        "source_authority": {
            "protocol_id": transition["source_protocol_id"],
            "registry": {
                "path": _relative(registry_path),
                "sha256": transition["source_registry_sha256"],
                "reference_git_commit": transition["source_registry_git_commit"],
                "git_blob_oid": transition["source_registry_git_blob_oid"],
            },
            "target": target,
            "hard_cap_physical_gpu_hours": cap,
        },
        "destination_authority": {
            "protocol_id": transition["destination_protocol_id"],
            "registry": {
                "path": _relative(registry_path),
                "sha256": transition["destination_registry_sha256"],
            },
            "target": target,
            "hard_cap_physical_gpu_hours": cap,
        },
        "changed_scope": transition["changed_scope"],
        "target_authority_unchanged": True,
    }


def _preserved_ledger_state(ledger: dict[str, Any]) -> dict[str, Any]:
    mutable_authority_fields = {
        "protocol_id",
        "registry_sha256",
        "protocol_migrations",
        "updated_at_utc",
    }
    return {key: value for key, value in ledger.items() if key not in mutable_authority_fields}


def _verify_preserved_candidate_extension(
    slot: str, source_state: dict[str, Any], current_state: dict[str, Any]
) -> None:
    source_history = source_state.get("throughput_measurement_history", [])
    current_history = current_state.get("throughput_measurement_history", [])
    if (
        not isinstance(source_history, list)
        or not isinstance(current_history, list)
        or current_history[: len(source_history)] != source_history
    ):
        raise GateError(f"{slot} predecessor throughput history changed after migration")
    source_measurement = source_state.get("throughput_measurement")
    current_measurement = current_state.get("throughput_measurement")
    measurement_preserved = source_measurement == current_measurement or any(
        isinstance(entry, dict) and entry.get("measurement") == source_measurement
        for entry in current_history[len(source_history) :]
    )
    if source_measurement is not None and not measurement_preserved:
        raise GateError(f"{slot} predecessor throughput measurement was not preserved")
    source_attempts = source_state.get("attempts", [])
    current_attempts = current_state.get("attempts", [])
    if not isinstance(source_attempts, list) or not isinstance(current_attempts, list):
        raise GateError(f"{slot} has invalid migration attempt history")
    if len(current_attempts) < len(source_attempts):
        raise GateError(f"{slot} predecessor attempt history was truncated")
    for index, source_attempt in enumerate(source_attempts):
        current_attempt = current_attempts[index]
        if current_attempt == source_attempt:
            continue
        allowed = dict(source_attempt)
        if index == len(source_attempts) - 1 and "fresh_retry_archive" in current_attempt:
            allowed["fresh_retry_archive"] = current_attempt["fresh_retry_archive"]
        if current_attempt != allowed:
            raise GateError(f"{slot} predecessor attempt {index + 1} changed after migration")
    for key, source_value in source_state.items():
        if key in {
            "attempts",
            "status",
            "throughput_measurement",
            "throughput_measurement_history",
        }:
            continue
        if key not in current_state or current_state[key] != source_value:
            raise GateError(f"{slot} predecessor candidate data changed after migration: {key}")


def _verify_migration_worker_evidence(evidence: dict[str, Any], transition: dict[str, Any]) -> None:
    records = evidence.get("grandfathered_worker_records")
    if not isinstance(records, list):
        raise GateError("protocol migration evidence lacks grandfathered worker records")
    declarations = {
        (entry["candidate_id"], int(entry["attempt"])): entry
        for entry in transition["grandfathered_launch_records"]
    }
    source_candidates = evidence["source_ledger"]["candidates"]
    observed: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "candidate_id",
            "attempt",
            "path",
            "sha256",
        }:
            raise GateError("protocol migration evidence has an invalid worker record")
        key = (str(record["candidate_id"]), int(record["attempt"]))
        if key not in declarations or key in observed:
            raise GateError("protocol migration evidence has an unapproved worker record")
        observed.add(key)
        source_attempts = source_candidates[key[0]]["attempts"]
        if key[1] > len(source_attempts):
            raise GateError("protocol migration worker is absent from its source ledger")
        source_attempt = source_attempts[key[1] - 1]
        if (
            source_attempt.get("status") not in TERMINAL_ATTEMPT_STATUSES
            or source_attempt.get("worker_record") != record["path"]
        ):
            raise GateError("protocol migration worker differs from its source ledger")
        worker, worker_sha256 = _load_json_snapshot(_root_path(record["path"]))
        if worker_sha256 != record["sha256"]:
            raise GateError("grandfathered migration worker SHA-256 mismatch")
        _verify_reconciled_worker(
            worker,
            source_attempt,
            protocol_id=transition["source_protocol_id"],
            slot=key[0],
            index=key[1],
        )
    if observed != set(declarations):
        raise GateError("protocol migration evidence omits a grandfathered worker")


def _verify_current_migration_reference(
    ledger_path: Path,
    ledger: dict[str, Any],
    registry_path: Path,
    registry: dict[str, Any],
    transition: dict[str, Any],
) -> None:
    migrations = ledger.get("protocol_migrations")
    if not isinstance(migrations, list) or not migrations:
        raise GateError("v1.2 ledger lacks the required append-only protocol migration")
    reference = migrations[-1]
    if not isinstance(reference, dict):
        raise GateError("invalid protocol migration reference")
    evidence_path = _root_path(reference.get("evidence_path", ""))
    evidence_sha256 = str(reference.get("evidence_sha256", ""))
    _verify_file(evidence_path, evidence_sha256, "protocol migration evidence")
    evidence = _load_json(evidence_path)
    if (
        evidence.get("schema_version") != 1
        or evidence.get("evidence_type") != MIGRATION_EVIDENCE_TYPE
        or evidence.get("ledger_path") != _relative(ledger_path)
        or evidence.get("transition")
        != _migration_transition_payload(registry_path, registry, transition)
    ):
        raise GateError("protocol migration evidence has invalid authority")
    created_at = str(evidence.get("created_at_utc", ""))
    _parse_utc(created_at)
    source_ledger_sha256 = str(evidence.get("source_ledger_sha256", ""))
    if len(source_ledger_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_ledger_sha256
    ):
        raise GateError("protocol migration evidence has an invalid source ledger hash")
    source_ledger = evidence.get("source_ledger")
    if (
        not isinstance(source_ledger, dict)
        or evidence.get("source_ledger_canonical_sha256") != _canonical_sha256(source_ledger)
        or source_ledger.get("protocol_id") != transition["source_protocol_id"]
        or source_ledger.get("registry_sha256") != transition["source_registry_sha256"]
    ):
        raise GateError("protocol migration evidence has invalid source ledger state")
    prior_migrations = source_ledger.get("protocol_migrations", [])
    if not isinstance(prior_migrations, list) or migrations[:-1] != prior_migrations:
        raise GateError("protocol migration history is not append-only")
    source_candidates = source_ledger.get("candidates")
    current_candidates = ledger.get("candidates")
    if not isinstance(source_candidates, dict) or not isinstance(current_candidates, dict):
        raise GateError("protocol migration evidence has invalid candidate state")
    for slot, source_state in source_candidates.items():
        current_state = current_candidates.get(slot, {})
        if not isinstance(source_state, dict) or not isinstance(current_state, dict):
            raise GateError(f"{slot} has invalid migration candidate state")
        _verify_preserved_candidate_extension(slot, source_state, current_state)
    for key, source_value in _preserved_ledger_state(source_ledger).items():
        if (
            key not in {"accounting", "candidates", "external_jobs"}
            and ledger.get(key) != source_value
        ):
            raise GateError(f"preserved ledger field changed after migration: {key}")
    source_jobs = source_ledger.get("external_jobs", [])
    if ledger.get("external_jobs", [])[: len(source_jobs)] != source_jobs:
        raise GateError("predecessor external-job history changed after migration")
    expected_reference = {
        "schema_version": 1,
        "from_protocol_id": transition["source_protocol_id"],
        "from_registry_sha256": transition["source_registry_sha256"],
        "to_protocol_id": transition["destination_protocol_id"],
        "to_registry_sha256": transition["destination_registry_sha256"],
        "source_ledger_sha256": source_ledger_sha256,
        "evidence_path": _relative(evidence_path),
        "evidence_sha256": evidence_sha256,
        "migrated_at_utc": created_at,
    }
    if reference != expected_reference:
        raise GateError("ledger protocol migration reference disagrees with its evidence")
    _verify_migration_worker_evidence(evidence, transition)


def _load_ledger(
    path: Path,
    registry_path: Path,
    registry: dict[str, Any],
    *,
    allow_predecessor: bool = False,
    verified_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = _load_json(path)
    if ledger.get("schema_version") != 1:
        raise GateError("unsupported GPU-ledger schema")
    if float(ledger.get("hard_cap_physical_gpu_hours", -1)) != float(
        registry["protocol"]["hard_cap_physical_gpu_hours"]
    ):
        raise GateError("ledger GPU-hour cap mismatch")
    if tuple(ledger.get("candidates", {})) != SLOTS:
        raise GateError("ledger does not contain exactly S1-S6")
    current_authority = ledger.get("protocol_id") == registry["protocol"]["id"] and ledger.get(
        "registry_sha256"
    ) == _sha256(registry_path)
    transition = verified_transition
    if transition is None:
        targets = _load_yaml(_root_path(registry["protocol"]["target_config"]))
        transition = _verify_protocol_authority(registry_path, registry, targets, deep=False)
    predecessor_authority = bool(
        transition
        and ledger.get("protocol_id") == transition["source_protocol_id"]
        and ledger.get("registry_sha256") == transition["source_registry_sha256"]
    )
    if current_authority:
        if transition is not None:
            _verify_current_migration_reference(path, ledger, registry_path, registry, transition)
        _verify_current_terminal_attempt_artifacts(ledger, registry)
    elif not (allow_predecessor and predecessor_authority):
        raise GateError("ledger is not pinned to the required protocol/registry authority")
    return ledger


def _measurement(
    slot: str, candidate: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any] | None:
    return candidate.get("throughput_measurement") or ledger["candidates"][slot].get(
        "throughput_measurement"
    )


def _derived_measurement_gpu_hours(
    windows_per_second: float, registry: dict[str, Any]
) -> tuple[float, float, float, int]:
    safety = float(registry["screening"]["measurement_safety_factor"])
    overhead = float(registry["screening"]["estimated_fixed_overhead_seconds"])
    examples = int(registry["screening"]["maximum_examples_processed"])
    estimate = (examples / windows_per_second + overhead) * safety / 3600.0
    return estimate, overhead, safety, examples


def _validate_s5_measurement_budget(
    candidate: dict[str, Any], measurement: dict[str, Any], registry: dict[str, Any]
) -> float:
    declared_maximum = float(candidate.get("maximum_estimated_physical_gpu_hours", math.nan))
    if not math.isclose(
        declared_maximum,
        S5_MAXIMUM_ESTIMATED_PHYSICAL_GPU_HOURS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise GateError("S5 measurement is not bound to its frozen 29.1-hour maximum")
    declared = float(measurement["windows_per_second"])
    derived, overhead, safety, examples = _derived_measurement_gpu_hours(declared, registry)
    if (
        int(measurement.get("maximum_examples_processed", -1)) != examples
        or not math.isclose(
            float(measurement.get("estimated_fixed_overhead_seconds", math.nan)),
            overhead,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(measurement.get("safety_factor", math.nan)),
            safety,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(measurement.get("estimated_gpu_hours", math.nan)),
            derived,
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
    ):
        raise GateError("S5 GPU-hour estimate is not immutably derived from frozen inputs")
    if derived > declared_maximum:
        raise GateError(
            f"S5 derived estimate {derived:.6f} exceeds its frozen {declared_maximum:.1f}-hour gate"
        )
    return derived


def _validate_measurement(
    measurement: dict[str, Any] | None,
    candidate: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> float:
    if not measurement:
        raise GateError("a measured throughput artifact is required before launch")
    source = _root_path(measurement["source"])
    payload, source_sha256 = _load_json_snapshot(source)
    if source_sha256 != measurement["source_sha256"]:
        raise GateError("throughput evidence SHA-256 mismatch")
    observed = float(_json_pointer(payload, measurement["json_pointer"]))
    declared = float(measurement["windows_per_second"])
    if (
        observed <= 0
        or not math.isfinite(observed)
        or not math.isclose(observed, declared, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise GateError("invalid or mismatched measured windows/second")
    estimate = float(measurement["estimated_gpu_hours"])
    if estimate <= 0 or not math.isfinite(estimate):
        raise GateError("invalid estimated GPU-hours")
    if (
        candidate is not None
        and registry is not None
        and candidate.get("maximum_estimated_physical_gpu_hours") is not None
    ):
        estimate = _validate_s5_measurement_budget(candidate, measurement, registry)
    return estimate


def _attempt_hours(attempt: dict[str, Any], now: dt.datetime) -> float:
    if attempt.get("actual_gpu_hours") is not None:
        return float(attempt["actual_gpu_hours"])
    if attempt.get("status") == "running":
        return max(0.0, (now - _parse_utc(attempt["started_at"])).total_seconds() / 3600.0)
    return 0.0


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise GateError(f"{label} must be finite and positive")
    return result


def _load_bound_autotune_child(
    job: dict[str, Any], launch_payload: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    child_path_value = str(job.get("child_identity_record", ""))
    if (
        not child_path_value
        or child_path_value != launch_payload.get("child_identity_record")
    ):
        raise GateError("S5 autotune child-identity path differs from its launch")
    child_path = _root_path(child_path_value)
    binding = job.get("child_record")
    if binding is None:
        if child_path.exists():
            raise GateError("S5 autotune child identity exists without a ledger hash binding")
        return None
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise GateError("S5 autotune child identity has an invalid ledger binding")
    if binding["path"] != child_path_value:
        raise GateError("S5 autotune child identity binding changed path")
    child_record, child_sha256 = _load_json_snapshot(child_path)
    if child_sha256 != binding["sha256"]:
        raise GateError("S5 autotune child-identity SHA-256 mismatch")
    _validate_autotune_child_identity(
        child_record,
        launch=launch_payload,
        launch_binding=job["launch_record"],
    )
    if (
        int(child_record["wrapper_pid"]) != int(job.get("pid", -1))
        or str(child_record["wrapper_process_start_ticks"])
        != str(job.get("process_start_ticks", ""))
    ):
        raise GateError("S5 autotune child identity differs from its wrapper binding")
    return child_record, child_sha256


def _bind_or_load_autotune_child_for_reconcile(
    job: dict[str, Any], launch_payload: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    """Recover a child binding written just before a wrapper interruption."""

    child_path = _root_path(job["child_identity_record"])
    if child_path.exists() and job.get("child_record") is None:
        child_record, child_sha256 = _load_json_snapshot(child_path)
        _validate_autotune_child_identity(
            child_record,
            launch=launch_payload,
            launch_binding=job["launch_record"],
        )
        if (
            int(child_record["wrapper_pid"]) != int(job.get("pid", -1))
            or str(child_record["wrapper_process_start_ticks"])
            != str(job.get("process_start_ticks", ""))
        ):
            raise GateError("orphaned autotune child differs from its wrapper")
        job["child_record"] = {
            "path": _relative(child_path),
            "sha256": child_sha256,
        }
    return _load_bound_autotune_child(job, launch_payload)


def _verify_autotune_worker_deadline(
    worker: dict[str, Any], launch_payload: dict[str, Any]
) -> dt.datetime:
    """Require process stop and terminal observation inside the hard reservation."""

    deadline = _parse_utc(str(launch_payload.get("deadline_at", "")))
    ended_at = _parse_utc(str(worker.get("ended_at", "")))
    if ended_at > deadline:
        raise GateError("S5 autotune terminal record exceeds its absolute deadline")
    child_binding = worker.get("child_record")
    child_stopped_value = worker.get("child_stopped_at")
    if child_binding is None:
        if child_stopped_value is not None:
            raise GateError("S5 autotune records an unbound child stop")
        return ended_at
    if child_stopped_value is None:
        raise GateError("S5 autotune worker did not timestamp its bound child stop")
    child_stopped_at = _parse_utc(str(child_stopped_value))
    if child_stopped_at > deadline:
        raise GateError("S5 autotune child survived its absolute deadline")
    return ended_at


def _validate_s5_autotune_job(job: dict[str, Any]) -> None:
    """Validate preregistration and derive terminal cost from immutable worker bytes."""

    if job.get("category") != S5_AUTOTUNE_CATEGORY:
        return
    if int(job.get("physical_gpu_count", -1)) != 1:
        raise GateError("S5 autotune must reserve exactly one physical GPU")
    _finite_positive(job.get("estimated_gpu_hours"), "S5 autotune reserved GPU-hours")
    launch = job.get("launch_record")
    if not isinstance(launch, dict) or set(launch) != {"path", "sha256"}:
        raise GateError("S5 autotune lacks an immutable launch binding")
    launch_payload, launch_sha256 = _load_json_snapshot(_root_path(launch["path"]))
    if launch_sha256 != launch["sha256"]:
        raise GateError("S5 autotune immutable launch hash changed")
    if (
        launch_payload.get("schema_version") != 1
        or launch_payload.get("kind") != S5_AUTOTUNE_CATEGORY
        or launch_payload.get("job_id") != job.get("job_id")
        or int(launch_payload.get("attempt", -1)) != int(job.get("attempt", -2))
        or launch_payload.get("git_commit") != job.get("git_commit")
        or launch_payload.get("output") != job.get("artifact")
        or launch_payload.get("worker_record") != job.get("worker_record")
        or launch_payload.get("child_identity_record")
        != job.get("child_identity_record")
        or int(launch_payload.get("physical_gpu", -1)) != int(job.get("physical_gpu", -2))
        or launch_payload.get("started_at") != job.get("started_at")
        or launch_payload.get("deadline_at") != job.get("deadline_at")
        or launch_payload.get("execution_deadline_at")
        != job.get("execution_deadline_at")
        or launch_payload.get("cleanup_budget_seconds")
        != job.get("cleanup_budget_seconds")
        or not math.isclose(
            float(launch_payload.get("reserved_gpu_hours", math.nan)),
            float(job.get("estimated_gpu_hours", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise GateError("S5 autotune ledger entry differs from its immutable launch")
    started_at = _parse_utc(str(job.get("started_at", "")))
    deadline_at = _parse_utc(str(job.get("deadline_at", "")))
    reserved_seconds = float(job["estimated_gpu_hours"]) * 3600.0
    cleanup_budget_seconds = _finite_positive(
        job.get("cleanup_budget_seconds"), "S5 autotune cleanup budget seconds"
    )
    execution_deadline_at = _parse_utc(str(job.get("execution_deadline_at", "")))
    if (
        deadline_at <= started_at
        or not math.isclose(
            (deadline_at - started_at).total_seconds(),
            reserved_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise GateError("S5 autotune absolute deadline differs from its reservation")
    if (
        not math.isclose(
            cleanup_budget_seconds,
            _s5_autotune_cleanup_budget_seconds(),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or execution_deadline_at
        != deadline_at - dt.timedelta(seconds=cleanup_budget_seconds)
        or execution_deadline_at <= started_at
    ):
        raise GateError("S5 autotune execution deadline does not reserve cleanup inside cap")
    status = str(job.get("status", ""))
    if status in {"launching", "running", "unreconciled"}:
        if job.get("actual_gpu_hours") is not None:
            raise GateError("nonterminal S5 autotune cannot claim actual GPU-hours")
        if status == "running" and (
            not isinstance(job.get("pid"), int)
            or isinstance(job.get("pid"), bool)
            or int(job["pid"]) <= 0
            or not str(job.get("process_start_ticks", ""))
        ):
            raise GateError("running S5 autotune lacks its exact worker identity")
        child = _load_bound_autotune_child(job, launch_payload)
        if status == "launching" and child is not None:
            raise GateError("launching S5 autotune cannot already bind a GPU child")
        return
    if status != "completed":
        raise GateError(f"invalid S5 autotune status {status!r}")
    terminal = job.get("terminal_record")
    if not isinstance(terminal, dict) or set(terminal) != {"path", "sha256"}:
        raise GateError("completed S5 autotune lacks its immutable worker record")
    worker, worker_sha256 = _load_json_snapshot(_root_path(terminal["path"]))
    if worker_sha256 != terminal["sha256"]:
        raise GateError("S5 autotune worker-record SHA-256 mismatch")
    elapsed = _finite_positive(worker.get("elapsed_seconds"), "S5 autotune elapsed seconds")
    child = _load_bound_autotune_child(job, launch_payload)
    actual = _finite_positive(job.get("actual_gpu_hours"), "S5 autotune actual GPU-hours")
    if (
        worker.get("job_id") != job.get("job_id")
        or int(worker.get("attempt", -1)) != int(job.get("attempt", -2))
        or worker.get("launch_record") != launch
        or worker.get("child_record") != job.get("child_record")
        or worker.get("started_at") != launch_payload.get("started_at")
        or worker.get("deadline_at") != launch_payload.get("deadline_at")
        or int(worker.get("physical_gpu", -1)) != int(job.get("physical_gpu", -2))
        or _normalize_gpu_uuid(str(worker.get("physical_gpu_uuid", "")))
        != _normalize_gpu_uuid(str(job.get("physical_gpu_uuid", "")))
        or not math.isclose(actual, elapsed / 3600.0, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            float(job.get("elapsed_seconds", math.nan)), elapsed, rel_tol=0.0, abs_tol=0.0
        )
        or job.get("ended_at") != worker.get("ended_at")
        or job.get("exit_code") != worker.get("exit_code")
    ):
        raise GateError("S5 autotune terminal GPU cost does not derive from its worker record")
    exit_code = worker.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise GateError("S5 autotune worker exit code is invalid")
    worker_ended_at = _verify_autotune_worker_deadline(worker, launch_payload)
    observed_interval = (worker_ended_at - started_at).total_seconds()
    if (
        observed_interval <= 0
        or not math.isclose(elapsed, observed_interval, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise GateError("S5 autotune elapsed time differs from its absolute timestamps")
    if exit_code == 0 and child is None:
        raise GateError("successful S5 autotune lacks its immutable child identity")
    if child is not None:
        _require_process_stopped(
            int(child[0]["child_pid"]),
            str(child[0]["child_process_start_ticks"]),
            "completed S5 autotune child",
        )
    expected_outcome = "succeeded" if exit_code == 0 else "failed"
    if job.get("outcome") != expected_outcome:
        raise GateError("S5 autotune outcome contradicts its worker record")
    worker_artifact = worker.get("artifact_at_exit")
    if not isinstance(worker_artifact, dict) or worker_artifact.get("path") != job.get("artifact"):
        raise GateError("S5 autotune worker did not bind its unique output")
    if job.get("artifact_sha256") != worker_artifact.get("sha256"):
        raise GateError("S5 autotune output hash differs from its worker record")
    _verify_file(
        _root_path(worker_artifact["path"]),
        str(worker_artifact["sha256"]),
        "S5 autotune immutable output",
    )


def _validate_external_jobs(ledger: dict[str, Any]) -> None:
    jobs = ledger.get("external_jobs", [])
    if not isinstance(jobs, list):
        raise GateError("external GPU jobs must be a list")
    ids: set[str] = set()
    attempts: list[int] = []
    outputs: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise GateError("external GPU job must be an object")
        job_id = str(job.get("job_id", ""))
        if not job_id or job_id in ids:
            raise GateError("external GPU job ids must be nonempty and unique")
        ids.add(job_id)
        _finite_positive(job.get("estimated_gpu_hours"), f"{job_id} estimated GPU-hours")
        actual = job.get("actual_gpu_hours")
        if actual is not None:
            _finite_positive(actual, f"{job_id} actual GPU-hours")
        status = str(job.get("status", ""))
        if status not in {"launching", "running", "unreconciled", "completed"}:
            raise GateError(f"{job_id} has invalid external-job status")
        if status == "completed" and actual is None:
            raise GateError(f"{job_id} completed without positive actual GPU-hours")
        if status != "completed" and actual is not None:
            raise GateError(f"{job_id} nonterminal external job claims actual GPU-hours")
        if job.get("category") == S5_AUTOTUNE_CATEGORY:
            attempt = int(job.get("attempt", -1))
            artifact = str(job.get("artifact", ""))
            if attempt <= 0 or not artifact or artifact in outputs:
                raise GateError("S5 autotune attempts/outputs must be unique and positive")
            _root_path(artifact)
            attempts.append(attempt)
            outputs.add(artifact)
            _validate_s5_autotune_job(job)
    if attempts and sorted(attempts) != list(range(1, len(attempts) + 1)):
        raise GateError("S5 autotune attempt history must be contiguous")


def _accounting(ledger: dict[str, Any], registry: dict[str, Any]) -> dict[str, float]:
    now = dt.datetime.now(dt.UTC)
    _validate_external_jobs(ledger)
    actual = 0.0
    committed = 0.0
    unknown = False
    for job in ledger.get("external_jobs", []):
        if job["status"] == "unreconciled":
            unknown = True
        if job.get("actual_gpu_hours") is not None:
            value = float(job["actual_gpu_hours"])
        elif job.get("status") in {"launching", "running"}:
            started_at = _parse_utc(str(job.get("started_at", "")))
            elapsed = (now - started_at).total_seconds() / 3600.0
            if elapsed < 0 or not math.isfinite(elapsed):
                raise GateError(f"{job['job_id']} has invalid live elapsed GPU time")
            value = max(float(job["estimated_gpu_hours"]), elapsed)
        else:
            value = float(job["estimated_gpu_hours"])
        committed += value
        if job.get("actual_gpu_hours") is not None:
            actual += float(job["actual_gpu_hours"])
    for slot, state in ledger["candidates"].items():
        attempts = state.get("attempts", [])
        finished_actual = sum(
            float(attempt["actual_gpu_hours"])
            for attempt in attempts
            if attempt.get("actual_gpu_hours") is not None
        )
        running_elapsed = sum(
            _attempt_hours(attempt, now)
            for attempt in attempts
            if attempt.get("status") == "running"
        )
        actual += finished_actual
        if state.get("status") == "unreconciled":
            unknown = True
        if state.get("status") == "running":
            candidate = registry["candidates"][slot]
            measurement = _measurement(slot, candidate, ledger)
            estimate = _validate_measurement(measurement, candidate, registry)
            if candidate.get("maximum_estimated_physical_gpu_hours") is not None:
                _verify_s5_autotune_accounting(ledger, measurement)
            committed += finished_actual + max(running_elapsed, estimate)
        else:
            committed += finished_actual
    cap = float(ledger["hard_cap_physical_gpu_hours"])
    if unknown:
        committed = cap
    return {
        "actual_gpu_hours": actual,
        "committed_gpu_hours": committed,
        "remaining_uncommitted_gpu_hours": max(0.0, cap - committed),
    }


def _write_ledger(path: Path, ledger: dict[str, Any], registry: dict[str, Any]) -> None:
    ledger["accounting"] = _accounting(ledger, registry)
    ledger["updated_at_utc"] = _utc_now()
    _atomic_json(path, ledger)


def _verify_reconciled_worker(
    worker: dict[str, Any],
    attempt: dict[str, Any],
    *,
    protocol_id: str,
    slot: str,
    index: int,
) -> None:
    if (
        worker.get("protocol_id") != protocol_id
        or worker.get("candidate_id") != slot
        or worker.get("ended_at") != attempt["ended_at"]
        or worker.get("elapsed_seconds") != attempt["elapsed_seconds"]
        or worker.get("exit_code") != attempt["exit_code"]
        or worker.get("artifacts_at_exit", {}) != attempt.get("artifacts_at_exit", {})
        or worker.get("trainer_source_authority_at_exit")
        != attempt.get("trainer_source_authority_at_exit")
    ):
        raise GateError(f"{slot} attempt {index} worker record was not reconciled exactly")
    if (
        not isinstance(worker["elapsed_seconds"], (int, float))
        or isinstance(worker["elapsed_seconds"], bool)
        or not isinstance(worker["exit_code"], int)
        or isinstance(worker["exit_code"], bool)
    ):
        raise GateError(f"{slot} attempt {index} has invalid worker completion values")
    expected_gpu_hours = float(worker["elapsed_seconds"]) / 3600.0
    actual_gpu_hours = float(attempt["actual_gpu_hours"])
    if (
        expected_gpu_hours < 0
        or not math.isfinite(expected_gpu_hours)
        or actual_gpu_hours < 0
        or not math.isfinite(actual_gpu_hours)
        or not math.isclose(
            actual_gpu_hours,
            expected_gpu_hours,
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
    ):
        raise GateError(f"{slot} attempt {index} GPU accounting is not reconciled")
    exit_code = int(worker["exit_code"])
    status = str(attempt.get("status", ""))
    if (exit_code == 0 and status not in {"succeeded", "invalid"}) or (
        exit_code != 0 and status != "failed"
    ):
        raise GateError(f"{slot} attempt {index} terminal status contradicts its worker")


def _load_bound_attempt_launch_snapshot(
    registry: dict[str, Any],
    slot: str,
    attempt: dict[str, Any],
    *,
    index: int,
) -> tuple[dict[str, Any], str, str]:
    """Load a launch only after its ledger/allowlist byte authority is proven.

    No field from the launch record, including ``protocol_id``, is trusted until
    its complete byte snapshot matches either the current attempt's ledger SHA
    or an explicit predecessor-record declaration in the active registry.
    """

    launch_path = _root_path(attempt["launch_record"])
    launch, launch_sha256 = _load_json_snapshot(launch_path)
    declarations = [
        declaration
        for declaration in registry["protocol"].get("grandfathered_launch_records", [])
        if declaration.get("candidate_id") == slot
        and int(declaration.get("attempt", -1)) == index
        and declaration.get("path") == _relative(launch_path)
    ]
    if len(declarations) > 1:
        raise GateError(f"{slot} attempt {index} has duplicate grandfather authority")
    declaration = declarations[0] if declarations else None
    attempt_binding = attempt.get("launch_record_sha256")
    expected_launch_sha256 = (
        str(attempt_binding)
        if attempt_binding is not None
        else str(declaration["sha256"])
        if declaration is not None
        else ""
    )
    if not expected_launch_sha256 or expected_launch_sha256 != launch_sha256:
        raise GateError(f"{slot} attempt {index} launch record changed")

    current_protocol = str(registry["protocol"]["id"])
    launch_protocol = str(launch.get("protocol_id", ""))
    if launch_protocol == current_protocol:
        if attempt_binding != launch_sha256:
            raise GateError(f"{slot} attempt {index} current launch lacks its ledger SHA")
        return launch, launch_sha256, current_protocol

    predecessor = registry["protocol"].get("supersedes", {}).get("protocol_id")
    if (
        launch_protocol != predecessor
        or declaration is None
        or declaration.get("sha256") != launch_sha256
    ):
        raise GateError(
            f"{slot} attempt {index} has unapproved non-current protocol authority"
        )
    return launch, launch_sha256, str(predecessor)


def _validate_reconciled_grandfathered_attempts(
    ledger: dict[str, Any], transition: dict[str, Any]
) -> list[dict[str, Any]]:
    allowlisted = {
        (entry["candidate_id"], int(entry["attempt"])): entry
        for entry in transition["grandfathered_launch_records"]
    }
    observed: set[tuple[str, int]] = set()
    worker_records: list[dict[str, Any]] = []
    for slot, state in ledger["candidates"].items():
        if state.get("status") in {"launching", "running", "unreconciled"}:
            raise GateError(f"{slot} has not reached a reconciled terminal state")
        attempts = state.get("attempts", [])
        if not isinstance(attempts, list):
            raise GateError(f"{slot} has an invalid attempt history")
        if not attempts and state.get("status") != "declared":
            raise GateError(f"{slot} has no attempts but is not declared")
        for index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict) or int(attempt.get("attempt", -1)) != index:
                raise GateError(f"{slot} attempt history is not contiguous")
            key = (slot, index)
            declaration = allowlisted.get(key)
            if declaration is None:
                raise GateError(f"unapproved predecessor launch in {slot} attempt {index}")
            if key in observed:
                raise GateError(f"duplicate predecessor launch in {slot} attempt {index}")
            observed.add(key)
            status = str(attempt.get("status", ""))
            if status not in TERMINAL_ATTEMPT_STATUSES:
                raise GateError(f"{slot} attempt {index} is not reconciled terminal")
            pid = attempt.get("pid")
            identity = attempt.get("process_start_ticks")
            if pid is None or identity is None:
                raise GateError(f"{slot} attempt {index} lacks its worker identity")
            _require_process_stopped(int(pid), str(identity), f"{slot} attempt {index}")
            if (
                attempt.get("actual_gpu_hours") is None
                or attempt.get("elapsed_seconds") is None
                or attempt.get("ended_at") is None
                or attempt.get("exit_code") is None
            ):
                raise GateError(f"{slot} attempt {index} lacks reconciled completion data")

            launch_path = _root_path(attempt.get("launch_record", ""))
            if _relative(launch_path) != declaration["path"]:
                raise GateError(f"{slot} attempt {index} launch path changed")
            _verify_file(
                launch_path,
                declaration["sha256"],
                f"{slot} attempt {index} grandfathered launch",
            )
            launch = _load_json(launch_path)
            worker_path = _root_path(attempt.get("worker_record", ""))
            worker, worker_sha256 = _load_json_snapshot(worker_path)
            if (
                launch.get("protocol_id") != transition["source_protocol_id"]
                or launch.get("candidate_id") != slot
                or launch.get("worker_record") != _relative(worker_path)
            ):
                raise GateError(f"{slot} attempt {index} launch record disagrees with the ledger")
            _verify_reconciled_worker(
                worker,
                attempt,
                protocol_id=transition["source_protocol_id"],
                slot=slot,
                index=index,
            )
            worker_records.append(
                {
                    "candidate_id": slot,
                    "attempt": index,
                    "path": _relative(worker_path),
                    "sha256": worker_sha256,
                }
            )
        if attempts and state.get("status") != attempts[-1].get("status"):
            raise GateError(f"{slot} state disagrees with its latest terminal attempt")
    if observed != set(allowlisted):
        missing = sorted(set(allowlisted) - observed)
        raise GateError(f"grandfathered attempts are missing from the ledger: {missing}")
    for job in ledger.get("external_jobs", []):
        if job.get("status") != "completed" or job.get("actual_gpu_hours") is None:
            raise GateError("all predecessor external GPU jobs must be reconciled completed")
        actual_gpu_hours = float(job["actual_gpu_hours"])
        if actual_gpu_hours <= 0 or not math.isfinite(actual_gpu_hours):
            raise GateError("predecessor external GPU accounting is invalid")
    return worker_records


def _migration_evidence_payload(
    *,
    created_at_utc: str,
    ledger_path: Path,
    source_ledger_sha256: str,
    ledger: dict[str, Any],
    registry_path: Path,
    registry: dict[str, Any],
    transition: dict[str, Any],
    worker_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_type": MIGRATION_EVIDENCE_TYPE,
        "created_at_utc": created_at_utc,
        "ledger_path": _relative(ledger_path),
        "source_ledger_sha256": source_ledger_sha256,
        "source_ledger": copy.deepcopy(ledger),
        "source_ledger_canonical_sha256": _canonical_sha256(ledger),
        "transition": _migration_transition_payload(registry_path, registry, transition),
        "grandfathered_worker_records": worker_records,
    }


def _command_migrate_protocol(args: argparse.Namespace) -> int:
    evidence_path = args.ledger.with_name(
        f"{args.ledger.stem}-migration-v1.1-to-v1.2.json"
    ).resolve()
    _root_path(evidence_path)
    if evidence_path in {args.ledger, args.registry}:
        raise GateError("migration evidence must use its own immutable path")

    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=True)
        targets = _load_yaml(_root_path(registry["protocol"]["target_config"]))
        transition = _verify_protocol_authority(args.registry, registry, targets, deep=False)
        if transition is None:
            raise GateError("the current registry does not declare a protocol overlay")
        loaded_ledger_sha256 = _sha256(args.ledger)
        ledger = _load_ledger(
            args.ledger,
            args.registry,
            registry,
            allow_predecessor=True,
            verified_transition=transition,
        )
        if _sha256(args.ledger) != loaded_ledger_sha256:
            raise GateError("ledger changed while its migration source was loaded")
        if (
            ledger.get("protocol_id") == transition["destination_protocol_id"]
            and ledger.get("registry_sha256") == transition["destination_registry_sha256"]
        ):
            reference = ledger["protocol_migrations"][-1]
            if _root_path(reference["evidence_path"]) != evidence_path:
                raise GateError("completed migration uses an unexpected evidence path")
            _fsync_directory(evidence_path.parent)
            _fsync_directory(args.ledger.parent)
            print(
                json.dumps(
                    {
                        "already_migrated": True,
                        "protocol_id": ledger["protocol_id"],
                        "registry_sha256": ledger["registry_sha256"],
                        "source_ledger_sha256": reference["source_ledger_sha256"],
                        "evidence_path": _relative(evidence_path),
                        "evidence_sha256": reference["evidence_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if (
            ledger.get("protocol_id") != transition["source_protocol_id"]
            or ledger.get("registry_sha256") != transition["source_registry_sha256"]
        ):
            raise GateError("migration requires the exact predecessor ledger authority")
        predecessor_cap = float(
            transition["source_registry"]["protocol"]["hard_cap_physical_gpu_hours"]
        )
        current_cap = float(registry["protocol"]["hard_cap_physical_gpu_hours"])
        ledger_cap = float(ledger["hard_cap_physical_gpu_hours"])
        if predecessor_cap != current_cap or ledger_cap != current_cap:
            raise GateError("protocol migration cannot change the GPU-hour cap")

        prior_value = ledger.get("protocol_migrations", [])
        if not isinstance(prior_value, list) or not all(
            isinstance(item, dict) for item in prior_value
        ):
            raise GateError("existing protocol migration history is invalid")
        prior_migrations: list[dict[str, Any]] = copy.deepcopy(prior_value)
        grandfathered = _validate_reconciled_grandfathered_attempts(ledger, transition)
        recomputed_source_accounting = _accounting(ledger, transition["source_registry"])
        if ledger.get("accounting") != recomputed_source_accounting:
            raise GateError("predecessor ledger accounting is stale or inconsistent")

        source_ledger_sha256 = loaded_ledger_sha256
        original = copy.deepcopy(ledger)
        evidence_preexisted = evidence_path.exists()
        if evidence_preexisted:
            existing_evidence = _load_json(evidence_path)
            created_at = str(existing_evidence.get("created_at_utc", ""))
            _parse_utc(created_at)
        else:
            created_at = _utc_now()
        evidence = _migration_evidence_payload(
            created_at_utc=created_at,
            ledger_path=args.ledger,
            source_ledger_sha256=source_ledger_sha256,
            ledger=ledger,
            registry_path=args.registry,
            registry=registry,
            transition=transition,
            worker_records=grandfathered,
        )
        evidence_sha256 = hashlib.sha256(_json_document(evidence).encode()).hexdigest()
        if evidence_preexisted:
            if existing_evidence != evidence:
                raise GateError("existing migration evidence belongs to a different source ledger")
        else:
            _atomic_json_new(evidence_path, evidence)
        _verify_file(evidence_path, evidence_sha256, "protocol migration evidence")
        if _sha256(args.ledger) != source_ledger_sha256:
            raise GateError("source ledger changed while migration evidence was written")
        _verify_file(
            args.registry,
            transition["destination_registry_sha256"],
            "destination registry",
        )
        _verify_file(
            _root_path(transition["target_path"]),
            transition["target_sha256"],
            "unchanged target authority",
        )

        migrated = copy.deepcopy(ledger)
        migration_reference = {
            "schema_version": 1,
            "from_protocol_id": transition["source_protocol_id"],
            "from_registry_sha256": transition["source_registry_sha256"],
            "to_protocol_id": transition["destination_protocol_id"],
            "to_registry_sha256": transition["destination_registry_sha256"],
            "source_ledger_sha256": source_ledger_sha256,
            "evidence_path": _relative(evidence_path),
            "evidence_sha256": evidence_sha256,
            "migrated_at_utc": created_at,
        }
        migrated["protocol_migrations"] = [*prior_migrations, migration_reference]
        migrated["protocol_id"] = transition["destination_protocol_id"]
        migrated["registry_sha256"] = transition["destination_registry_sha256"]
        migrated["accounting"] = _accounting(migrated, registry)
        migrated["updated_at_utc"] = created_at
        if migrated["candidates"] != original["candidates"]:
            raise GateError("migration changed candidate or attempt data")
        if migrated["accounting"] != original["accounting"]:
            raise GateError("migration changed reconciled accounting data")
        if _preserved_ledger_state(migrated) != _preserved_ledger_state(original):
            raise GateError("migration changed non-authority ledger data")

        _verify_file(evidence_path, evidence_sha256, "protocol migration evidence")
        _atomic_json(args.ledger, migrated)
        verified = _load_ledger(args.ledger, args.registry, registry)
        if (
            verified["candidates"] != original["candidates"]
            or verified["accounting"] != original["accounting"]
        ):
            raise GateError("written migration did not preserve ledger data")
    print(
        json.dumps(
            {
                "protocol_id": verified["protocol_id"],
                "registry_sha256": verified["registry_sha256"],
                "source_ledger_sha256": source_ledger_sha256,
                "evidence_path": _relative(evidence_path),
                "evidence_sha256": evidence_sha256,
                "resumed_existing_evidence": evidence_preexisted,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _process_snapshot(pid: int) -> tuple[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return fields[0], fields[19]  # state (field 3), starttime (field 22)
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return None


def _process_identity(pid: int) -> str | None:
    snapshot = _process_snapshot(pid)
    return snapshot[1] if snapshot is not None else None


def _process_alive(pid: int, identity: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    snapshot = _process_snapshot(pid)
    return bool(
        snapshot is not None
        and snapshot[0] != "Z"
        and (identity is None or snapshot[1] == identity)
    )


def _require_process_stopped(pid: int, identity: str, label: str) -> None:
    """Require proof that a recorded worker identity is no longer live."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise GateError(f"cannot prove that {label} worker stopped") from error
    snapshot = _process_snapshot(pid)
    if snapshot is None:
        raise GateError(f"cannot verify the current process identity for {label}")
    state, observed = snapshot
    if state == "Z":
        return
    if observed == identity:
        raise GateError(f"{label} worker is still alive")


def _set_parent_death_signal(expected_parent_pid: int) -> None:
    """Arm Linux parent-death handling before the autotune child executes Python."""

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    # The parent could have exited between fork and prctl.  Exit before exec in
    # that case so no unowned process can reach CUDA.
    if os.getppid() != expected_parent_pid:
        os._exit(125)


def _wait_for_process_identity_to_stop(
    pid: int, identity: str, *, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while _process_alive(pid, identity):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


@contextmanager
def _bound_pidfd(pid: int, identity: str) -> Iterator[int]:
    """Open an exact process handle and reject PID reuse before signalling."""

    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise GateError("pidfd support is required for safe managed-process cleanup")
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError as error:
        raise GateError("managed process exited before its exact handle was opened") from error
    try:
        if _process_identity(pid) != identity:
            raise GateError("managed process identity changed before cleanup")
        yield descriptor
    finally:
        os.close(descriptor)


def _terminate_process_identity_and_wait(pid: int, identity: str) -> None:
    """Terminate a non-child process without ever signalling a reused PID."""

    if not _process_alive(pid, identity):
        return
    with _bound_pidfd(pid, identity) as descriptor:
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        if _wait_for_process_identity_to_stop(
            pid, identity, timeout_seconds=S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
        ):
            return
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        if not _wait_for_process_identity_to_stop(
            pid, identity, timeout_seconds=S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
        ):
            raise GateError("autotune wrapper did not stop after SIGKILL")


def _terminate_process_group_and_wait(
    *,
    pid: int,
    identity: str,
    process_group_id: int,
    child: subprocess.Popen[Any] | None = None,
    absolute_deadline: dt.datetime | None = None,
) -> None:
    """Stop the exact private-group leader without signalling a reusable PGID."""

    if child is not None and child.pid != pid:
        raise GateError("autotune child handle differs from its immutable identity")
    leader_alive = _process_alive(pid, identity)
    if not leader_alive:
        if child is not None:
            child.wait(timeout=0)
        return
    if process_group_id != pid:
        raise GateError("refusing to signal an unbound autotune process group")
    with _bound_pidfd(pid, identity) as descriptor:
        if os.getpgid(pid) != process_group_id:
            raise GateError("refusing to signal an unbound autotune process group")
        # Signal the exact group leader through pidfd.  The current autotune has
        # no worker subprocesses; the private group is retained as an isolation
        # boundary, while pidfd avoids every PID/PGID-reuse signalling race.
        term_timeout = S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
        if absolute_deadline is not None:
            remaining = (absolute_deadline - dt.datetime.now(dt.UTC)).total_seconds()
            term_timeout = min(
                term_timeout,
                max(
                    0.0,
                    remaining
                    - S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
                    - S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS,
                ),
            )
        if term_timeout > 0:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            if child is not None:
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=term_timeout)
            else:
                _wait_for_process_identity_to_stop(
                    pid,
                    identity,
                    timeout_seconds=term_timeout,
                )
        if _process_alive(pid, identity):
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            kill_timeout = S5_AUTOTUNE_TERMINATION_GRACE_SECONDS
            if absolute_deadline is not None:
                kill_timeout = min(
                    kill_timeout,
                    max(
                        0.0,
                        (absolute_deadline - dt.datetime.now(dt.UTC)).total_seconds()
                        - S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS,
                    ),
                )
            if child is not None:
                try:
                    child.wait(timeout=kill_timeout)
                except subprocess.TimeoutExpired as error:
                    raise GateError("autotune child did not stop after SIGKILL") from error
            elif not _wait_for_process_identity_to_stop(
                pid,
                identity,
                timeout_seconds=kill_timeout,
            ):
                raise GateError("autotune child did not stop after SIGKILL")


def _validate_autotune_child_identity(
    record: dict[str, Any],
    *,
    launch: dict[str, Any],
    launch_binding: dict[str, str],
) -> None:
    """Validate the persisted child identity before trusting or signalling it."""

    required_integer_fields = ("wrapper_pid", "child_pid", "child_process_group_id")
    if any(
        not isinstance(record.get(field), int)
        or isinstance(record.get(field), bool)
        or int(record[field]) <= 0
        for field in required_integer_fields
    ):
        raise GateError("S5 autotune child identity has invalid process identifiers")
    if (
        record.get("schema_version") != 1
        or record.get("kind") != S5_AUTOTUNE_CHILD_KIND
        or record.get("job_id") != launch.get("job_id")
        or int(record.get("attempt", -1)) != int(launch.get("attempt", -2))
        or record.get("launch_record") != launch_binding
        or int(record["wrapper_pid"]) <= 0
        or not str(record.get("wrapper_process_start_ticks", ""))
        or not str(record.get("child_process_start_ticks", ""))
        or int(record["child_process_group_id"]) != int(record["child_pid"])
        or record.get("started_at") != launch.get("started_at")
        or record.get("deadline_at") != launch.get("deadline_at")
        or record.get("command_sha256")
        != _canonical_sha256(launch.get("command_without_self_digest"))
    ):
        raise GateError("S5 autotune child identity differs from its immutable launch")


def _bind_autotune_child_record(
    *,
    ledger_path: Path,
    job_id: str,
    launch_binding: dict[str, str],
    child_path: Path,
    child_sha256: str,
) -> None:
    """Atomically bind the spawned GPU child into its preregistered ledger job."""

    binding = {"path": _relative(child_path), "sha256": child_sha256}
    with _ledger_lock(ledger_path):
        ledger = _load_json(ledger_path)
        matches = [
            job for job in ledger.get("external_jobs", []) if job.get("job_id") == job_id
        ]
        if len(matches) != 1:
            raise GateError("autotune child cannot find its unique ledger reservation")
        job = matches[0]
        if (
            job.get("status") != "running"
            or job.get("launch_record") != launch_binding
            or int(job.get("pid", -1)) != os.getpid()
            or str(job.get("process_start_ticks", ""))
            != str(_process_identity(os.getpid()))
        ):
            raise GateError("autotune child cannot bind to an inactive wrapper reservation")
        existing = job.get("child_record")
        if existing is not None and existing != binding:
            raise GateError("autotune child ledger binding is immutable")
        job["child_record"] = binding
        ledger["updated_at_utc"] = _utc_now()
        _atomic_json(ledger_path, ledger)


def _gpu_inventory() -> dict[int, str]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    mapping: dict[int, str] = {}
    for row in gpu_rows:
        index, uuid = (part.strip() for part in row.split(",", 1))
        mapping[int(index)] = uuid
    return mapping


def _normalize_gpu_uuid(value: str) -> str:
    return str(value).removeprefix("GPU-").lower()


def _physical_gpu_uuid(gpu: int) -> str:
    mapping = _gpu_inventory()
    if gpu not in mapping:
        raise GateError(f"physical GPU {gpu} does not exist")
    return mapping[gpu]


def _resolve_visible_physical_gpu_uuid(
    cuda_visible_devices: str | None, logical_device_index: int
) -> str:
    mapping = _gpu_inventory()
    tokens = (
        [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]
        if cuda_visible_devices is not None
        else [str(index) for index in sorted(mapping)]
    )
    resolved: list[str] = []
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if index not in mapping:
                raise GateError(f"CUDA_VISIBLE_DEVICES names absent physical GPU {index}")
            resolved.append(mapping[index])
            continue
        matches = [
            uuid
            for uuid in mapping.values()
            if _normalize_gpu_uuid(uuid).startswith(_normalize_gpu_uuid(token))
        ]
        if len(matches) != 1:
            raise GateError(f"cannot uniquely resolve CUDA_VISIBLE_DEVICES token {token!r}")
        resolved.append(matches[0])
    if logical_device_index < 0 or logical_device_index >= len(resolved):
        raise GateError("logical CUDA device is outside CUDA_VISIBLE_DEVICES")
    return resolved[logical_device_index]


def _gpu_processes(gpu: int) -> list[dict[str, Any]]:
    mapping = _gpu_inventory()
    if gpu not in mapping:
        raise GateError(f"physical GPU {gpu} does not exist")
    process_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    matches = []
    for row in process_output.splitlines():
        parts = [part.strip() for part in row.split(",", 3)]
        if len(parts) >= 3 and parts[0] == mapping[gpu]:
            matches.append(
                {
                    "pid": int(parts[1]),
                    "process": parts[2],
                    "used_gpu_memory_mib": parts[3] if len(parts) == 4 else None,
                }
            )
    return matches


def _resume_path(candidate: dict[str, Any]) -> Path:
    return _root_path(candidate["checkpoint_dir"]) / f"student-{candidate['variant']}-resume.pt"


def _resume_input_snapshot_path(slot: str, attempt_number: int) -> Path:
    return _root_path(
        "results/reproduction/distillation/"
        f"performance-recovery-resume-input-{slot}-attempt{attempt_number:02d}.pt"
    )


def _candidate_artifact_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    checkpoint_dir = _root_path(candidate["checkpoint_dir"])
    paths = {
        "output": _root_path(candidate["output"]),
        "resume_checkpoint": _resume_path(candidate),
        "best_checkpoint": checkpoint_dir / f"student-{candidate['variant']}-best.pt",
        "final_checkpoint": checkpoint_dir / f"student-{candidate['variant']}-final.pt",
    }
    if candidate.get("log"):
        paths["log"] = _root_path(candidate["log"])
    return paths


def _failed_attempt_archive_root(candidate: dict[str, Any], slot: str, attempt_number: int) -> Path:
    checkpoint_dir = _root_path(candidate["checkpoint_dir"])
    return _root_path(
        checkpoint_dir.parent.parent
        / "failed-attempt-archives"
        / slot
        / f"attempt{attempt_number:02d}"
    )


def _attempt_protocol_authority(
    registry: dict[str, Any], slot: str, attempt: dict[str, Any]
) -> str:
    attempt_number = int(attempt["attempt"])
    _, _, protocol_id = _load_bound_attempt_launch_snapshot(
        registry,
        slot,
        attempt,
        index=attempt_number,
    )
    return protocol_id


def _verify_terminal_failed_attempt(
    state: dict[str, Any], registry: dict[str, Any], slot: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the last failed attempt and its immutable launch/worker records."""

    attempts = state.get("attempts", [])
    if state.get("status") != "failed" or not attempts:
        raise GateError("fresh retry requires a reconciled terminal failed attempt")
    attempt = attempts[-1]
    if attempt.get("status") != "failed":
        raise GateError("fresh retry requires the latest attempt to be terminal failed")
    if attempt.get("actual_gpu_hours") is None or attempt.get("ended_at") is None:
        raise GateError("fresh retry requires reconciled GPU cost and end time")
    if int(attempt.get("exit_code", 0)) == 0:
        raise GateError("fresh retry refuses an attempt with a successful exit code")
    if _process_alive(int(attempt["pid"]), attempt.get("process_start_ticks")):
        raise GateError("fresh retry refuses an attempt whose recorded worker is still alive")

    launch_path = _root_path(attempt["launch_record"])
    worker_path = _root_path(attempt["worker_record"])
    launch = _load_json(launch_path)
    worker = _load_json(worker_path)
    attempt_protocol_id = _attempt_protocol_authority(registry, slot, attempt)
    if (
        launch.get("protocol_id") != attempt_protocol_id
        or launch.get("candidate_id") != slot
        or launch.get("worker_record") != attempt["worker_record"]
    ):
        raise GateError("failed attempt launch record disagrees with the ledger")
    if (
        worker.get("protocol_id") != attempt_protocol_id
        or worker.get("candidate_id") != slot
        or int(worker.get("exit_code", 0)) != int(attempt["exit_code"])
        or worker.get("artifacts_at_exit", {}) != attempt.get("artifacts_at_exit", {})
    ):
        raise GateError("failed attempt worker record disagrees with the ledger")
    return attempt, launch, worker


def _inspect_fresh_retry(
    state: dict[str, Any],
    registry: dict[str, Any],
    candidate: dict[str, Any],
    slot: str,
) -> dict[str, Any]:
    """Verify and describe a non-mutating, attempt-specific archival plan."""

    attempt, launch, worker = _verify_terminal_failed_attempt(state, registry, slot)
    attempt_number = int(attempt["attempt"])
    expected_paths = _candidate_artifact_paths(candidate)
    recorded = attempt.get("artifacts_at_exit", {})
    unknown_labels = sorted(set(recorded) - set(expected_paths))
    if unknown_labels:
        raise GateError(f"failed attempt has unknown artifacts: {unknown_labels}")

    archive_root = _failed_attempt_archive_root(candidate, slot, attempt_number)
    manifest_path = archive_root / "archive-manifest.json"
    archive_record = attempt.get("fresh_retry_archive")
    if archive_record is not None:
        if archive_record.get("manifest_path") != _relative(manifest_path):
            raise GateError("failed-attempt archive manifest path changed")
        _verify_file(
            manifest_path,
            archive_record["manifest_sha256"],
            "failed-attempt archive manifest",
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for label, item in recorded.items():
        source = _root_path(item["path"])
        if source != expected_paths[label]:
            raise GateError(f"failed-attempt {label} path is not canonical")
        destination = archive_root / f"{label}--{source.name}"
        artifacts[label] = {
            "source_path": _relative(source),
            "archive_path": _relative(destination),
            "sha256": item["sha256"],
        }

    # Older worker records did not include their log. Preserve it explicitly;
    # newer records include it in ``artifacts_at_exit`` and take the branch above.
    log_path = expected_paths.get("log")
    archived_log = archive_root / f"log--{log_path.name}" if log_path else None
    if log_path is not None and "log" not in artifacts:
        if log_path.is_file():
            log_hash = _sha256(log_path)
        elif archived_log is not None and archived_log.is_file():
            log_hash = _sha256(archived_log)
        else:
            raise GateError("failed attempt log is unavailable for archival")
        artifacts["log"] = {
            "source_path": _relative(log_path),
            "archive_path": _relative(archived_log),
            "sha256": log_hash,
        }
    if archive_record is not None and archive_record.get("artifacts") != artifacts:
        raise GateError("failed-attempt archive artifact map changed")

    recorded_sources = {_root_path(item["source_path"]) for item in artifacts.values()}
    checkpoint_dir = _root_path(candidate["checkpoint_dir"])
    if checkpoint_dir.exists():
        unexpected = sorted(
            _relative(path)
            for path in checkpoint_dir.iterdir()
            if path.resolve() not in recorded_sources
        )
        if unexpected:
            raise GateError(
                "fresh retry refuses unrecorded checkpoint artifacts: " + ", ".join(unexpected)
            )
    output_path = _root_path(candidate["output"])
    if output_path.exists() and output_path not in recorded_sources:
        raise GateError("fresh retry refuses an unrecorded candidate output")

    allowed_archive_entries = {_root_path(item["archive_path"]) for item in artifacts.values()} | {
        manifest_path
    }
    if archive_root.exists():
        unexpected = sorted(
            _relative(path)
            for path in archive_root.iterdir()
            if path.resolve() not in allowed_archive_entries
        )
        if unexpected:
            raise GateError(
                "failed-attempt archive contains unexpected entries: " + ", ".join(unexpected)
            )

    for label, item in artifacts.items():
        source = _root_path(item["source_path"])
        destination = _root_path(item["archive_path"])
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if not source_exists and not destination_exists:
            raise GateError(f"failed-attempt {label} is missing from source and archive")
        if source_exists:
            _verify_file(source, item["sha256"], f"failed-attempt {label}")
        if destination_exists:
            _verify_file(destination, item["sha256"], f"archived failed-attempt {label}")

    if archive_record is not None:
        manifest = _load_json(manifest_path)
        if (
            manifest.get("protocol_id") != registry["protocol"]["id"]
            or manifest.get("candidate_id") != slot
            or int(manifest.get("attempt", -1)) != attempt_number
            or manifest.get("artifacts") != artifacts
        ):
            raise GateError("failed-attempt archive manifest contents changed")

    return {
        "source_attempt": attempt_number,
        "source_launch_record": {
            "path": attempt["launch_record"],
            "sha256": _sha256(_root_path(attempt["launch_record"])),
        },
        "source_worker_record": {
            "path": attempt["worker_record"],
            "sha256": _sha256(_root_path(attempt["worker_record"])),
        },
        "source_git_commit": launch["git_commit"],
        "source_exit_code": worker["exit_code"],
        "archive_root": _relative(archive_root),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": (
            archive_record["manifest_sha256"] if archive_record is not None else None
        ),
        "artifacts": artifacts,
        "archive_complete": archive_record is not None,
    }


def _archive_failed_attempt(
    plan: dict[str, Any], registry: dict[str, Any], slot: str
) -> dict[str, Any]:
    """Atomically hard-link then unlink canonical failed-attempt artifacts."""

    archive_root = _root_path(plan["archive_root"])
    manifest_path = _root_path(plan["manifest_path"])
    archive_root.mkdir(parents=True, exist_ok=True)
    for label, item in plan["artifacts"].items():
        source = _root_path(item["source_path"])
        destination = _root_path(item["archive_path"])
        if source.is_file():
            _verify_file(source, item["sha256"], f"failed-attempt {label}")
            if not destination.exists():
                # Source and archive are within the same repository filesystem.
                # Hard-linking first makes interruption recoverable and refuses
                # to overwrite an existing historical destination.
                os.link(source, destination)
            _verify_file(destination, item["sha256"], f"archived failed-attempt {label}")
            source.unlink()
        else:
            _verify_file(destination, item["sha256"], f"archived failed-attempt {label}")

    payload = {
        "schema_version": 1,
        "protocol_id": registry["protocol"]["id"],
        "candidate_id": slot,
        "attempt": plan["source_attempt"],
        "archived_at_utc": _utc_now(),
        "reason": "explicit fresh retry from the declared initialization after terminal failure",
        "source_launch_record": plan["source_launch_record"],
        "source_worker_record": plan["source_worker_record"],
        "artifacts": plan["artifacts"],
    }
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        for key in (
            "schema_version",
            "protocol_id",
            "candidate_id",
            "attempt",
            "reason",
            "source_launch_record",
            "source_worker_record",
            "artifacts",
        ):
            if existing.get(key) != payload.get(key):
                raise GateError("existing failed-attempt archive manifest is inconsistent")
    else:
        _atomic_json(manifest_path, payload)
    return {
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "artifacts": plan["artifacts"],
    }


def _build_training_command(
    registry: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resume: Path | None,
    resume_sha256: str | None,
    physical_gpu_uuid: str,
) -> list[str]:
    selection_args = registry["trainer"]["required_selection_arguments"]
    command = [
        registry["trainer"]["python"],
        registry["trainer"]["path"],
        candidate["config"],
        registry["corpus"]["plan"],
        "--variant",
        candidate["variant"],
        "--training-seed",
        str(registry["screening"]["seed"]),
        "--split-seed",
        str(registry["screening"]["split_seed"]),
        "--data-root",
        registry["corpus"]["data_root"],
        "--cache-root",
        registry["corpus"]["cache_root"],
        "--checkpoint-dir",
        candidate["checkpoint_dir"],
        "--output",
        candidate["output"],
        "--max-steps",
        str(registry["screening"]["maximum_steps"]),
        selection_args["manifest"],
        registry["selection_split"]["manifest"],
        selection_args["partition"],
        selection_args["partition_value"],
        "--expected-physical-gpu-uuid",
        physical_gpu_uuid,
    ]
    if resume is not None:
        if resume_sha256 is None:
            raise GateError("resume command lacks its adjacent immutable digest")
        command.extend(("--resume", _relative(resume), "--expected-resume-sha256", resume_sha256))
    elif candidate["initialization"]["kind"] == "checkpoint":
        command.extend(("--initialize-from", candidate["initialization"]["path"]))
    return command


def _verify_resume(
    resume: Path, state: dict[str, Any], registry: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bytes, str]:
    attempts = state.get("attempts", [])
    if not attempts:
        raise GateError("a first S5 launch cannot resume a pre-existing canonical checkpoint")
    artifact = attempts[-1].get("artifacts_at_exit", {}).get("resume_checkpoint")
    if not isinstance(artifact, dict):
        raise GateError("the immediately preceding attempt did not record a resume checkpoint")
    payload, observed_hash, encoded = _load_torch_snapshot(resume)
    if artifact.get("path") != _relative(resume) or observed_hash != artifact.get("sha256"):
        raise GateError("resume checkpoint changed since the preceding recorded attempt")
    if int(payload["step"]) >= int(registry["screening"]["maximum_steps"]):
        raise GateError("resume checkpoint has already reached the screen step ceiling")
    if int(payload.get("training_seed", -1)) != int(registry["screening"]["seed"]):
        raise GateError("resume checkpoint training seed mismatch")
    if int(payload.get("split_seed", -1)) != int(registry["screening"]["split_seed"]):
        raise GateError("resume checkpoint split seed mismatch")
    if candidate.get("maximum_estimated_physical_gpu_hours") is not None:
        expected_recipe, expected_recipe_sha256 = _s5_training_recipe(registry, candidate)
        if (
            payload.get("training_recipe_fingerprint") != expected_recipe
            or payload.get("training_recipe_sha256") != expected_recipe_sha256
        ):
            raise GateError("S5 resume training recipe differs from the active registry")
        expected_origin, expected_origin_sha256 = _s5_initialization_origin(candidate)
        if (
            payload.get("initialization_origin_fingerprint") != expected_origin
            or payload.get("initialization_origin_sha256") != expected_origin_sha256
        ):
            raise GateError("S5 resume seeded origin differs from the active registry")
    return encoded, observed_hash


def _input_hashes(
    registry_path: Path,
    registry: dict[str, Any],
    candidate: dict[str, Any],
    resume: Path | None,
    resume_sha256: str | None = None,
    exact_source_sha256: dict[str, str] | None = None,
    additional_bindings: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    paths = [
        registry_path,
        Path(__file__).resolve(),
        _root_path(registry["trainer"]["path"]),
        _root_path(candidate["config"]),
        _root_path(registry["protocol"]["target_config"]),
        _root_path(registry["corpus"]["plan"]),
        _root_path(registry["corpus"]["cache_audit"]),
        _root_path(registry["selection_split"]["manifest"]),
    ]
    paths.extend(sorted((ROOT / "src/timesfm_lab").rglob("*.py")))
    activation = candidate.get("activation_evidence")
    if isinstance(activation, dict):
        paths.append(_root_path(activation["path"]))
    autotune = candidate.get("throughput_autotune")
    if isinstance(autotune, dict):
        paths.extend((_root_path(autotune["implementation"]), _root_path(autotune["config"])))
    bindings = dict(exact_source_sha256 or {})
    for path in paths:
        relative = _relative(path)
        if relative not in bindings:
            bindings[relative] = _sha256(path)
    if resume is not None:
        if resume_sha256 is None:
            raise GateError("resume input hash is absent")
        bindings[_relative(resume)] = resume_sha256
    elif candidate["initialization"]["kind"] == "checkpoint":
        initialization = _root_path(candidate["initialization"]["path"])
        relative = _relative(initialization)
        if relative not in bindings:
            bindings[relative] = _sha256(initialization)
    for binding in additional_bindings or []:
        relative = str(binding["path"])
        digest = str(binding["sha256"])
        existing = bindings.get(relative)
        if existing is not None and existing != digest:
            raise GateError(f"conflicting immutable input binding for {relative}")
        bindings[relative] = digest
    return [
        {"path": relative, "sha256": digest}
        for relative, digest in sorted(bindings.items())
    ]


def _s5_initialization_origin(candidate: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if candidate.get("initialization") != {
        "kind": "seeded_random",
        "state_sha256": S5_INITIALIZATION_STATE_SHA256,
    }:
        raise GateError("S5 registry initialization differs from its frozen seeded origin")
    fingerprint = {
        "schema_version": 1,
        "kind": "seeded_random",
        "random_initialization_sha256": S5_INITIALIZATION_STATE_SHA256,
        "initialization_sha256": S5_INITIALIZATION_STATE_SHA256,
        "initialization_checkpoint": None,
        "initialization_checkpoint_sha256": None,
    }
    return fingerprint, _canonical_sha256(fingerprint)


def _s5_training_recipe(
    registry: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    config_path = _root_path(candidate["config"])
    config = _load_yaml(config_path)
    training = config["training"]
    source_paths = [
        _root_path(registry["trainer"]["path"]),
        *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
    ]
    training_source_sha256 = {_relative(path): _sha256(path) for path in source_paths}
    domain_configuration = {
        "domain_weight_source": {
            str(key): str(value) for key, value in training["domain_weight_source"].items()
        },
        "domain_weights": {
            str(domain): float(weight) for domain, weight in training["domain_weights"].items()
        },
        "outer_training_counts": {
            str(domain): int(count)
            for domain, count in training["domain_weight_outer_training_counts"].items()
        },
        "denominator": training["domain_weight_denominator"],
        "zero_target_window_policy": training["zero_target_window_policy"],
    }
    fingerprint = {
        "schema_version": 1,
        "config_sha256": _sha256(config_path),
        "corpus_plan_sha256": registry["corpus"]["plan_sha256"],
        "selection_split_manifest_sha256": registry["selection_split"]["manifest_sha256"],
        "training_source_sha256": training_source_sha256,
        "variant": candidate["variant"],
        "loss_weights": training["loss_weights"][candidate["variant"]],
        "loss_reduction": "per_window_domain_balanced",
        "domain_weight_configuration_sha256": _canonical_sha256(domain_configuration),
        "training_seed": int(registry["screening"]["seed"]),
        "split_seed": int(registry["screening"]["split_seed"]),
        "validation_partition": "development",
        "logical_batch_size_windows": 256,
        "maximum_steps": int(registry["screening"]["maximum_steps"]),
        "distributed": False,
        "early_stopping_enabled": True,
    }
    return fingerprint, _canonical_sha256(fingerprint)


def _s5_relevant_source_paths(registry_path: Path, candidate: dict[str, Any]) -> list[Path]:
    autotune = candidate["throughput_autotune"]
    probe_config_path = _root_path(autotune["config"])
    probe_config = _load_yaml(probe_config_path)
    authority = probe_config["authority"]
    paths = {
        _root_path(autotune["implementation"]),
        probe_config_path,
        _root_path(candidate["config"]),
        _root_path(probe_config["production_plan"]),
        _root_path(probe_config["selection_split_manifest"]),
        _root_path(probe_config["activation_evidence"]),
        _root_path(authority["training_implementation"]),
        _root_path(authority["loss_implementation"]),
        _root_path(authority["manager_implementation"]),
        *(_root_path(path) for path in authority.get("additional_relevant_files", [])),
        *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
    }
    if registry_path.resolve() not in paths:
        raise GateError("S5 exact-code map does not include the active candidate registry")
    return sorted(paths)


def _s5_autotune_attempt_paths(candidate: dict[str, Any], attempt_number: int) -> dict[str, Path]:
    config = _load_yaml(_root_path(candidate["throughput_autotune"]["config"]))
    output = _root_path(str(config["output_template"]).format(attempt=attempt_number))
    stem = f"performance-recovery-s5-autotune-attempt{attempt_number:02d}"
    return {
        "output": output,
        "launch": _root_path(f"results/reproduction/systems/{stem}-launch.json"),
        "worker": _root_path(f"results/reproduction/systems/{stem}-worker.json"),
        "child": _root_path(f"results/reproduction/systems/{stem}-child.json"),
        "log": _root_path(f"results/raw/{stem}.log"),
    }


def _s5_autotune_reservation(candidate: dict[str, Any]) -> float:
    config = _load_yaml(_root_path(candidate["throughput_autotune"]["config"]))
    return _finite_positive(
        config.get("probe", {}).get("maximum_physical_gpu_hours"),
        "S5 autotune maximum physical GPU-hours",
    )


def _verify_source_map_at_commit(source_map: dict[str, str], commit: str) -> None:
    """Independently bind each measured file to its exact committed Git blob."""

    _require_full_git_oid(commit, "S5 autotune Git commit")
    for relative, expected in source_map.items():
        _, contents = _git_blob(commit, _root_path(relative))
        if hashlib.sha256(contents).hexdigest() != expected:
            raise GateError(f"S5 measured source differs from its Git blob: {relative}")


def _verify_s5_throughput_artifact(
    *,
    artifact: Path,
    payload: dict[str, Any],
    artifact_sha256: str,
    registry_path: Path,
    registry: dict[str, Any],
    candidate: dict[str, Any],
    json_pointer: str,
    external_job: dict[str, Any],
) -> dict[str, Any]:
    autotune = candidate["throughput_autotune"]
    probe_config_path = _root_path(autotune["config"])
    probe_config = _load_yaml(probe_config_path)
    probe = probe_config["probe"]
    extra = payload.get("extra", {})
    attempt_number = int(external_job.get("attempt", -1))
    expected_output = _root_path(
        str(probe_config["output_template"]).format(attempt=attempt_number)
    )
    if artifact.resolve() != expected_output:
        raise GateError("S5 throughput artifact is not at its immutable attempt output path")
    if json_pointer != autotune["json_pointer"]:
        raise GateError("measurement JSON pointer differs from the frozen S5 probe")
    current_commit = _git_commit()
    measured_commit = _require_full_git_oid(
        payload.get("git_commit"), "S5 throughput measured Git commit"
    )
    if payload.get("status") != "succeeded" or payload.get("git_commit") != external_job.get(
        "git_commit"
    ):
        raise GateError("S5 throughput artifact is not a successful managed-code run")
    _require_git_ancestor(measured_commit, current_commit, "S5 throughput measured Git commit")
    source_paths = _s5_relevant_source_paths(registry_path, candidate)
    _require_paths_clean([_relative(path) for path in source_paths])
    expected_source_map = {_relative(path): _sha256(path) for path in source_paths}
    _verify_source_map_at_commit(expected_source_map, measured_commit)
    launch_binding = external_job.get("launch_record")
    if not isinstance(launch_binding, dict):
        raise GateError("S5 throughput job lacks its immutable manager launch")
    expected_extra = {
        "manager_job_id": external_job.get("job_id"),
        "manager_attempt": attempt_number,
        "manager_launch_record": launch_binding,
        "manager_child_record": external_job.get("child_record"),
        "manager_execution_deadline_at": external_job.get("execution_deadline_at"),
        "manager_cleanup_budget_seconds": external_job.get("cleanup_budget_seconds"),
        "imported_module_origins": {
            "train_production_student": str(
                (ROOT / "scripts/train_production_student.py").resolve()
            ),
            "timesfm_lab.config": str((ROOT / "src/timesfm_lab/config.py").resolve()),
            "timesfm_lab.distill.losses": str(
                (ROOT / "src/timesfm_lab/distill/losses.py").resolve()
            ),
            "timesfm_lab.models": str((ROOT / "src/timesfm_lab/models/__init__.py").resolve()),
            "timesfm_lab.run_record": str((ROOT / "src/timesfm_lab/run_record.py").resolve()),
        },
        "candidate_config_sha256": _sha256(_root_path(candidate["config"])),
        "production_plan_sha256": registry["corpus"]["plan_sha256"],
        "selection_split_manifest_sha256": registry["selection_split"]["manifest_sha256"],
        "activation_evidence_sha256": candidate["activation_evidence"]["sha256"],
        "training_implementation_sha256": _sha256(_root_path(registry["trainer"]["path"])),
        "loss_implementation_sha256": _sha256(_root_path("src/timesfm_lab/distill/losses.py")),
        "manager_implementation_sha256": _sha256(Path(__file__).resolve()),
        "autotune_implementation_sha256": _sha256(_root_path(autotune["implementation"])),
        "autotune_config_sha256": _sha256(probe_config_path),
        "relevant_source_sha256": expected_source_map,
        "output_path": _relative(artifact),
        "initialization_state_sha256": probe["initialization_state_sha256"],
        "replayed_training_sequence_sha256": probe["expected_replayed_training_sequence_sha256"],
        "warmup_optimizer_steps": int(probe["warmup_optimizer_steps"]),
        "measured_optimizer_steps": int(probe["measured_optimizer_steps"]),
        "replayed_optimizer_steps": int(probe["warmup_optimizer_steps"])
        + int(probe["measured_optimizer_steps"]),
        "replayed_windows": int(probe["expected_replayed_windows"]),
        "measured_windows": int(probe["expected_measured_windows"]),
        "physical_gpu_count": 1,
        "logical_cuda_device": str(probe["device"]),
    }
    mismatched = sorted(
        key for key, expected in expected_extra.items() if extra.get(key) != expected
    )
    if mismatched:
        raise GateError(
            "S5 throughput evidence differs from exact launch code: " + ", ".join(mismatched)
        )
    measurements = extra.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != int(
        probe["measured_optimizer_steps"]
    ):
        raise GateError("S5 throughput artifact has incomplete measured optimizer steps")
    if sum(int(item.get("windows", -1)) for item in measurements) != int(
        probe["expected_measured_windows"]
    ):
        raise GateError("S5 throughput artifact has the wrong measured window count")
    semantics = extra.get("training_semantics", {})
    if (
        semantics.get("order")
        != "exact production _epoch_batches(seed=42, epoch=0) followed by "
        "_pack_logical_batches(..., 256)"
        or semantics.get("physical_batch_map")
        != _load_yaml(_root_path(candidate["config"]))["training"]["batch_size_by_context"]
    ):
        raise GateError("S5 throughput evidence changed order or physical batches")
    variant_summary = extra.get("variant_summaries", {}).get(candidate["variant"], {})
    if (
        variant_summary.get("all_gradients_finite") is not True
        or variant_summary.get("all_parameters_finite") is not True
        or variant_summary.get("stable_manage_json_pointer") != json_pointer
    ):
        raise GateError("S5 throughput evidence lacks finite exact-variant proof")
    elapsed = float(extra.get("physical_gpu_elapsed_seconds", math.nan))
    physical_gpu = int(extra.get("physical_gpu_index", -1))
    requested_uuid = str(extra.get("requested_physical_gpu_uuid", ""))
    runtime_uuid = str(extra.get("runtime_physical_gpu_uuid", ""))
    current_uuid = _physical_gpu_uuid(physical_gpu)
    logical_device = torch.device(str(extra.get("logical_cuda_device", "")))
    if logical_device.type != "cuda" or logical_device.index is None:
        raise GateError("S5 throughput artifact lacks an explicit logical CUDA device")
    visible_uuid = _resolve_visible_physical_gpu_uuid(
        extra.get("cuda_visible_devices"), logical_device.index
    )
    if (
        elapsed <= 0
        or not math.isfinite(elapsed)
        or physical_gpu != int(external_job.get("physical_gpu", -1))
        or _normalize_gpu_uuid(requested_uuid) != _normalize_gpu_uuid(runtime_uuid)
        or _normalize_gpu_uuid(requested_uuid) != _normalize_gpu_uuid(current_uuid)
        or _normalize_gpu_uuid(requested_uuid) != _normalize_gpu_uuid(visible_uuid)
        or _normalize_gpu_uuid(requested_uuid)
        != _normalize_gpu_uuid(str(external_job.get("physical_gpu_uuid", "")))
    ):
        raise GateError("S5 throughput artifact has invalid physical-GPU binding or elapsed time")
    observed = float(_json_pointer(payload, json_pointer))
    if observed <= 0 or not math.isfinite(observed):
        raise GateError("S5 measured throughput must be finite and positive")
    return {
        "artifact_sha256": artifact_sha256,
        "windows_per_second": observed,
        "physical_gpu": physical_gpu,
        "physical_gpu_uuid": current_uuid,
        "physical_gpu_elapsed_seconds": float(external_job["elapsed_seconds"]),
        "probe_reported_physical_gpu_elapsed_seconds": elapsed,
        "external_job_id": external_job["job_id"],
        "measured_git_commit": measured_commit,
        "relevant_source_sha256": expected_source_map,
        "relevant_source_map_sha256": _canonical_sha256(expected_source_map),
    }


def _s5_autotune_jobs(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        job
        for job in ledger.get("external_jobs", [])
        if job.get("category") == S5_AUTOTUNE_CATEGORY
    ]


def _s5_autotune_job_for_artifact(
    ledger: dict[str, Any], artifact: Path, artifact_sha256: str
) -> dict[str, Any]:
    matches = [
        job
        for job in _s5_autotune_jobs(ledger)
        if job.get("artifact") == _relative(artifact)
        and job.get("artifact_sha256") == artifact_sha256
    ]
    if len(matches) != 1:
        raise GateError("S5 throughput artifact lacks one preregistered reconciled GPU attempt")
    job = matches[0]
    _validate_s5_autotune_job(job)
    if job.get("status") != "completed" or job.get("outcome") != "succeeded":
        raise GateError("S5 throughput artifact did not come from a successful managed attempt")
    return job


def _verify_s5_autotune_accounting(ledger: dict[str, Any], measurement: dict[str, Any]) -> None:
    job_id = measurement.get("external_job_id")
    matches = [job for job in ledger.get("external_jobs", []) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise GateError("S5 throughput autotune is not charged exactly once to the GPU ledger")
    job = matches[0]
    _validate_s5_autotune_job(job)
    expected_hours = float(job.get("elapsed_seconds", math.nan)) / 3600.0
    if (
        job.get("category") != S5_AUTOTUNE_CATEGORY
        or job.get("status") != "completed"
        or job.get("outcome") != "succeeded"
        or job.get("artifact") != measurement["source"]
        or job.get("artifact_sha256") != measurement["source_sha256"]
        or int(job.get("physical_gpu_count", -1)) != 1
        or int(job.get("physical_gpu", -1)) != int(measurement.get("physical_gpu", -2))
        or _normalize_gpu_uuid(str(job.get("physical_gpu_uuid", "")))
        != _normalize_gpu_uuid(str(measurement.get("physical_gpu_uuid", "")))
        or not math.isclose(
            float(job.get("elapsed_seconds", math.nan)),
            float(measurement.get("autotune_elapsed_seconds", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isfinite(expected_hours)
        or not math.isclose(
            float(job.get("actual_gpu_hours", math.nan)),
            expected_hours,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise GateError("S5 throughput autotune GPU accounting differs from its artifact")


def _revalidate_s5_measurement_for_launch(
    *,
    measurement: dict[str, Any],
    ledger: dict[str, Any],
    registry_path: Path,
    registry: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Reject evidence made stale by any clean commit or launch-relevant byte change."""

    artifact = _root_path(measurement["source"])
    payload, artifact_sha256 = _load_json_snapshot(artifact)
    if artifact_sha256 != measurement["source_sha256"]:
        raise GateError("S5 launch throughput snapshot changed")
    external_job = _s5_autotune_job_for_artifact(ledger, artifact, artifact_sha256)
    metadata = _verify_s5_throughput_artifact(
        artifact=artifact,
        payload=payload,
        artifact_sha256=artifact_sha256,
        registry_path=registry_path,
        registry=registry,
        candidate=candidate,
        json_pointer=str(measurement["json_pointer"]),
        external_job=external_job,
    )
    if (
        metadata["external_job_id"] != measurement.get("external_job_id")
        or not math.isclose(
            float(metadata["windows_per_second"]),
            float(measurement["windows_per_second"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(metadata["physical_gpu_elapsed_seconds"]),
            float(measurement.get("autotune_elapsed_seconds", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise GateError("S5 launch measurement differs from its exact-code managed attempt")
    return {
        "schema_version": 1,
        "measurement_artifact": {
            "path": _relative(artifact),
            "sha256": artifact_sha256,
        },
        "external_job_id": metadata["external_job_id"],
        "measured_git_commit": metadata["measured_git_commit"],
        "relevant_source_sha256": metadata["relevant_source_sha256"],
        "relevant_source_map_sha256": metadata["relevant_source_map_sha256"],
    }


def _verify_launch_exact_code_authority(launch: dict[str, Any]) -> None:
    """Verify the worker uses the exact source snapshot authorized by measurement."""

    authority = launch.get("throughput_exact_code_authority")
    if not isinstance(authority, dict) or authority.get("schema_version") != 1:
        raise GateError("S5 launch lacks its measured exact-code authority")
    source_map = authority.get("relevant_source_sha256")
    if (
        not isinstance(source_map, dict)
        or not source_map
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in source_map.items()
        )
        or authority.get("relevant_source_map_sha256") != _canonical_sha256(source_map)
    ):
        raise GateError("S5 launch exact-code source map is invalid")
    measurement = launch.get("throughput_measurement")
    artifact = authority.get("measurement_artifact")
    if (
        not isinstance(measurement, dict)
        or not isinstance(artifact, dict)
        or set(artifact) != {"path", "sha256"}
        or artifact.get("path") != measurement.get("source")
        or artifact.get("sha256") != measurement.get("source_sha256")
        or authority.get("external_job_id") != measurement.get("external_job_id")
    ):
        raise GateError("S5 launch exact-code authority changed measurement binding")
    input_items = launch.get("input_hashes")
    if not isinstance(input_items, list):
        raise GateError("S5 launch input bindings are absent")
    input_map = {
        str(item.get("path")): str(item.get("sha256"))
        for item in input_items
        if isinstance(item, dict)
    }
    if len(input_map) != len(input_items):
        raise GateError("S5 launch input bindings contain duplicates or invalid entries")
    for relative, digest in source_map.items():
        if input_map.get(relative) != digest:
            raise GateError(f"S5 launch input map is not measurement-bound: {relative}")
        _verify_file(_root_path(relative), digest, f"S5 measured launch source {relative}")
    if input_map.get(str(artifact["path"])) != str(artifact["sha256"]):
        raise GateError("S5 launch omits its immutable measurement artifact")
    _verify_file(
        _root_path(str(artifact["path"])),
        str(artifact["sha256"]),
        "S5 launch throughput artifact",
    )
    measured_commit = _require_full_git_oid(
        authority.get("measured_git_commit"), "S5 launch measured Git commit"
    )
    _require_git_ancestor(measured_commit, str(launch.get("git_commit", "")), "S5 launch code")
    _verify_source_map_at_commit(source_map, measured_commit)


def _verify_trainer_source_authority(
    binding: Any,
    *,
    launch: dict[str, Any],
    launch_binding: dict[str, str],
) -> dict[str, Any]:
    """Independently validate the source snapshot emitted by the S5 trainer."""

    _verify_launch_exact_code_authority(launch)
    authority = launch["throughput_exact_code_authority"]
    source_map = authority["relevant_source_sha256"]
    if not isinstance(binding, dict) or set(binding) != {
        "schema_version",
        "manager_launch_record",
        "measured_git_commit",
        "relevant_source_sha256",
        "relevant_source_map_sha256",
        "loaded_module_origins",
        "loaded_source_sha256",
    }:
        raise GateError("S5 result lacks its exact trainer source authority")
    if (
        binding.get("schema_version") != 1
        or binding.get("manager_launch_record") != launch_binding
        or binding.get("measured_git_commit") != authority.get("measured_git_commit")
        or binding.get("relevant_source_sha256") != source_map
        or binding.get("relevant_source_map_sha256")
        != authority.get("relevant_source_map_sha256")
    ):
        raise GateError("S5 trainer source authority differs from its immutable launch")
    origins = binding.get("loaded_module_origins")
    loaded = binding.get("loaded_source_sha256")
    if not isinstance(origins, dict) or not isinstance(loaded, dict):
        raise GateError("S5 trainer loaded-source authority is malformed")
    required_modules = {
        "timesfm_lab.config",
        "timesfm_lab.distill.data",
        "timesfm_lab.distill.losses",
        "timesfm_lab.models",
        "timesfm_lab.run_record",
    }
    if not required_modules.issubset(origins):
        raise GateError("S5 trainer omitted required imported-module origins")
    expected_loaded_paths = {
        str(relative) for relative in origins.values()
    } | {str(launch["command"][1])}
    if set(loaded) != expected_loaded_paths:
        raise GateError("S5 trainer loaded-source map differs from its module origins")
    for module_name, relative in origins.items():
        if (
            not isinstance(module_name, str)
            or not isinstance(relative, str)
            or not (
                module_name == "timesfm_lab" or module_name.startswith("timesfm_lab.")
            )
            or source_map.get(relative) != loaded.get(relative)
        ):
            raise GateError("S5 trainer imported source outside its measured map")
    for relative, digest in loaded.items():
        if source_map.get(relative) != digest:
            raise GateError("S5 trainer loaded-source digest differs from its measured map")
        _verify_file(
            _root_path(relative),
            str(digest),
            f"S5 trainer loaded source {relative}",
        )
    return binding


def _validate_result(
    output: Path,
    slot: str,
    candidate: dict[str, Any],
    registry: dict[str, Any],
    launch: dict[str, Any],
    *,
    launch_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    result, result_sha256 = _load_json_snapshot(output)
    if result.get("status") != "succeeded":
        raise GateError("trainer result is not succeeded")
    extra = result.get("extra", {})
    training = extra.get("training", {})
    expected_config = f"{candidate['config']};{registry['corpus']['plan']}"
    checks = {
        "config_path": result.get("config_path") == expected_config,
        "git_commit": result.get("git_commit") == launch["git_commit"],
        "variant": extra.get("variant") == candidate["variant"],
        "training_seed": extra.get("training_seed") == registry["screening"]["seed"],
        "split_seed": extra.get("validation_split_seed") == registry["screening"]["split_seed"],
        "parameter_count": extra.get("parameter_count") == candidate["parameter_count"],
        "world_size": training.get("world_size") == 1,
        "step_ceiling": 0 < int(training.get("steps", 0)) <= registry["screening"]["maximum_steps"],
        "declared_max_steps": training.get("maximum_steps")
        == registry["screening"]["maximum_steps"],
        "example_ceiling": 0
        < int(training.get("windows_processed", 0))
        <= registry["screening"]["maximum_examples_processed"],
        "physical_gpu_uuid": _normalize_gpu_uuid(str(extra.get("runtime_physical_gpu_uuid", "")))
        == _normalize_gpu_uuid(str(launch.get("physical_gpu_uuid", ""))),
        "expected_physical_gpu_uuid": _normalize_gpu_uuid(
            str(extra.get("expected_physical_gpu_uuid", ""))
        )
        == _normalize_gpu_uuid(str(launch.get("physical_gpu_uuid", ""))),
    }
    required = registry["screening"]["required_result_provenance"]
    checks.update(
        {
            "selection_manifest": extra.get("selection_split_manifest_sha256")
            == required["selection_split_manifest_sha256"],
            "development_only": extra.get("validation_partition")
            == required["validation_partition"],
            "confirmation_unaccessed": extra.get("confirmation_partition_accessed")
            is required["confirmation_partition_accessed"],
        }
    )
    initialization = candidate["initialization"]
    if initialization["kind"] == "checkpoint" and not launch["resumed"]:
        checks["initialization_checkpoint"] = (
            extra.get("initialization_checkpoint_sha256") == initialization["file_sha256"]
        )
        checks["loaded_initial_state"] = (
            extra.get("initialization_sha256") == initialization["loaded_state_sha256"]
        )
    elif initialization["kind"] == "seeded_random" and not launch["resumed"]:
        checks["seeded_initial_state"] = (
            extra.get("initialization_sha256") == initialization["state_sha256"]
        )
    if candidate.get("maximum_estimated_physical_gpu_hours") is not None:
        if launch_binding is None:
            raise GateError("S5 result validation lacks its immutable launch binding")
        trainer_source_authority = _verify_trainer_source_authority(
            extra.get("manager_source_authority"),
            launch=launch,
            launch_binding=launch_binding,
        )
        expected_recipe, expected_recipe_sha256 = _s5_training_recipe(registry, candidate)
        expected_origin, expected_origin_sha256 = _s5_initialization_origin(candidate)
        checks.update(
            {
                "s5_training_recipe": extra.get("training_recipe_fingerprint") == expected_recipe,
                "s5_training_recipe_sha256": extra.get("training_recipe_sha256")
                == expected_recipe_sha256,
                "s5_initialization_origin": extra.get("initialization_origin_fingerprint")
                == expected_origin,
                "s5_initialization_origin_sha256": extra.get("initialization_origin_sha256")
                == expected_origin_sha256,
                "s5_no_initialization_checkpoint": extra.get("initialization_checkpoint") is None,
            }
        )
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise GateError(f"{slot} result provenance failed: {', '.join(failed)}")
    checkpoints = {}
    for label in ("best_checkpoint", "final_checkpoint"):
        path = Path(extra[label]).resolve()
        path.relative_to(ROOT)
        checkpoints[label] = {"path": _relative(path), "sha256": _sha256(path)}
    return {
        "checks": checks,
        "checkpoints": checkpoints,
        "result_sha256": result_sha256,
        "trainer_source_authority": (
            trainer_source_authority
            if candidate.get("maximum_estimated_physical_gpu_hours") is not None
            else None
        ),
    }


def _verify_success_artifacts_against_worker(
    *,
    worker: dict[str, Any],
    candidate: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    """Require result and checkpoints to be the exact bytes hashed at worker exit."""

    recorded = worker.get("artifacts_at_exit")
    if not isinstance(recorded, dict):
        raise GateError("successful worker lacks its immutable artifact map")
    expected_paths = _candidate_artifact_paths(candidate)
    for label in ("output", "best_checkpoint", "final_checkpoint"):
        binding = recorded.get(label)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise GateError(f"successful worker lacks its {label} binding")
        if binding["path"] != _relative(expected_paths[label]):
            raise GateError(f"successful worker changed its canonical {label} path")
    if validation.get("result_sha256") != recorded["output"]["sha256"]:
        raise GateError("trainer result changed after the worker exit snapshot")
    for label in ("best_checkpoint", "final_checkpoint"):
        if validation.get("checkpoints", {}).get(label) != recorded[label]:
            raise GateError(f"{label} changed after the worker exit snapshot")
    if validation.get("trainer_source_authority") != worker.get(
        "trainer_source_authority_at_exit"
    ):
        raise GateError("trainer source authority changed after the worker exit snapshot")


def _verify_current_terminal_attempt_artifacts(
    ledger: dict[str, Any], registry: dict[str, Any]
) -> None:
    """Recheck immutable current-protocol worker and success artifacts on every load."""

    protocol_id = str(registry["protocol"]["id"])
    for slot, state in ledger["candidates"].items():
        candidate = registry["candidates"][slot]
        for index, attempt in enumerate(state.get("attempts", []), start=1):
            if attempt.get("status") not in TERMINAL_ATTEMPT_STATUSES:
                continue
            launch, launch_sha256, launch_protocol = _load_bound_attempt_launch_snapshot(
                registry,
                slot,
                attempt,
                index=index,
            )
            if launch_protocol != protocol_id:
                continue
            worker_path = _root_path(attempt["worker_record"])
            worker, worker_sha256 = _load_json_snapshot(worker_path)
            if attempt.get("worker_record_sha256") != worker_sha256:
                raise GateError(f"{slot} attempt {index} worker record changed")
            _verify_reconciled_worker(
                worker,
                attempt,
                protocol_id=protocol_id,
                slot=slot,
                index=index,
            )
            if int(worker["exit_code"]) != 0:
                continue
            recorded = worker.get("artifacts_at_exit", {})
            expected_paths = _candidate_artifact_paths(candidate)
            for label in ("output", "best_checkpoint", "final_checkpoint"):
                binding = recorded.get(label)
                if (
                    not isinstance(binding, dict)
                    or binding.get("path") != _relative(expected_paths[label])
                ):
                    raise GateError(f"{slot} attempt {index} lacks canonical {label} evidence")
                _verify_file(
                    expected_paths[label],
                    str(binding.get("sha256", "")),
                    f"{slot} attempt {index} immutable {label}",
                )
            validation = attempt.get("result_validation")
            if attempt.get("status") == "succeeded" and (
                not isinstance(validation, dict)
                or validation.get("result_sha256") != recorded["output"]["sha256"]
                or validation.get("checkpoints", {}).get("best_checkpoint")
                != recorded["best_checkpoint"]
                or validation.get("checkpoints", {}).get("final_checkpoint")
                != recorded["final_checkpoint"]
            ):
                raise GateError(f"{slot} attempt {index} validation is not worker-bound")
            if slot == "S5" and attempt.get("status") == "succeeded":
                trainer_authority = _verify_trainer_source_authority(
                    worker.get("trainer_source_authority_at_exit"),
                    launch=launch,
                    launch_binding={
                        "path": attempt["launch_record"],
                        "sha256": launch_sha256,
                    },
                )
                if validation.get("trainer_source_authority") != trainer_authority:
                    raise GateError(
                        f"{slot} attempt {index} trainer source authority is not reconciled"
                    )


def _reconcile_s5_autotune_jobs(ledger: dict[str, Any]) -> None:
    for job in _s5_autotune_jobs(ledger):
        if job.get("status") not in {"launching", "running"}:
            continue
        launch, launch_sha256 = _load_json_snapshot(_root_path(job["launch_record"]["path"]))
        if launch_sha256 != job["launch_record"]["sha256"]:
            raise GateError("S5 autotune launch changed before reconciliation")
        worker_path = _root_path(job["worker_record"])
        if not worker_path.exists():
            pid = job.get("pid")
            identity = job.get("process_start_ticks")
            wrapper_alive = pid is not None and _process_alive(int(pid), str(identity))
            deadline_elapsed = dt.datetime.now(dt.UTC) >= _parse_utc(
                str(job["deadline_at"])
            )
            if wrapper_alive and not deadline_elapsed:
                continue
            child = _bind_or_load_autotune_child_for_reconcile(job, launch)
            if child is not None and _process_alive(
                int(child[0]["child_pid"]), str(child[0]["child_process_start_ticks"])
            ):
                _terminate_process_group_and_wait(
                    pid=int(child[0]["child_pid"]),
                    identity=str(child[0]["child_process_start_ticks"]),
                    process_group_id=int(child[0]["child_process_group_id"]),
                )
            if wrapper_alive:
                _terminate_process_identity_and_wait(int(pid), str(identity))
            job["status"] = "unreconciled"
            job["failure"] = (
                "autotune absolute deadline elapsed; child and wrapper were terminated, "
                "and the attempt is charged fail-closed"
                if deadline_elapsed
                else "autotune worker exited without immutable terminal evidence; "
                "any bound GPU child was terminated and awaited"
            )
            continue
        worker, worker_sha256 = _load_json_snapshot(worker_path)
        elapsed = _finite_positive(worker.get("elapsed_seconds"), "S5 autotune elapsed seconds")
        if worker.get("launch_record") != job["launch_record"]:
            raise GateError("S5 autotune worker changed its launch binding")
        child_binding = worker.get("child_record")
        if child_binding is not None and job.get("child_record") is None:
            job["child_record"] = child_binding
        child = _load_bound_autotune_child(job, launch)
        if child_binding != job.get("child_record"):
            raise GateError("S5 autotune worker changed its child binding")
        if child is not None and _process_alive(
            int(child[0]["child_pid"]), str(child[0]["child_process_start_ticks"])
        ):
            _terminate_process_group_and_wait(
                pid=int(child[0]["child_pid"]),
                identity=str(child[0]["child_process_start_ticks"]),
                process_group_id=int(child[0]["child_process_group_id"]),
            )
        try:
            _verify_autotune_worker_deadline(worker, launch)
        except GateError as error:
            job["status"] = "unreconciled"
            job["failure"] = str(error)
            continue
        artifact = worker.get("artifact_at_exit")
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise GateError("S5 autotune worker lacks its output binding")
        job.update(
            {
                "status": "completed",
                "outcome": "succeeded" if int(worker.get("exit_code", 1)) == 0 else "failed",
                "ended_at": worker.get("ended_at"),
                "elapsed_seconds": elapsed,
                "actual_gpu_hours": elapsed / 3600.0,
                "exit_code": worker.get("exit_code"),
                "artifact_sha256": artifact["sha256"],
                "terminal_record": {
                    "path": _relative(worker_path),
                    "sha256": worker_sha256,
                },
            }
        )
        if worker.get("failure") is not None:
            job["failure"] = worker["failure"]
        _validate_s5_autotune_job(job)


def _reconcile_locked(
    ledger_path: Path, registry_path: Path, registry: dict[str, Any], ledger: dict[str, Any]
) -> None:
    original = copy.deepcopy(ledger)
    _reconcile_s5_autotune_jobs(ledger)
    for slot, state in ledger["candidates"].items():
        if state.get("status") != "running":
            continue
        attempt = state["attempts"][-1]
        worker_path = _root_path(attempt["worker_record"])
        if not worker_path.exists():
            if _process_alive(int(attempt["pid"]), attempt.get("process_start_ticks")):
                continue
            state["status"] = "unreconciled"
            attempt["status"] = "unreconciled"
            attempt["failure"] = "worker exited without an auditable completion record"
            continue
        attempt_number = int(attempt["attempt"])
        launch, launch_sha256, _launch_protocol = _load_bound_attempt_launch_snapshot(
            registry,
            slot,
            attempt,
            index=attempt_number,
        )
        worker, worker_sha256 = _load_json_snapshot(worker_path)
        attempt["ended_at"] = worker["ended_at"]
        attempt["elapsed_seconds"] = worker["elapsed_seconds"]
        attempt["actual_gpu_hours"] = float(worker["elapsed_seconds"]) / 3600.0
        attempt["exit_code"] = worker["exit_code"]
        attempt["artifacts_at_exit"] = worker.get("artifacts_at_exit", {})
        if "trainer_source_authority_at_exit" in worker:
            attempt["trainer_source_authority_at_exit"] = worker[
                "trainer_source_authority_at_exit"
            ]
        attempt["worker_record_sha256"] = worker_sha256
        candidate = registry["candidates"][slot]
        if worker["exit_code"] == 0:
            try:
                validation = _validate_result(
                    _root_path(candidate["output"]),
                    slot,
                    candidate,
                    registry,
                    launch,
                    launch_binding={
                        "path": attempt["launch_record"],
                        "sha256": launch_sha256,
                    },
                )
                _verify_success_artifacts_against_worker(
                    worker=worker,
                    candidate=candidate,
                    validation=validation,
                )
            except Exception as error:
                attempt["status"] = "invalid"
                attempt["failure"] = str(error)
                state["status"] = "invalid"
            else:
                attempt["status"] = "succeeded"
                attempt["result_validation"] = validation
                state["status"] = "succeeded"
        else:
            attempt["status"] = "failed"
            attempt["failure"] = worker.get("failure", f"trainer exit code {worker['exit_code']}")
            state["status"] = "failed"
    ledger["accounting"] = _accounting(ledger, registry)
    if ledger != original:
        ledger["updated_at_utc"] = _utc_now()
        _atomic_json(ledger_path, ledger)


def _command_status(args: argparse.Namespace) -> int:
    try:
        registry = _verify_registry(args.registry, deep=not args.shallow)
        ledger = _load_ledger(args.ledger, args.registry, registry, allow_predecessor=True)
        account = _accounting(ledger, registry)
        candidates = {}
        for slot, candidate in registry["candidates"].items():
            state = ledger["candidates"][slot]
            measurement = _measurement(slot, candidate, ledger)
            candidates[slot] = {
                "status": state["status"],
                "launchable": candidate.get("launchable", False),
                "variant": candidate.get("variant"),
                "measurement_ready": measurement is not None,
                "estimated_gpu_hours": (
                    measurement.get("estimated_gpu_hours") if measurement else None
                ),
                "attempts": len(state.get("attempts", [])),
            }
        payload = {
            "ready": True,
            "migration_required": ledger.get("protocol_id") != registry["protocol"]["id"],
            "ledger_protocol_id": ledger.get("protocol_id"),
            "registry_protocol_id": registry["protocol"]["id"],
            "accounting": account,
            "candidates": candidates,
        }
        exit_code = 0
    except Exception as error:
        payload = {"ready": False, "failure": str(error)}
        exit_code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def _command_autotune(args: argparse.Namespace) -> int:
    """Preregister and launch one immutable S5 exact-code throughput attempt."""

    if args.slot != "S5":
        raise GateError("the managed exact-code autotune is defined only for S5")
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=True)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        _reconcile_locked(args.ledger, args.registry, registry, ledger)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        candidate = registry["candidates"]["S5"]
        state = ledger["candidates"]["S5"]
        if state.get("status") != "declared" or state.get("attempts"):
            raise GateError("S5 autotune is forbidden after any S5 training attempt")
        if _measurement("S5", candidate, ledger) is not None:
            raise GateError("S5 already has frozen throughput evidence")
        prior = _s5_autotune_jobs(ledger)
        if any(job.get("status") in {"launching", "running", "unreconciled"} for job in prior):
            raise GateError("a prior S5 autotune is active or unreconciled")
        if any(
            job.get("status") == "completed" and job.get("outcome") == "succeeded" for job in prior
        ):
            raise GateError("a successful managed S5 autotune already awaits measurement")
        attempt_number = len(prior) + 1
        paths = _s5_autotune_attempt_paths(candidate, attempt_number)
        occupied_paths = [path for path in paths.values() if path.exists()]
        if occupied_paths:
            raise GateError(
                "immutable S5 autotune attempt paths already exist: "
                + ", ".join(_relative(path) for path in occupied_paths)
            )
        physical_gpu_uuid = _physical_gpu_uuid(args.gpu)
        occupied = _gpu_processes(args.gpu)
        if occupied:
            raise GateError(f"physical GPU {args.gpu} is occupied: {occupied}")
        if any(
            job.get("status") in {"launching", "running"}
            and int(job.get("physical_gpu", -1)) == args.gpu
            for job in ledger.get("external_jobs", [])
        ) or any(
            state.get("status") == "running"
            and int(state["attempts"][-1]["physical_gpu"]) == args.gpu
            for state in ledger["candidates"].values()
        ):
            raise GateError(f"physical GPU {args.gpu} is already reserved in the ledger")
        reservation = _s5_autotune_reservation(candidate)
        account = _accounting(ledger, registry)
        cap = float(ledger["hard_cap_physical_gpu_hours"])
        if account["committed_gpu_hours"] + reservation > cap:
            raise GateError("S5 autotune reservation would exceed the global GPU-hour cap")
        source_paths = _s5_relevant_source_paths(args.registry, candidate)
        _require_paths_clean([_relative(path) for path in source_paths])
        input_hashes = [{"path": _relative(path), "sha256": _sha256(path)} for path in source_paths]
        job_id = f"S5-exact-throughput-autotune-attempt{attempt_number:02d}"
        started_at = _utc_now()
        deadline_at = (
            _parse_utc(started_at) + dt.timedelta(hours=reservation)
        ).isoformat().replace("+00:00", "Z")
        cleanup_budget_seconds = _s5_autotune_cleanup_budget_seconds()
        if cleanup_budget_seconds >= reservation * 3600.0:
            raise GateError("S5 autotune reservation cannot contain its hard cleanup budget")
        execution_deadline_at = (
            _parse_utc(deadline_at) - dt.timedelta(seconds=cleanup_budget_seconds)
        ).isoformat().replace("+00:00", "Z")
        command = [
            registry["trainer"]["python"],
            candidate["throughput_autotune"]["implementation"],
            candidate["throughput_autotune"]["config"],
            "--output",
            _relative(paths["output"]),
            "--physical-gpu-index",
            str(args.gpu),
            "--manager-launch-record",
            _relative(paths["launch"]),
            "--manager-ledger",
            _relative(args.ledger),
            "--manager-child-record",
            _relative(paths["child"]),
        ]
        launch = {
            "schema_version": 1,
            "kind": S5_AUTOTUNE_CATEGORY,
            "protocol_id": registry["protocol"]["id"],
            "job_id": job_id,
            "attempt": attempt_number,
            "git_commit": _git_commit(),
            "declared_at_utc": started_at,
            "started_at": started_at,
            "deadline_at": deadline_at,
            "execution_deadline_at": execution_deadline_at,
            "cleanup_budget_seconds": cleanup_budget_seconds,
            "ledger": _relative(args.ledger),
            "physical_gpu": args.gpu,
            "physical_gpu_uuid": physical_gpu_uuid,
            "physical_gpu_count": 1,
            "reserved_gpu_hours": reservation,
            "implementation": candidate["throughput_autotune"]["implementation"],
            "config": candidate["throughput_autotune"]["config"],
            "output": _relative(paths["output"]),
            "worker_record": _relative(paths["worker"]),
            "child_identity_record": _relative(paths["child"]),
            "log": _relative(paths["log"]),
            "input_hashes": input_hashes,
            "command_without_self_digest": command,
        }
        _atomic_json_new(paths["launch"], launch)
        launch_sha256 = _sha256(paths["launch"])
        job = {
            "job_id": job_id,
            "category": S5_AUTOTUNE_CATEGORY,
            "status": "launching",
            "attempt": attempt_number,
            "git_commit": launch["git_commit"],
            "started_at": started_at,
            "deadline_at": deadline_at,
            "execution_deadline_at": execution_deadline_at,
            "cleanup_budget_seconds": cleanup_budget_seconds,
            "estimated_gpu_hours": reservation,
            "actual_gpu_hours": None,
            "physical_gpu_count": 1,
            "physical_gpu": args.gpu,
            "physical_gpu_uuid": physical_gpu_uuid,
            "launch_record": {
                "path": _relative(paths["launch"]),
                "sha256": launch_sha256,
            },
            "worker_record": _relative(paths["worker"]),
            "child_identity_record": _relative(paths["child"]),
            "artifact": _relative(paths["output"]),
            "basis": "Preregistered exact-code S5 throughput autotune attempt.",
        }
        ledger.setdefault("external_jobs", []).append(job)
        _write_ledger(args.ledger, ledger, registry)
        worker_command = [
            str(_python_executable(registry["trainer"]["python"])),
            __file__,
            "_autotune_worker",
            "--launch-record",
            str(paths["launch"]),
            "--expected-launch-sha256",
            launch_sha256,
            "--worker-ledger",
            str(args.ledger),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["PYTHONPATH"] = str((ROOT / "src").resolve())
        environment.pop("GIFT_EVAL", None)
        try:
            process = subprocess.Popen(
                worker_command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            job["status"] = "unreconciled"
            job["failure"] = "manager could not start the preregistered autotune worker"
            _write_ledger(args.ledger, ledger, registry)
            raise
        identity = _process_identity(process.pid)
        if identity is None:
            job["status"] = "unreconciled"
            job["failure"] = "manager could not bind the autotune worker process identity"
            _write_ledger(args.ledger, ledger, registry)
            raise GateError("could not bind the launched autotune worker identity")
        job.update(
            {
                "status": "running",
                "pid": process.pid,
                "process_start_ticks": identity,
            }
        )
        _write_ledger(args.ledger, ledger, registry)
    print(
        json.dumps(
            {
                "job_id": job_id,
                "attempt": attempt_number,
                "pid": process.pid,
                "artifact": _relative(paths["output"]),
                "launch_record": job["launch_record"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_measure(args: argparse.Namespace) -> int:
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=False)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        _reconcile_locked(args.ledger, args.registry, registry, ledger)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        candidate = registry["candidates"][args.slot]
        state = ledger["candidates"][args.slot]
        if candidate.get("throughput_measurement") is not None:
            raise GateError(f"{args.slot} already has frozen throughput evidence")
        if state["status"] != "declared" or state.get("attempts"):
            raise GateError("throughput evidence cannot change after a launch attempt")
        artifact = _root_path(args.artifact)
        payload, artifact_sha256 = _load_json_snapshot(artifact)
        if payload.get("status") not in {"success", "succeeded"}:
            raise GateError("measurement artifact must have successful status")
        extra = payload.get("extra", {})
        candidate_config = _root_path(candidate["config"])
        if extra.get("candidate_config_sha256") != _sha256(candidate_config):
            raise GateError(
                "measurement artifact was produced with a different candidate configuration"
            )
        if extra.get("production_plan_sha256") != registry["corpus"]["plan_sha256"]:
            raise GateError("measurement artifact corpus-plan hash mismatch")
        variant_summary = extra.get("variant_summaries", {}).get(candidate["variant"], {})
        if variant_summary.get("all_gradients_finite") is not True:
            raise GateError("measurement did not establish finite gradients for this variant")
        autotune = candidate.get("throughput_autotune")
        autotune_metadata = None
        if autotune is not None:
            external_job = _s5_autotune_job_for_artifact(ledger, artifact, artifact_sha256)
            autotune_metadata = _verify_s5_throughput_artifact(
                artifact=artifact,
                payload=payload,
                artifact_sha256=artifact_sha256,
                registry_path=args.registry,
                registry=registry,
                candidate=candidate,
                json_pointer=args.json_pointer,
                external_job=external_job,
            )
        windows_per_second = float(_json_pointer(payload, args.json_pointer))
        if windows_per_second <= 0 or not math.isfinite(windows_per_second):
            raise GateError("measured windows/second must be finite and positive")
        estimated, overhead, safety, examples = _derived_measurement_gpu_hours(
            windows_per_second, registry
        )
        if candidate.get("maximum_estimated_physical_gpu_hours") is not None and estimated > float(
            candidate["maximum_estimated_physical_gpu_hours"]
        ):
            raise GateError("measured S5 throughput exceeds its frozen 29.1-hour estimate gate")
        external_job_id = (
            autotune_metadata["external_job_id"] if autotune_metadata is not None else None
        )
        state["throughput_measurement"] = {
            "windows_per_second": windows_per_second,
            "source": _relative(artifact),
            "source_sha256": artifact_sha256,
            "json_pointer": args.json_pointer,
            "estimated_fixed_overhead_seconds": overhead,
            "safety_factor": safety,
            "maximum_examples_processed": examples,
            "estimated_gpu_hours": estimated,
            "recorded_at_utc": _utc_now(),
        }
        if external_job_id is not None:
            state["throughput_measurement"]["external_job_id"] = external_job_id
            state["throughput_measurement"]["physical_gpu"] = autotune_metadata["physical_gpu"]
            state["throughput_measurement"]["physical_gpu_uuid"] = autotune_metadata[
                "physical_gpu_uuid"
            ]
            state["throughput_measurement"]["autotune_elapsed_seconds"] = autotune_metadata[
                "physical_gpu_elapsed_seconds"
            ]
        _validate_measurement(state["throughput_measurement"], candidate, registry)
        _write_ledger(args.ledger, ledger, registry)
    print(json.dumps(state["throughput_measurement"], indent=2, sort_keys=True))
    return 0


def _command_remeasure(args: argparse.Namespace) -> int:
    """Fail closed: post-launch evidence replacement requires a new protocol."""

    raise GateError(
        "remeasure is disabled fail-closed; authorize a new immutable protocol/slot instead"
    )

    # Kept unreachable for one release so old ledgers remain readable while the
    # command-line contract fails closed.
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=True)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        candidate = registry["candidates"][args.slot]
        state = ledger["candidates"][args.slot]
        if candidate.get("throughput_measurement") is not None:
            raise GateError("registry-pinned throughput evidence cannot be remeasured")
        attempt, launch, _ = _verify_terminal_failed_attempt(state, registry, args.slot)
        if any(
            item.get("status") in {"running", "succeeded"} for item in state.get("attempts", [])
        ):
            raise GateError("remeasure requires no running or succeeded attempt")
        if _resume_path(candidate).exists():
            raise GateError("remeasure is restricted to failures with no resume checkpoint")
        if not args.reason.strip():
            raise GateError("remeasure requires a non-empty correction reason")

        previous = state.get("throughput_measurement")
        _validate_measurement(previous, candidate, registry)
        if previous.get("replacement_for_failed_attempt") == int(attempt["attempt"]):
            raise GateError("this failed attempt already has replacement throughput evidence")

        artifact = _root_path(args.artifact)
        payload, artifact_sha256 = _load_json_snapshot(artifact)
        if payload.get("status") not in {"success", "succeeded"}:
            raise GateError("replacement measurement artifact must have successful status")
        previous_sources = {
            previous["source"],
            *(
                item["measurement"]["source"]
                for item in state.get("throughput_measurement_history", [])
            ),
        }
        if _relative(artifact) in previous_sources:
            raise GateError("replacement evidence must use a new immutable artifact path")
        if artifact_sha256 == previous["source_sha256"]:
            raise GateError("replacement evidence is byte-identical to stale evidence")

        current_commit = _git_commit()
        if payload.get("git_commit") != current_commit:
            raise GateError("replacement evidence was not measured at the current code commit")
        if current_commit == launch.get("git_commit"):
            raise GateError("replacement evidence does not follow a code correction commit")
        extra = payload.get("extra", {})
        candidate_config = _root_path(candidate["config"])
        if extra.get("candidate_config_sha256") != _sha256(candidate_config):
            raise GateError(
                "replacement artifact was produced with a different candidate configuration"
            )
        if extra.get("production_plan_sha256") != registry["corpus"]["plan_sha256"]:
            raise GateError("replacement artifact corpus-plan hash mismatch")
        variant_summary = extra.get("variant_summaries", {}).get(candidate["variant"], {})
        if variant_summary.get("all_gradients_finite") is not True:
            raise GateError("replacement measurement did not establish finite gradients")
        autotune = candidate.get("throughput_autotune")
        autotune_metadata = None
        if autotune is not None:
            external_job = _s5_autotune_job_for_artifact(ledger, artifact, artifact_sha256)
            autotune_metadata = _verify_s5_throughput_artifact(
                artifact=artifact,
                payload=payload,
                artifact_sha256=artifact_sha256,
                registry_path=args.registry,
                registry=registry,
                candidate=candidate,
                json_pointer=args.json_pointer,
                external_job=external_job,
            )
        windows_per_second = float(_json_pointer(payload, args.json_pointer))
        if windows_per_second <= 0 or not math.isfinite(windows_per_second):
            raise GateError("replacement windows/second must be finite and positive")

        estimated, overhead, safety, examples = _derived_measurement_gpu_hours(
            windows_per_second, registry
        )
        if candidate.get("maximum_estimated_physical_gpu_hours") is not None and estimated > float(
            candidate["maximum_estimated_physical_gpu_hours"]
        ):
            raise GateError("replacement S5 throughput exceeds its frozen 29.1-hour gate")
        external_job_id = (
            autotune_metadata["external_job_id"] if autotune_metadata is not None else None
        )
        replaced_at = _utc_now()
        history_entry = {
            "measurement": previous,
            "superseded_at_utc": replaced_at,
            "superseded_after_failed_attempt": int(attempt["attempt"]),
            "failed_launch_git_commit": launch["git_commit"],
            "replacement_source": _relative(artifact),
            "replacement_source_sha256": artifact_sha256,
            "correction_reason": args.reason.strip(),
        }
        state.setdefault("throughput_measurement_history", []).append(history_entry)
        state["throughput_measurement"] = {
            "windows_per_second": windows_per_second,
            "source": _relative(artifact),
            "source_sha256": artifact_sha256,
            "source_git_commit": current_commit,
            "json_pointer": args.json_pointer,
            "estimated_fixed_overhead_seconds": overhead,
            "safety_factor": safety,
            "maximum_examples_processed": examples,
            "estimated_gpu_hours": estimated,
            "recorded_at_utc": replaced_at,
            "replacement_for_failed_attempt": int(attempt["attempt"]),
            "correction_reason": args.reason.strip(),
        }
        if external_job_id is not None:
            state["throughput_measurement"]["external_job_id"] = external_job_id
            state["throughput_measurement"]["physical_gpu"] = autotune_metadata["physical_gpu"]
            state["throughput_measurement"]["physical_gpu_uuid"] = autotune_metadata[
                "physical_gpu_uuid"
            ]
            state["throughput_measurement"]["autotune_elapsed_seconds"] = autotune_metadata[
                "physical_gpu_elapsed_seconds"
            ]
        _validate_measurement(state["throughput_measurement"], candidate, registry)
        _write_ledger(args.ledger, ledger, registry)
    print(json.dumps(state["throughput_measurement"], indent=2, sort_keys=True))
    return 0


def _prepare_launch(
    args: argparse.Namespace,
    registry: dict[str, Any],
    ledger: dict[str, Any],
    *,
    materialize_resume_snapshot: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    slot = args.slot
    candidate = registry["candidates"][slot]
    state = ledger["candidates"][slot]
    if not candidate.get("launchable"):
        raise GateError(f"{slot} is gated and has no launchable configuration")
    if state["status"] in {"running", "succeeded", "invalid", "unreconciled"}:
        raise GateError(f"{slot} cannot launch from ledger status {state['status']!r}")
    measurement = _measurement(slot, candidate, ledger)
    estimate = _validate_measurement(measurement, candidate, registry)
    exact_code_authority: dict[str, Any] | None = None
    if candidate.get("maximum_estimated_physical_gpu_hours") is not None:
        _verify_s5_autotune_accounting(ledger, measurement)
        exact_code_authority = _revalidate_s5_measurement_for_launch(
            measurement=measurement,
            ledger=ledger,
            registry_path=args.registry,
            registry=registry,
            candidate=candidate,
        )
    physical_gpu_uuid = _physical_gpu_uuid(args.gpu)
    gpu_processes = _gpu_processes(args.gpu)
    if gpu_processes:
        raise GateError(f"physical GPU {args.gpu} is occupied: {gpu_processes}")
    if any(
        other.get("status") == "running" and int(other["attempts"][-1]["physical_gpu"]) == args.gpu
        for other in ledger["candidates"].values()
    ):
        raise GateError(f"physical GPU {args.gpu} is already reserved in the ledger")

    account = _accounting(ledger, registry)
    cap = float(ledger["hard_cap_physical_gpu_hours"])
    if account["committed_gpu_hours"] + estimate > cap:
        raise GateError(
            f"launch would exceed {cap:.3f} GPU-hours: "
            f"{account['committed_gpu_hours']:.3f} committed + {estimate:.3f} estimated"
        )
    attempt_number = len(state.get("attempts", [])) + 1
    fresh_retry_requested = bool(args.fresh_retry_from_initialization)
    fresh_retry = None
    canonical_resume = (
        _resume_path(candidate)
        if state.get("attempts") and _resume_path(candidate).is_file()
        else None
    )
    resume: Path | None = None
    resume_source: dict[str, str] | None = None
    if canonical_resume is not None:
        if fresh_retry_requested:
            raise GateError(
                "fresh retry from initialization is forbidden while a resume checkpoint exists"
            )
        resume_bytes, resume_sha256 = _verify_resume(canonical_resume, state, registry, candidate)
        resume_source = {"path": _relative(canonical_resume), "sha256": resume_sha256}
        resume = _resume_input_snapshot_path(slot, attempt_number)
        if materialize_resume_snapshot:
            _atomic_bytes_new(resume, resume_bytes)
            _verify_file(resume, resume_sha256, "immutable resume input snapshot")
    else:
        resume_sha256 = None
        checkpoint_dir = _root_path(candidate["checkpoint_dir"])
        output = _root_path(candidate["output"])
        if state.get("attempts"):
            if not fresh_retry_requested:
                raise GateError(
                    "preceding attempt has no resumable checkpoint; explicit "
                    "--fresh-retry-from-initialization is required"
                )
            fresh_retry = _inspect_fresh_retry(state, registry, candidate, slot)
        elif fresh_retry_requested:
            raise GateError("fresh retry requires a preceding failed attempt")
        elif output.exists() or (checkpoint_dir.exists() and any(checkpoint_dir.iterdir())):
            raise GateError("untracked pre-existing screen artifacts would be overwritten")

    command = _build_training_command(
        registry,
        candidate,
        resume=resume,
        resume_sha256=resume_sha256,
        physical_gpu_uuid=physical_gpu_uuid,
    )
    command_text = " ".join(command)
    if "evaluate_student" in command_text or "data/gift-eval" in command_text:
        raise GateError("screen command contains prohibited GIFT evaluation access")
    launch_record = _root_path(
        f"results/reproduction/distillation/performance-recovery-launch-{slot}-attempt{attempt_number:02d}.json"
    )
    worker_record = _root_path(
        f"results/reproduction/distillation/performance-recovery-worker-{slot}-attempt{attempt_number:02d}.json"
    )
    launch = {
        "schema_version": 1,
        "protocol_id": registry["protocol"]["id"],
        "candidate_id": slot,
        "family": candidate["family"],
        "hypothesis": candidate["hypothesis"],
        "variant": candidate["variant"],
        "git_commit": _git_commit(),
        "declared_at_utc": _utc_now(),
        "physical_gpu": args.gpu,
        "physical_gpu_uuid": physical_gpu_uuid,
        "physical_gpu_count": 1,
        "step_ceiling": registry["screening"]["maximum_steps"],
        "example_ceiling": registry["screening"]["maximum_examples_processed"],
        "estimated_gpu_hours": estimate,
        "cumulative_committed_gpu_hours_before_launch": account["committed_gpu_hours"],
        "resumed": resume is not None,
        "resume_source": resume_source,
        "resume_snapshot": (
            {"path": _relative(resume), "sha256": resume_sha256} if resume is not None else None
        ),
        "resume_checkpoint_sha256": resume_sha256,
        "fresh_retry_from_initialization": fresh_retry,
        "initialization": candidate["initialization"],
        "throughput_measurement": _measurement(slot, candidate, ledger),
        "throughput_exact_code_authority": exact_code_authority,
        "selection_partition": "development",
        "gift_evaluation": False,
        "command": command,
        "input_hashes": _input_hashes(
            args.registry,
            registry,
            candidate,
            resume,
            resume_sha256,
            exact_source_sha256=(
                exact_code_authority["relevant_source_sha256"]
                if exact_code_authority is not None
                else None
            ),
            additional_bindings=(
                [exact_code_authority["measurement_artifact"]]
                if exact_code_authority is not None
                else None
            ),
        ),
        "worker_record": _relative(worker_record),
    }
    worker_command = [
        str(_python_executable(registry["trainer"]["python"])),
        __file__,
        "_worker",
        "--launch-record",
        str(launch_record),
    ]
    return launch, worker_command


def _command_preflight(args: argparse.Namespace) -> int:
    registry = _verify_registry(args.registry, deep=True)
    ledger = _load_ledger(args.ledger, args.registry, registry)
    launch, _ = _prepare_launch(args, registry, ledger)
    print(json.dumps(launch, indent=2, sort_keys=True))
    return 0


def _command_launch(args: argparse.Namespace) -> int:
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=True)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        _reconcile_locked(args.ledger, args.registry, registry, ledger)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        launch, worker_command = _prepare_launch(
            args, registry, ledger, materialize_resume_snapshot=True
        )
        slot = args.slot
        state = ledger["candidates"][slot]
        fresh_retry = launch.get("fresh_retry_from_initialization")
        if fresh_retry is not None:
            archived = _archive_failed_attempt(fresh_retry, registry, slot)
            source_attempt = state["attempts"][-1]
            source_attempt["fresh_retry_archive"] = archived
            _write_ledger(args.ledger, ledger, registry)
            # Re-run every launch gate against the now-clean canonical paths and
            # bind the completed archive manifest into the launch record.
            launch, worker_command = _prepare_launch(
                args, registry, ledger, materialize_resume_snapshot=True
            )
            if not launch["fresh_retry_from_initialization"]["archive_complete"]:
                raise GateError("failed-attempt archival did not complete")
        attempt_number = len(state.get("attempts", [])) + 1
        launch_record = _root_path(
            f"results/reproduction/distillation/performance-recovery-launch-{slot}-attempt{attempt_number:02d}.json"
        )
        _atomic_json_new(launch_record, launch)
        launch_record_sha256 = _sha256(launch_record)
        worker_command.extend(("--expected-launch-sha256", launch_record_sha256))
        attempt = {
            "attempt": attempt_number,
            "status": "launching",
            "physical_gpu": args.gpu,
            "estimated_gpu_hours": launch["estimated_gpu_hours"],
            "launch_record": _relative(launch_record),
            "launch_record_sha256": launch_record_sha256,
            "worker_record": launch["worker_record"],
            "started_at": _utc_now(),
        }
        if fresh_retry is not None:
            attempt["fresh_retry_from_attempt"] = fresh_retry["source_attempt"]
            attempt["failed_attempt_archive"] = state["attempts"][-1]["fresh_retry_archive"]
        state.setdefault("attempts", []).append(attempt)
        state["status"] = "running"
        _write_ledger(args.ledger, ledger, registry)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["PYTHONPATH"] = "src"
        environment.pop("GIFT_EVAL", None)
        try:
            process = subprocess.Popen(
                worker_command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException as error:
            attempt["status"] = "unreconciled"
            attempt["failure"] = (
                f"worker launch failed without immutable terminal evidence: {error}"
            )
            state["status"] = "unreconciled"
            _write_ledger(args.ledger, ledger, registry)
            raise
        attempt["pid"] = process.pid
        attempt["process_start_ticks"] = _process_identity(process.pid)
        attempt["status"] = "running"
        _write_ledger(args.ledger, ledger, registry)
    summary = {
        "candidate": args.slot,
        "pid": process.pid,
        "launch_record": _relative(launch_record),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _command_reconcile(args: argparse.Namespace) -> int:
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=False)
        ledger = _load_ledger(args.ledger, args.registry, registry, allow_predecessor=True)
        _reconcile_locked(args.ledger, args.registry, registry, ledger)
        ledger = _load_ledger(args.ledger, args.registry, registry, allow_predecessor=True)
    payload = {
        "accounting": ledger["accounting"],
        "candidates": (
            {args.slot: ledger["candidates"][args.slot]} if args.slot else ledger["candidates"]
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _artifact_hashes(candidate: dict[str, Any]) -> dict[str, Any]:
    artifacts = {}
    for name, path in _candidate_artifact_paths(candidate).items():
        if path.is_file():
            artifacts[name] = {"path": _relative(path), "sha256": _sha256(path)}
    return artifacts


def _command_autotune_worker(args: argparse.Namespace) -> int:
    launch, launch_sha256 = _load_json_snapshot(args.launch_record)
    if launch_sha256 != args.expected_launch_sha256:
        raise GateError("autotune worker launch-record byte snapshot changed")
    started_wall = _parse_utc(str(launch["started_at"]))
    deadline = _parse_utc(str(launch["deadline_at"]))
    execution_deadline = _parse_utc(str(launch["execution_deadline_at"]))
    cleanup_budget_seconds = float(launch["cleanup_budget_seconds"])
    if (
        cleanup_budget_seconds != _s5_autotune_cleanup_budget_seconds()
        or execution_deadline
        != deadline - dt.timedelta(seconds=cleanup_budget_seconds)
        or execution_deadline <= started_wall
    ):
        raise GateError("autotune worker has an invalid absolute attempt deadline")
    job_id = str(launch["job_id"])
    launch_binding = {"path": _relative(args.launch_record), "sha256": launch_sha256}
    with _ledger_lock(args.worker_ledger):
        ledger = _load_json(args.worker_ledger)
        matches = [job for job in ledger.get("external_jobs", []) if job.get("job_id") == job_id]
        if len(matches) != 1:
            raise GateError("autotune worker is not uniquely preregistered")
        job = matches[0]
        if (
            job.get("status") != "running"
            or job.get("launch_record") != launch_binding
            or int(job.get("pid", -1)) != os.getpid()
            or str(job.get("process_start_ticks", "")) != str(_process_identity(os.getpid()))
            or job.get("started_at") != launch.get("started_at")
            or job.get("deadline_at") != launch.get("deadline_at")
            or job.get("execution_deadline_at") != launch.get("execution_deadline_at")
            or job.get("cleanup_budget_seconds") != launch.get("cleanup_budget_seconds")
        ):
            raise GateError("autotune worker identity differs from its preregistration")
    exit_code = 1
    failure: str | None = None
    output = _root_path(launch["output"])
    child_path = _root_path(launch["child_identity_record"])
    child: subprocess.Popen[Any] | None = None
    child_identity: str | None = None
    child_group: int | None = None
    child_binding: dict[str, str] | None = None
    child_stopped_at: dt.datetime | None = None
    prior_handlers: dict[int, Any] = {}

    def _interrupt(signum: int, _frame: Any) -> None:
        raise _AutotuneWorkerInterrupted(signum)

    for managed_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        prior_handlers[managed_signal] = signal.getsignal(managed_signal)
        signal.signal(managed_signal, _interrupt)
    try:
        if launch.get("git_commit") != _git_commit():
            raise GateError("autotune code commit changed after preregistration")
        for binding in launch["input_hashes"]:
            _verify_file(_root_path(binding["path"]), binding["sha256"], binding["path"])
        command = list(launch["command_without_self_digest"])
        command.extend(("--expected-launch-sha256", launch_sha256))
        log_path = _root_path(launch["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.pop("GIFT_EVAL", None)

        wrapper_pid = os.getpid()

        def _child_preexec() -> None:
            _set_parent_death_signal(wrapper_pid)

        with log_path.open("xb") as log:
            if dt.datetime.now(dt.UTC) >= execution_deadline:
                raise subprocess.TimeoutExpired(command, 0.0)
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_child_preexec,
            )
            child_identity = _process_identity(child.pid)
            if child_identity is None:
                raise GateError("could not bind the spawned autotune child identity")
            child_group = os.getpgid(child.pid)
            if child_group != child.pid:
                raise GateError("autotune child did not receive a private process group")
            child_record = {
                "schema_version": 1,
                "kind": S5_AUTOTUNE_CHILD_KIND,
                "job_id": job_id,
                "attempt": int(launch["attempt"]),
                "launch_record": launch_binding,
                "wrapper_pid": wrapper_pid,
                "wrapper_process_start_ticks": _process_identity(wrapper_pid),
                "child_pid": child.pid,
                "child_process_start_ticks": child_identity,
                "child_process_group_id": child_group,
                "started_at": launch["started_at"],
                "deadline_at": launch["deadline_at"],
                "command_sha256": _canonical_sha256(launch["command_without_self_digest"]),
            }
            _validate_autotune_child_identity(
                child_record,
                launch=launch,
                launch_binding=launch_binding,
            )
            _atomic_json_new(child_path, child_record)
            child_sha256 = _sha256(child_path)
            child_binding = {"path": _relative(child_path), "sha256": child_sha256}
            _bind_autotune_child_record(
                ledger_path=args.worker_ledger,
                job_id=job_id,
                launch_binding=launch_binding,
                child_path=child_path,
                child_sha256=child_sha256,
            )
            remaining = (execution_deadline - dt.datetime.now(dt.UTC)).total_seconds()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 0.0)
            exit_code = int(child.wait(timeout=remaining))
            child_stopped_at = dt.datetime.now(dt.UTC)
        if exit_code:
            failure = f"autotune subprocess exited with code {exit_code}"
    except subprocess.TimeoutExpired:
        failure = "autotune attempt exceeded its immutable absolute deadline"
        exit_code = 1
    except _AutotuneWorkerInterrupted as error:
        failure = str(error)
        exit_code = 128 + int(error.signum)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        # Disable catch-and-raise handlers while cleanup is in progress.  A
        # SIGKILL still invokes the child's PDEATHSIG safeguard.
        for managed_signal in prior_handlers:
            signal.signal(managed_signal, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            if child_identity is None or child_group is None:
                with suppress(ProcessLookupError):
                    child.kill()
                fallback_timeout = min(
                    S5_AUTOTUNE_TERMINATION_GRACE_SECONDS,
                    max(
                        0.0,
                        (deadline - dt.datetime.now(dt.UTC)).total_seconds()
                        - S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS,
                    ),
                )
                child.wait(timeout=fallback_timeout)
            else:
                _terminate_process_group_and_wait(
                    pid=child.pid,
                    identity=child_identity,
                    process_group_id=child_group,
                    child=child,
                    absolute_deadline=deadline,
                )
        if child is not None and child.poll() is not None and child_stopped_at is None:
            child_stopped_at = dt.datetime.now(dt.UTC)
    if not output.exists():
        _atomic_json_new(
            output,
            {
                "schema_version": 1,
                "status": "failed",
                "git_commit": launch.get("git_commit"),
                "failure": failure or "autotune exited without an output artifact",
                "manager_job_id": job_id,
                "manager_launch_record": {
                    "path": _relative(args.launch_record),
                    "sha256": launch_sha256,
                },
            },
        )
        exit_code = 1
    artifact_payload, artifact_sha256 = _load_json_snapshot(output)
    if artifact_payload.get("status") != "succeeded":
        exit_code = 1
        failure = failure or str(artifact_payload.get("failure", "autotune artifact failed"))
    ended_wall = dt.datetime.now(dt.UTC)
    elapsed = (ended_wall - started_wall).total_seconds()
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise GateError("autotune worker elapsed time is not finite and positive")
    if ended_wall > deadline or (
        child_stopped_at is not None and child_stopped_at > deadline
    ):
        exit_code = 1
        failure = "autotune process or completion exceeded its immutable absolute deadline"
    worker_path = _root_path(launch["worker_record"])
    worker = {
        "schema_version": 1,
        "kind": S5_AUTOTUNE_CATEGORY,
        "protocol_id": launch["protocol_id"],
        "job_id": job_id,
        "attempt": int(launch["attempt"]),
        "launch_record": launch_binding,
        "child_record": child_binding,
        "started_at": started_wall.isoformat().replace("+00:00", "Z"),
        "deadline_at": deadline.isoformat().replace("+00:00", "Z"),
        "child_stopped_at": (
            child_stopped_at.isoformat().replace("+00:00", "Z")
            if child_stopped_at is not None
            else None
        ),
        "ended_at": ended_wall.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": elapsed,
        "physical_gpu": int(launch["physical_gpu"]),
        "physical_gpu_uuid": launch["physical_gpu_uuid"],
        "exit_code": exit_code,
        "failure": failure,
        "artifact_at_exit": {"path": _relative(output), "sha256": artifact_sha256},
    }
    _atomic_json_new(worker_path, worker)
    for managed_signal, prior_handler in prior_handlers.items():
        signal.signal(managed_signal, prior_handler)
    return exit_code


def _command_worker(args: argparse.Namespace) -> int:
    launch, launch_sha256 = _load_json_snapshot(args.launch_record)
    if launch_sha256 != args.expected_launch_sha256:
        raise GateError("worker launch-record byte snapshot changed")
    worker_record = _root_path(launch["worker_record"])
    started = dt.datetime.now(dt.UTC)
    exit_code = 1
    failure = None
    trainer_source_authority_at_exit: dict[str, Any] | None = None
    try:
        for item in launch["input_hashes"]:
            _verify_file(_root_path(item["path"]), item["sha256"], item["path"])
        if launch.get("candidate_id") == "S5":
            _verify_launch_exact_code_authority(launch)
        command = list(launch["command"])
        if launch.get("candidate_id") == "S5":
            command.extend(
                (
                    "--manager-launch-record",
                    _relative(args.launch_record),
                    "--expected-manager-launch-sha256",
                    args.expected_launch_sha256,
                )
            )
        command_text = " ".join(command)
        if "evaluate_student" in command_text or "data/gift-eval" in command_text:
            raise GateError("worker rejected a GIFT evaluation command")
        candidate = {
            "variant": launch["variant"],
            "checkpoint_dir": command[command.index("--checkpoint-dir") + 1],
            "output": command[command.index("--output") + 1],
            "log": f"results/raw/performance-recovery-screen-{launch['candidate_id']}.log",
        }
        log_path = _root_path(candidate["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.pop("GIFT_EVAL", None)
        with log_path.open("ab") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code = completed.returncode
        if exit_code:
            failure = f"trainer exited with code {exit_code}"
        elif launch.get("candidate_id") == "S5":
            result, _ = _load_json_snapshot(_root_path(candidate["output"]))
            trainer_source_authority_at_exit = result.get("extra", {}).get(
                "manager_source_authority"
            )
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        exit_code = 1
        candidate = {
            "variant": launch["variant"],
            "checkpoint_dir": f"checkpoints/performance-recovery/screens/{launch['candidate_id']}",
            "output": (
                "results/reproduction/distillation/performance-recovery-screen-"
                f"{launch['candidate_id']}.json"
            ),
        }
    ended = dt.datetime.now(dt.UTC)
    payload = {
        "schema_version": 1,
        "protocol_id": launch["protocol_id"],
        "candidate_id": launch["candidate_id"],
        "started_at": started.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ended_at": ended.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": (ended - started).total_seconds(),
        "exit_code": exit_code,
        "failure": failure,
        "artifacts_at_exit": _artifact_hashes(candidate),
    }
    if launch.get("candidate_id") == "S5":
        payload["trainer_source_authority_at_exit"] = trainer_source_authority_at_exit
    _atomic_json_new(worker_record, payload)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show gates, measurements, and budget")
    status.add_argument(
        "--shallow",
        action="store_true",
        help="skip model-state/split reconstruction (launch always runs the deep checks)",
    )
    status.set_defaults(handler=_command_status)

    autotune = subparsers.add_parser(
        "autotune", help="preregister and launch an immutable exact-code S5 throughput attempt"
    )
    autotune.add_argument("slot", choices=("S5",))
    autotune.add_argument("--gpu", type=int, required=True)
    autotune.set_defaults(handler=_command_autotune)

    measure = subparsers.add_parser("measure", help="record hash-backed throughput evidence")
    measure.add_argument("slot", choices=SLOTS)
    measure.add_argument("--artifact", type=Path, required=True)
    measure.add_argument("--json-pointer", required=True)
    measure.set_defaults(handler=_command_measure)

    remeasure = subparsers.add_parser(
        "remeasure",
        help="disabled fail-closed; evidence replacement requires a new protocol",
    )
    remeasure.add_argument("slot", choices=SLOTS)
    remeasure.add_argument("--artifact", type=Path, required=True)
    remeasure.add_argument("--json-pointer", required=True)
    remeasure.add_argument("--reason", required=True)
    remeasure.set_defaults(handler=_command_remeasure)

    for name, handler in (("preflight", _command_preflight), ("launch", _command_launch)):
        command = subparsers.add_parser(name)
        command.add_argument("slot", choices=SLOTS)
        command.add_argument("--gpu", type=int, required=True)
        command.add_argument(
            "--fresh-retry-from-initialization",
            action="store_true",
            help=(
                "after a reconciled terminal failure with no resume checkpoint, "
                "archive its artifacts and explicitly restart from the declared initialization"
            ),
        )
        command.set_defaults(handler=handler)

    reconcile = subparsers.add_parser("reconcile", help="ingest completed worker records")
    reconcile.add_argument("slot", choices=SLOTS, nargs="?")
    reconcile.set_defaults(handler=_command_reconcile)

    migrate = subparsers.add_parser(
        "migrate-protocol",
        help="immutably migrate a reconciled v1.1 GPU ledger to the v1.2 registry",
    )
    migrate.set_defaults(handler=_command_migrate_protocol)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--launch-record", type=Path, required=True)
    worker.add_argument("--expected-launch-sha256", required=True)
    worker.set_defaults(handler=_command_worker)
    autotune_worker = subparsers.add_parser("_autotune_worker")
    autotune_worker.add_argument("--launch-record", type=Path, required=True)
    autotune_worker.add_argument("--expected-launch-sha256", required=True)
    autotune_worker.add_argument("--worker-ledger", type=Path, required=True)
    autotune_worker.set_defaults(handler=_command_autotune_worker)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.registry = args.registry.resolve()
    args.ledger = args.ledger.resolve()
    try:
        return int(args.handler(args))
    except (
        GateError,
        FileNotFoundError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
