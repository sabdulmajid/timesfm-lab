#!/usr/bin/env python3
"""Fail-closed launcher and GPU-hour ledger for bounded recovery screens."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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


class GateError(RuntimeError):
    """A launch-safety invariant is not satisfied."""


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
    diff_commands = (
        ["git", "diff", "--quiet", "--", *paths],
        ["git", "diff", "--cached", "--quiet", "--", *paths],
    )
    for arguments in diff_commands:
        result = subprocess.run(arguments, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise GateError("launch-relevant tracked files have uncommitted changes")


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
            source = _root_path(measurement["source"])
            _verify_file(source, measurement["source_sha256"], f"{slot} throughput evidence")
            observed = float(_json_pointer(_load_json(source), measurement["json_pointer"]))
            if not math.isclose(
                observed, float(measurement["windows_per_second"]), rel_tol=1e-12, abs_tol=0.0
            ):
                raise GateError(f"{slot}: throughput evidence value mismatch")
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
    elif not (allow_predecessor and predecessor_authority):
        raise GateError("ledger is not pinned to the required protocol/registry authority")
    return ledger


def _measurement(
    slot: str, candidate: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any] | None:
    return candidate.get("throughput_measurement") or ledger["candidates"][slot].get(
        "throughput_measurement"
    )


def _validate_measurement(measurement: dict[str, Any] | None) -> float:
    if not measurement:
        raise GateError("a measured throughput artifact is required before launch")
    source = _root_path(measurement["source"])
    _verify_file(source, measurement["source_sha256"], "throughput evidence")
    observed = float(_json_pointer(_load_json(source), measurement["json_pointer"]))
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
    return estimate


def _attempt_hours(attempt: dict[str, Any], now: dt.datetime) -> float:
    if attempt.get("actual_gpu_hours") is not None:
        return float(attempt["actual_gpu_hours"])
    if attempt.get("status") == "running":
        return max(0.0, (now - _parse_utc(attempt["started_at"])).total_seconds() / 3600.0)
    return 0.0


def _accounting(ledger: dict[str, Any], registry: dict[str, Any]) -> dict[str, float]:
    now = dt.datetime.now(dt.UTC)
    actual = 0.0
    committed = 0.0
    unknown = False
    for job in ledger.get("external_jobs", []):
        if job["status"] == "unreconciled":
            unknown = True
        value = (
            float(job["actual_gpu_hours"])
            if job.get("actual_gpu_hours") is not None
            else float(job["estimated_gpu_hours"])
        )
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
            estimate = _validate_measurement(_measurement(slot, candidate, ledger))
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
        if actual_gpu_hours < 0 or not math.isfinite(actual_gpu_hours):
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


def _process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def _process_alive(pid: int, identity: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return identity is None or _process_identity(pid) == identity


def _require_process_stopped(pid: int, identity: str, label: str) -> None:
    """Require proof that a recorded worker identity is no longer live."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise GateError(f"cannot prove that {label} worker stopped") from error
    observed = _process_identity(pid)
    if observed is None:
        raise GateError(f"cannot verify the current process identity for {label}")
    if observed == identity:
        raise GateError(f"{label} worker is still alive")


def _gpu_processes(gpu: int) -> list[dict[str, Any]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    mapping = {}
    for row in gpu_rows:
        index, uuid = (part.strip() for part in row.split(",", 1))
        mapping[int(index)] = uuid
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


def _failed_attempt_archive_root(
    candidate: dict[str, Any], slot: str, attempt_number: int
) -> Path:
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
    launch_path = _root_path(attempt["launch_record"])
    launch = _load_json(launch_path)
    current_protocol_id = str(registry["protocol"]["id"])
    if launch.get("protocol_id") == current_protocol_id:
        return current_protocol_id
    predecessor_protocol_id = registry["protocol"].get("supersedes", {}).get("protocol_id")
    attempt_number = int(attempt["attempt"])
    for declaration in registry["protocol"].get("grandfathered_launch_records", []):
        if (
            declaration.get("candidate_id") == slot
            and int(declaration.get("attempt", -1)) == attempt_number
            and declaration.get("path") == _relative(launch_path)
            and launch.get("protocol_id") == predecessor_protocol_id
        ):
            _verify_file(
                launch_path,
                declaration["sha256"],
                f"{slot} attempt {attempt_number} grandfathered launch",
            )
            return str(predecessor_protocol_id)
    raise GateError("failed attempt does not have current or migrated protocol authority")


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
                "fresh retry refuses unrecorded checkpoint artifacts: "
                + ", ".join(unexpected)
            )
    output_path = _root_path(candidate["output"])
    if output_path.exists() and output_path not in recorded_sources:
        raise GateError("fresh retry refuses an unrecorded candidate output")

    allowed_archive_entries = {
        _root_path(item["archive_path"]) for item in artifacts.values()
    } | {manifest_path}
    if archive_root.exists():
        unexpected = sorted(
            _relative(path)
            for path in archive_root.iterdir()
            if path.resolve() not in allowed_archive_entries
        )
        if unexpected:
            raise GateError(
                "failed-attempt archive contains unexpected entries: "
                + ", ".join(unexpected)
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
    registry: dict[str, Any], candidate: dict[str, Any], *, resume: Path | None
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
    ]
    if resume is not None:
        command.extend(("--resume", _relative(resume)))
    elif candidate["initialization"]["kind"] == "checkpoint":
        command.extend(("--initialize-from", candidate["initialization"]["path"]))
    return command


def _verify_resume(
    resume: Path, state: dict[str, Any], registry: dict[str, Any], candidate: dict[str, Any]
) -> str:
    observed_hash = _sha256(resume)
    attempts = state.get("attempts", [])
    last_recorded_hash = None
    for attempt in reversed(attempts):
        artifact = attempt.get("artifacts_at_exit", {}).get("resume_checkpoint")
        if artifact:
            last_recorded_hash = artifact["sha256"]
            break
    if last_recorded_hash is not None and observed_hash != last_recorded_hash:
        raise GateError("resume checkpoint changed since the preceding recorded attempt")
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if int(payload["step"]) >= int(registry["screening"]["maximum_steps"]):
        raise GateError("resume checkpoint has already reached the screen step ceiling")
    if int(payload.get("training_seed", -1)) != int(registry["screening"]["seed"]):
        raise GateError("resume checkpoint training seed mismatch")
    if int(payload.get("split_seed", -1)) != int(registry["screening"]["split_seed"]):
        raise GateError("resume checkpoint split seed mismatch")
    return observed_hash


def _input_hashes(
    registry_path: Path,
    registry: dict[str, Any],
    candidate: dict[str, Any],
    resume: Path | None,
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
    if resume is not None:
        paths.append(resume)
    elif candidate["initialization"]["kind"] == "checkpoint":
        paths.append(_root_path(candidate["initialization"]["path"]))
    return [{"path": _relative(path), "sha256": _sha256(path)} for path in paths]


def _validate_result(
    output: Path,
    slot: str,
    candidate: dict[str, Any],
    registry: dict[str, Any],
    launch: dict[str, Any],
) -> dict[str, Any]:
    result = _load_json(output)
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
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise GateError(f"{slot} result provenance failed: {', '.join(failed)}")
    checkpoints = {}
    for label in ("best_checkpoint", "final_checkpoint"):
        path = Path(extra[label]).resolve()
        path.relative_to(ROOT)
        checkpoints[label] = {"path": _relative(path), "sha256": _sha256(path)}
    return {"checks": checks, "checkpoints": checkpoints, "result_sha256": _sha256(output)}


def _reconcile_locked(
    ledger_path: Path, registry_path: Path, registry: dict[str, Any], ledger: dict[str, Any]
) -> None:
    original = copy.deepcopy(ledger)
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
        worker = _load_json(worker_path)
        attempt["ended_at"] = worker["ended_at"]
        attempt["elapsed_seconds"] = worker["elapsed_seconds"]
        attempt["actual_gpu_hours"] = float(worker["elapsed_seconds"]) / 3600.0
        attempt["exit_code"] = worker["exit_code"]
        attempt["artifacts_at_exit"] = worker.get("artifacts_at_exit", {})
        launch = _load_json(_root_path(attempt["launch_record"]))
        candidate = registry["candidates"][slot]
        if worker["exit_code"] == 0:
            try:
                validation = _validate_result(
                    _root_path(candidate["output"]), slot, candidate, registry, launch
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


def _command_measure(args: argparse.Namespace) -> int:
    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=False)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        candidate = registry["candidates"][args.slot]
        state = ledger["candidates"][args.slot]
        if candidate.get("throughput_measurement") is not None:
            raise GateError(f"{args.slot} already has frozen throughput evidence")
        if state["status"] != "declared" or state.get("attempts"):
            raise GateError("throughput evidence cannot change after a launch attempt")
        artifact = _root_path(args.artifact)
        payload = _load_json(artifact)
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
        windows_per_second = float(_json_pointer(payload, args.json_pointer))
        if windows_per_second <= 0 or not math.isfinite(windows_per_second):
            raise GateError("measured windows/second must be finite and positive")
        safety = float(registry["screening"]["measurement_safety_factor"])
        overhead = float(registry["screening"]["estimated_fixed_overhead_seconds"])
        examples = int(registry["screening"]["maximum_examples_processed"])
        estimated = (examples / windows_per_second + overhead) * safety / 3600.0
        state["throughput_measurement"] = {
            "windows_per_second": windows_per_second,
            "source": _relative(artifact),
            "source_sha256": _sha256(artifact),
            "json_pointer": args.json_pointer,
            "estimated_fixed_overhead_seconds": overhead,
            "safety_factor": safety,
            "estimated_gpu_hours": estimated,
            "recorded_at_utc": _utc_now(),
        }
        _write_ledger(args.ledger, ledger, registry)
    print(json.dumps(state["throughput_measurement"], indent=2, sort_keys=True))
    return 0


def _command_remeasure(args: argparse.Namespace) -> int:
    """Replace stale throughput evidence after a terminal numerical failure."""

    with _ledger_lock(args.ledger):
        registry = _verify_registry(args.registry, deep=True)
        ledger = _load_ledger(args.ledger, args.registry, registry)
        candidate = registry["candidates"][args.slot]
        state = ledger["candidates"][args.slot]
        if candidate.get("throughput_measurement") is not None:
            raise GateError("registry-pinned throughput evidence cannot be remeasured")
        attempt, launch, _ = _verify_terminal_failed_attempt(state, registry, args.slot)
        if any(
            item.get("status") in {"running", "succeeded"}
            for item in state.get("attempts", [])
        ):
            raise GateError("remeasure requires no running or succeeded attempt")
        if _resume_path(candidate).exists():
            raise GateError("remeasure is restricted to failures with no resume checkpoint")
        if not args.reason.strip():
            raise GateError("remeasure requires a non-empty correction reason")

        previous = state.get("throughput_measurement")
        _validate_measurement(previous)
        if previous.get("replacement_for_failed_attempt") == int(attempt["attempt"]):
            raise GateError("this failed attempt already has replacement throughput evidence")

        artifact = _root_path(args.artifact)
        payload = _load_json(artifact)
        if payload.get("status") not in {"success", "succeeded"}:
            raise GateError("replacement measurement artifact must have successful status")
        artifact_sha256 = _sha256(artifact)
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
        windows_per_second = float(_json_pointer(payload, args.json_pointer))
        if windows_per_second <= 0 or not math.isfinite(windows_per_second):
            raise GateError("replacement windows/second must be finite and positive")

        safety = float(registry["screening"]["measurement_safety_factor"])
        overhead = float(registry["screening"]["estimated_fixed_overhead_seconds"])
        examples = int(registry["screening"]["maximum_examples_processed"])
        estimated = (examples / windows_per_second + overhead) * safety / 3600.0
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
            "estimated_gpu_hours": estimated,
            "recorded_at_utc": replaced_at,
            "replacement_for_failed_attempt": int(attempt["attempt"]),
            "correction_reason": args.reason.strip(),
        }
        _write_ledger(args.ledger, ledger, registry)
    print(json.dumps(state["throughput_measurement"], indent=2, sort_keys=True))
    return 0


def _prepare_launch(
    args: argparse.Namespace, registry: dict[str, Any], ledger: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    slot = args.slot
    candidate = registry["candidates"][slot]
    state = ledger["candidates"][slot]
    if not candidate.get("launchable"):
        raise GateError(f"{slot} is gated and has no launchable configuration")
    if state["status"] in {"running", "succeeded", "invalid", "unreconciled"}:
        raise GateError(f"{slot} cannot launch from ledger status {state['status']!r}")
    estimate = _validate_measurement(_measurement(slot, candidate, ledger))
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
    fresh_retry_requested = bool(args.fresh_retry_from_initialization)
    fresh_retry = None
    resume = _resume_path(candidate) if _resume_path(candidate).is_file() else None
    if resume is not None:
        if fresh_retry_requested:
            raise GateError(
                "fresh retry from initialization is forbidden while a resume checkpoint exists"
            )
        resume_sha256 = _verify_resume(resume, state, registry, candidate)
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

    command = _build_training_command(registry, candidate, resume=resume)
    command_text = " ".join(command)
    if "evaluate_student" in command_text or "data/gift-eval" in command_text:
        raise GateError("screen command contains prohibited GIFT evaluation access")
    attempt_number = len(state.get("attempts", [])) + 1
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
        "physical_gpu_count": 1,
        "step_ceiling": registry["screening"]["maximum_steps"],
        "example_ceiling": registry["screening"]["maximum_examples_processed"],
        "estimated_gpu_hours": estimate,
        "cumulative_committed_gpu_hours_before_launch": account["committed_gpu_hours"],
        "resumed": resume is not None,
        "resume_checkpoint_sha256": resume_sha256,
        "fresh_retry_from_initialization": fresh_retry,
        "initialization": candidate["initialization"],
        "throughput_measurement": _measurement(slot, candidate, ledger),
        "selection_partition": "development",
        "gift_evaluation": False,
        "command": command,
        "input_hashes": _input_hashes(args.registry, registry, candidate, resume),
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
        launch, worker_command = _prepare_launch(args, registry, ledger)
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
            launch, worker_command = _prepare_launch(args, registry, ledger)
            if not launch["fresh_retry_from_initialization"]["archive_complete"]:
                raise GateError("failed-attempt archival did not complete")
        attempt_number = len(state.get("attempts", [])) + 1
        launch_record = _root_path(
            f"results/reproduction/distillation/performance-recovery-launch-{slot}-attempt{attempt_number:02d}.json"
        )
        _atomic_json(launch_record, launch)
        attempt = {
            "attempt": attempt_number,
            "status": "launching",
            "physical_gpu": args.gpu,
            "estimated_gpu_hours": launch["estimated_gpu_hours"],
            "launch_record": _relative(launch_record),
            "worker_record": launch["worker_record"],
            "started_at": _utc_now(),
        }
        if fresh_retry is not None:
            attempt["fresh_retry_from_attempt"] = fresh_retry["source_attempt"]
            attempt["failed_attempt_archive"] = state["attempts"][-1][
                "fresh_retry_archive"
            ]
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
            attempt["status"] = "failed"
            attempt["failure"] = f"worker launch failed: {error}"
            attempt["actual_gpu_hours"] = 0.0
            state["status"] = "failed"
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


def _command_worker(args: argparse.Namespace) -> int:
    launch = _load_json(args.launch_record)
    worker_record = _root_path(launch["worker_record"])
    started = dt.datetime.now(dt.UTC)
    exit_code = 1
    failure = None
    try:
        for item in launch["input_hashes"]:
            _verify_file(_root_path(item["path"]), item["sha256"], item["path"])
        command = list(launch["command"])
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
    _atomic_json(worker_record, payload)
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

    measure = subparsers.add_parser("measure", help="record hash-backed throughput evidence")
    measure.add_argument("slot", choices=SLOTS)
    measure.add_argument("--artifact", type=Path, required=True)
    measure.add_argument("--json-pointer", required=True)
    measure.set_defaults(handler=_command_measure)

    remeasure = subparsers.add_parser(
        "remeasure",
        help="replace stale throughput evidence after a reconciled numerical failure",
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
    worker.set_defaults(handler=_command_worker)
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
