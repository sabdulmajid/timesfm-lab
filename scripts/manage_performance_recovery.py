#!/usr/bin/env python3
"""Fail-closed launcher and GPU-hour ledger for bounded recovery screens."""

from __future__ import annotations

import argparse
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


def _verify_registry(registry_path: Path, *, deep: bool) -> dict[str, Any]:
    registry = _load_yaml(registry_path)
    protocol = registry["protocol"]
    targets = _load_yaml(_root_path(protocol["target_config"]))
    if registry["schema_version"] != 1 or targets["protocol_id"] != protocol["id"]:
        raise GateError("registry/target protocol mismatch")
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


def _load_ledger(path: Path, registry_path: Path, registry: dict[str, Any]) -> dict[str, Any]:
    ledger = _load_json(path)
    if ledger.get("protocol_id") != registry["protocol"]["id"]:
        raise GateError("ledger protocol mismatch")
    if float(ledger.get("hard_cap_physical_gpu_hours", -1)) != float(
        registry["protocol"]["hard_cap_physical_gpu_hours"]
    ):
        raise GateError("ledger GPU-hour cap mismatch")
    if ledger.get("registry_sha256") != _sha256(registry_path):
        raise GateError("ledger is pinned to a different candidate registry")
    if tuple(ledger.get("candidates", {})) != SLOTS:
        raise GateError("ledger does not contain exactly S1-S6")
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
    _write_ledger(ledger_path, ledger, registry)


def _command_status(args: argparse.Namespace) -> int:
    try:
        registry = _verify_registry(args.registry, deep=not args.shallow)
        ledger = _load_ledger(args.ledger, args.registry, registry)
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
        payload = {"ready": True, "accounting": account, "candidates": candidates}
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
    resume = _resume_path(candidate) if _resume_path(candidate).is_file() else None
    if resume is not None:
        resume_sha256 = _verify_resume(resume, state, registry, candidate)
    else:
        resume_sha256 = None
        checkpoint_dir = _root_path(candidate["checkpoint_dir"])
        output = _root_path(candidate["output"])
        if state.get("attempts"):
            raise GateError("preceding attempt has no resumable checkpoint")
        if output.exists() or (checkpoint_dir.exists() and any(checkpoint_dir.iterdir())):
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
        ledger = _load_ledger(args.ledger, args.registry, registry)
        _reconcile_locked(args.ledger, args.registry, registry, ledger)
        ledger = _load_ledger(args.ledger, args.registry, registry)
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
    paths = {
        "output": _root_path(candidate["output"]),
        "resume_checkpoint": _resume_path(candidate),
        "best_checkpoint": _root_path(candidate["checkpoint_dir"])
        / f"student-{candidate['variant']}-best.pt",
        "final_checkpoint": _root_path(candidate["checkpoint_dir"])
        / f"student-{candidate['variant']}-final.pt",
    }
    for name, path in paths.items():
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

    for name, handler in (("preflight", _command_preflight), ("launch", _command_launch)):
        command = subparsers.add_parser(name)
        command.add_argument("slot", choices=SLOTS)
        command.add_argument("--gpu", type=int, required=True)
        command.set_defaults(handler=handler)

    reconcile = subparsers.add_parser("reconcile", help="ingest completed worker records")
    reconcile.add_argument("slot", choices=SLOTS, nargs="?")
    reconcile.set_defaults(handler=_command_reconcile)

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
