#!/usr/bin/env python3
"""Measure S5 with its exact production logical-batch and reducer path.

This is deliberately not a synthetic tensor or homogeneous-shape benchmark.
It reconstructs the frozen epoch-zero production order, packs the selected
base's physical microbatches into 256-window optimizer batches, and executes
the per-window, domain-weighted reducer used by the production trainer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import train_production_student as production

import timesfm_lab.config as config_module
import timesfm_lab.distill.losses as losses_module
import timesfm_lab.models as models_module
import timesfm_lab.run_record as run_record_module
from timesfm_lab.config import load_config
from timesfm_lab.distill.losses import DistillationLoss, LossWeights
from timesfm_lab.models import build_student
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


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


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, hashlib.sha256(encoded).hexdigest()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable result; a repeated attempt may never overwrite it."""

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
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"immutable autotune output already exists: {path}") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _process_identity(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def _assert_import_origins() -> dict[str, str]:
    """Reject an alternate installed package even if repository files are hashed."""

    expected = {
        "train_production_student": ROOT / "scripts/train_production_student.py",
        "timesfm_lab.config": ROOT / "src/timesfm_lab/config.py",
        "timesfm_lab.distill.losses": ROOT / "src/timesfm_lab/distill/losses.py",
        "timesfm_lab.models": ROOT / "src/timesfm_lab/models/__init__.py",
        "timesfm_lab.run_record": ROOT / "src/timesfm_lab/run_record.py",
    }
    modules = {
        "train_production_student": production,
        "timesfm_lab.config": config_module,
        "timesfm_lab.distill.losses": losses_module,
        "timesfm_lab.models": models_module,
        "timesfm_lab.run_record": run_record_module,
    }
    observed = {
        name: str(Path(str(getattr(module, "__file__", ""))).resolve())
        for name, module in modules.items()
    }
    for name, path in expected.items():
        if observed[name] != str(path.resolve()):
            raise ValueError(
                f"{name} imported from {observed[name]!r}, expected exact repository path {path}"
            )
    pythonpath = os.environ.get("PYTHONPATH")
    if (
        pythonpath is None
        or len(pythonpath.split(os.pathsep)) != 1
        or Path(pythonpath).resolve() != (ROOT / "src").resolve()
    ):
        raise ValueError("S5 autotune requires PYTHONPATH bound exactly to repository src")
    return observed


def _validate_manager_authorization(
    *,
    launch_path: Path,
    expected_launch_sha256: str,
    ledger_path: Path,
    output: Path,
    physical_gpu: int,
    child_record_path: Path,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    launch, launch_sha256 = _load_json_snapshot(launch_path)
    if launch_sha256 != expected_launch_sha256:
        raise ValueError("manager autotune launch-record snapshot changed")
    parent_pid = os.getppid()
    if _path(launch.get("child_identity_record", "")) != child_record_path.resolve():
        raise ValueError("autotune child-identity path differs from its manager launch")
    # The managed wrapper persists and ledger-binds our exact process identity
    # immediately after Popen.  Wait only for that bounded CPU-side handshake;
    # no CUDA work is allowed before it succeeds.
    authorization_deadline = time.monotonic() + 30.0
    while True:
        ledger = _load_json(ledger_path)
        matches = [
            job
            for job in ledger.get("external_jobs", [])
            if job.get("job_id") == launch.get("job_id")
        ]
        if len(matches) == 1 and matches[0].get("child_record") is not None:
            break
        if time.monotonic() >= authorization_deadline:
            raise ValueError("manager did not persist the autotune child identity")
        time.sleep(0.05)
    if len(matches) != 1:
        raise ValueError("autotune launch is not uniquely preregistered in the GPU ledger")
    job = matches[0]
    child_record, child_sha256 = _load_json_snapshot(child_record_path)
    child_binding = {
        "path": str(child_record_path.relative_to(ROOT)),
        "sha256": child_sha256,
    }
    if (
        launch.get("schema_version") != 1
        or launch.get("kind") != "s5_exact_throughput_autotune"
        or launch.get("git_commit")
        != subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip()
        or _path(launch.get("ledger", "")) != ledger_path.resolve()
        or _path(launch.get("output", "")) != output.resolve()
        or int(launch.get("physical_gpu", -1)) != physical_gpu
        or job.get("status") != "running"
        or job.get("launch_record")
        != {"path": str(launch_path.relative_to(ROOT)), "sha256": launch_sha256}
        or _path(job.get("artifact", "")) != output.resolve()
        or int(job.get("physical_gpu", -1)) != physical_gpu
        or int(job.get("pid", -1)) != parent_pid
        or str(job.get("process_start_ticks", "")) != str(_process_identity(parent_pid))
        or job.get("child_identity_record") != child_binding["path"]
        or job.get("child_record") != child_binding
        or child_record.get("schema_version") != 1
        or child_record.get("kind") != "s5_exact_throughput_autotune_child"
        or child_record.get("job_id") != launch.get("job_id")
        or int(child_record.get("attempt", -1)) != int(launch.get("attempt", -2))
        or child_record.get("launch_record")
        != {"path": str(launch_path.relative_to(ROOT)), "sha256": launch_sha256}
        or int(child_record.get("wrapper_pid", -1)) != parent_pid
        or str(child_record.get("wrapper_process_start_ticks", ""))
        != str(_process_identity(parent_pid))
        or int(child_record.get("child_pid", -1)) != os.getpid()
        or str(child_record.get("child_process_start_ticks", ""))
        != str(_process_identity(os.getpid()))
        or int(child_record.get("child_process_group_id", -1)) != os.getpgrp()
        or os.getpgrp() != os.getpid()
        or child_record.get("started_at") != launch.get("started_at")
        or child_record.get("deadline_at") != launch.get("deadline_at")
        or job.get("execution_deadline_at") != launch.get("execution_deadline_at")
        or job.get("cleanup_budget_seconds") != launch.get("cleanup_budget_seconds")
    ):
        raise ValueError("autotune GPU work lacks a live exact manager preregistration")
    deadline = dt.datetime.fromisoformat(str(launch["deadline_at"]).replace("Z", "+00:00"))
    execution_deadline = dt.datetime.fromisoformat(
        str(launch["execution_deadline_at"]).replace("Z", "+00:00")
    )
    cleanup_budget_seconds = float(launch["cleanup_budget_seconds"])
    if (
        cleanup_budget_seconds <= 0
        or execution_deadline
        != deadline - dt.timedelta(seconds=cleanup_budget_seconds)
        or dt.datetime.now(dt.UTC) >= execution_deadline
    ):
        raise ValueError("autotune execution deadline elapsed before GPU work")
    if output.exists():
        raise ValueError("immutable autotune attempt output already exists")
    for binding in launch.get("input_hashes", []):
        path = _path(binding["path"])
        if _sha256(path) != binding["sha256"]:
            raise ValueError(f"manager-bound autotune input changed: {binding['path']}")
    return launch, launch_sha256, child_binding


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _gpu_inventory() -> dict[int, str]:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {int(index.strip()): uuid.strip() for index, uuid in (row.split(",", 1) for row in rows)}


def _normalize_gpu_uuid(value: str) -> str:
    return str(value).removeprefix("GPU-").lower()


def _resolve_physical_gpu(physical_gpu: int, device: torch.device) -> str:
    mapping = _gpu_inventory()
    if physical_gpu not in mapping:
        raise ValueError(f"physical GPU {physical_gpu} does not exist")
    if device.type != "cuda" or device.index is None:
        raise ValueError("S5 probe requires an explicit logical CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = (
        [token.strip() for token in visible.split(",") if token.strip()]
        if visible is not None
        else [str(index) for index in sorted(mapping)]
    )
    resolved: list[str] = []
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if index not in mapping:
                raise ValueError(f"CUDA_VISIBLE_DEVICES names absent physical GPU {index}")
            resolved.append(mapping[index])
            continue
        matches = [
            uuid
            for uuid in mapping.values()
            if _normalize_gpu_uuid(uuid).startswith(_normalize_gpu_uuid(token))
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely resolve CUDA_VISIBLE_DEVICES token {token!r}")
        resolved.append(matches[0])
    if device.index >= len(resolved):
        raise ValueError(f"logical {device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    requested_uuid = mapping[physical_gpu]
    if _normalize_gpu_uuid(resolved[device.index]) != _normalize_gpu_uuid(requested_uuid):
        raise ValueError(
            f"logical {device} maps to {resolved[device.index]}, not physical GPU "
            f"{physical_gpu} ({requested_uuid})"
        )
    return requested_uuid


def _gpu_processes(physical_gpu: int) -> list[dict[str, Any]]:
    mapping = _gpu_inventory()
    if physical_gpu not in mapping:
        raise ValueError(f"physical GPU {physical_gpu} does not exist")
    processes = subprocess.run(
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
    for row in processes.splitlines():
        fields = [field.strip() for field in row.split(",", 3)]
        if len(fields) >= 3 and fields[0] == mapping[physical_gpu]:
            matches.append(
                {
                    "pid": int(fields[1]),
                    "process": fields[2],
                    "used_gpu_memory_mib": fields[3] if len(fields) == 4 else None,
                }
            )
    return matches


def _relevant_source_paths(config_path: Path, config: dict[str, Any]) -> list[Path]:
    authority = config["authority"]
    paths = {
        Path(__file__).resolve(),
        config_path.resolve(),
        _path(config["candidate_config"]),
        _path(config["production_plan"]),
        _path(config["selection_split_manifest"]),
        _path(config["activation_evidence"]),
        _path(authority["training_implementation"]),
        _path(authority["loss_implementation"]),
        _path(authority["manager_implementation"]),
        *(_path(path) for path in authority.get("additional_relevant_files", [])),
        *sorted((ROOT / "src/timesfm_lab").rglob("*.py")),
    }
    return sorted(paths)


def _require_relevant_tree_clean(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)) for path in paths]
    for command in (
        ["git", "diff", "--quiet", "--", *relative],
        ["git", "diff", "--cached", "--quiet", "--", *relative],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise ValueError("S5 throughput-relevant tracked files have uncommitted changes")


def _validate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority = config["authority"]
    paths = {
        "candidate_config": _path(config["candidate_config"]),
        "production_plan": _path(config["production_plan"]),
        "selection_split_manifest": _path(config["selection_split_manifest"]),
        "activation_evidence": _path(config["activation_evidence"]),
        "training_implementation": _path(authority["training_implementation"]),
        "loss_implementation": _path(authority["loss_implementation"]),
        "manager_implementation": _path(authority["manager_implementation"]),
    }
    for name, path in paths.items():
        expected = str(authority[f"{name}_sha256"])
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected={expected}, observed={observed}")

    candidate = load_config(paths["candidate_config"])
    plan = _load_json(paths["production_plan"])
    selection = _load_json(paths["selection_split_manifest"])
    activation = _load_json(paths["activation_evidence"])
    training = candidate["training"]
    probe = config["probe"]
    variant = str(probe["variant"])

    if config["model_revision"] != candidate["model_revision"]:
        raise ValueError("probe/candidate model revision mismatch")
    if config["dataset_revision"] != candidate["dataset_revision"]:
        raise ValueError("probe/candidate dataset revision mismatch")
    if config["dataset_revision"] != plan["dataset_revision"]:
        raise ValueError("probe/production-plan dataset revision mismatch")
    if selection.get("status") != "frozen_uninspected":
        raise ValueError("selection manifest is not frozen and uninspected")
    if selection.get("target_accessed") or selection.get("teacher_output_accessed"):
        raise ValueError("selection manifest reports prohibited target access")
    if activation.get("status") != "completed" or not activation["decision"]["activate_s5"]:
        raise ValueError("S5 activation evidence did not pass")
    if activation["decision"]["selected_variant"] != "compact_gt":
        raise ValueError("S5 activation evidence did not select the frozen GT base")
    if candidate["student"].get("architecture") != "compact_timesfm3":
        raise ValueError("candidate is not the compact TimesFM-3 architecture")
    if candidate.get("initialization") != {
        "kind": "seeded_random",
        "state_sha256": str(probe["initialization_state_sha256"]),
    }:
        raise ValueError("candidate does not bind the frozen seeded initialization")
    if training.get("loss_reduction") != "per_window_domain_balanced":
        raise ValueError("candidate does not use the S5 per-window reducer")
    if training.get("logical_batch_size_windows") != probe.get("logical_batch_size_windows"):
        raise ValueError("probe/candidate logical-batch mismatch")
    if training.get("domain_weight_denominator") != "unweighted_valid_windows":
        raise ValueError("S5 denominator is not the frozen unweighted-window denominator")
    if (
        training.get("zero_target_window_policy")
        != "sequence_only_excluded_from_loss_and_denominator"
    ):
        raise ValueError("S5 zero-target-window policy changed")
    if variant not in training["loss_weights"]:
        raise ValueError("probe variant is absent from the candidate config")
    expected_objective = {
        "ground_truth": 1.0,
        "multivariate_kd": 0.0,
        "univariate_kd": 0.0,
        "cvrd": 0.0,
    }
    if training["loss_weights"][variant] != expected_objective:
        raise ValueError("S5 objective differs from the selected GT base")
    if (
        training["domain_weights"]
        != activation["frozen_gate"]["s5_domain_weights_source"]["weights"]
    ):
        raise ValueError("S5 domain weights differ from the activation gate")
    if (
        training["domain_weight_outer_training_counts"]
        != activation["frozen_gate"]["s5_domain_weights_source"][
            "counts_include_zero_target_windows"
        ]
    ):
        raise ValueError("S5 domain counts differ from the activation gate")
    if int(probe["epoch"]) != 0:
        raise ValueError("S5 throughput evidence must replay the frozen epoch-zero order")
    if int(probe["warmup_optimizer_steps"]) < 1 or int(probe["measured_optimizer_steps"]) < 1:
        raise ValueError("S5 probe requires warmup and measured optimizer steps")
    expected_sequence = str(probe["expected_replayed_training_sequence_sha256"])
    if len(expected_sequence) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sequence
    ):
        raise ValueError("S5 expected replay sequence must be a lowercase SHA-256")
    return candidate, plan, selection


def _load_corpora(
    candidate: dict[str, Any],
    plan: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
) -> list[Any]:
    training = candidate["training"]
    entries = {str(item["dataset"]): item for item in selection["datasets"]}
    corpora = [
        production._load_corpus(
            item,
            data_root=_path(config["data_root"]),
            cache_root=_path(config["cache_root"]),
            validation_fraction=float(training["validation_fraction"]),
            validation_mode=str(training["validation_split"]),
            seed=int(config["seed"]),
            batch_sizes=training["batch_size_by_context"],
            selection_manifest=selection,
            selection_entry=entries[str(item["dataset"])],
            validation_partition="development",
        )
        for item in plan["datasets"]
    ]
    realized = {domain: 0 for domain in training["domain_weights"]}
    for corpus in corpora:
        if corpus.domain not in realized:
            raise ValueError(f"unfrozen S5 domain: {corpus.domain!r}")
        realized[corpus.domain] += len(corpus.training_indices)
    if realized != training["domain_weight_outer_training_counts"]:
        raise ValueError(f"realized S5 domain counts changed: {realized}")
    return corpora


def _sequence_update(
    chain: bytes, optimizer_batch: list[tuple[int, np.ndarray]], corpora: list[Any]
) -> bytes:
    for corpus_index, indices in optimizer_batch:
        chain = hashlib.sha256(
            chain + corpora[corpus_index].name.encode() + np.asarray(indices, dtype="<i8").tobytes()
        ).digest()
    return chain


def _logical_step(
    *,
    training_model: torch.nn.Module,
    objective: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    optimizer_batch: list[tuple[int, np.ndarray]],
    corpora: list[Any],
    training: dict[str, Any],
    variant: str,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    counts: list[tuple[int, int]] = []
    for corpus_index, indices in optimizer_batch:
        corpus = corpora[corpus_index]
        observed_targets = 0
        valid_windows = 0
        for index in indices:
            row = int(corpus.row_index[index])
            end = int(corpus.context_end[index])
            target = corpus.source[row][:, end : end + corpus.horizon]
            observed = int(np.count_nonzero(np.isfinite(target)))
            observed_targets += observed
            valid_windows += int(observed > 0)
        counts.append((observed_targets, valid_windows))
    logical_valid_windows = sum(valid for _, valid in counts)
    if logical_valid_windows <= 0:
        raise ValueError("S5 replay optimizer batch has no valid target windows")

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    realized_valid_windows = 0
    datasets = []
    for position, (corpus_index, indices) in enumerate(optimizer_batch):
        corpus = corpora[corpus_index]
        datasets.append(corpus.name)
        context, target, teacher_primary, teacher_uv = production._materialize(
            corpus,
            indices,
            device,
            str(training["input_preprocessing"]),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = production._loss(
                training_model,
                objective,
                variant,
                corpus,
                context,
                target,
                teacher_primary,
                teacher_uv,
                float(training_model.student.config.normalization_epsilon),
                "per_window_domain_balanced",
            )
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise FloatingPointError(f"non-finite S5 replay loss on {corpus.name}")
        valid = torch.isfinite(target).flatten(1).any(dim=1)
        observed = int(torch.isfinite(target).sum().item())
        valid_count = int(valid.sum().item())
        if (observed, valid_count) != counts[position]:
            raise RuntimeError("CPU/GPU S5 target-count mismatch")
        if any(value.shape != (len(indices),) for value in values.values()):
            raise RuntimeError("S5 replay did not receive per-window loss components")
        if valid_count:
            weighted = values["loss"][valid].sum() * (
                float(training["domain_weights"][corpus.domain]) / logical_valid_windows
            )
            weighted.backward()
            losses.append(float(weighted.detach()))
        # Match the production statistics reduction so its GPU work is timed.
        torch.stack(
            [
                values[key][valid].detach().to(torch.float64).sum()
                * float(training["domain_weights"][corpus.domain])
                for key in ("loss", "ground_truth", "multivariate_kd", "univariate_kd", "cvrd")
            ]
            + [torch.tensor(valid_count, dtype=torch.float64, device=device)]
        )
        realized_valid_windows += valid_count
    if realized_valid_windows != logical_valid_windows:
        raise RuntimeError("S5 logical valid-window count mismatch")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        training_model.parameters(),
        float(training["gradient_clip"]),
        error_if_nonfinite=True,
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite S5 replay gradient norm")
    optimizer.step()
    progress = (step + 1) / int(training["max_steps"])
    learning_rate = float(training["min_learning_rate"]) + 0.5 * (
        float(training["learning_rate"]) - float(training["min_learning_rate"])
    ) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in training_model.parameters()
        if parameter.grad is not None
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in training_model.parameters()
    )
    if not gradients_finite or not parameters_finite:
        raise FloatingPointError("non-finite S5 replay gradients or parameters")
    return {
        "windows": sum(len(indices) for _, indices in optimizer_batch),
        "microbatches": len(optimizer_batch),
        "datasets": datasets,
        "valid_windows": logical_valid_windows,
        "weighted_loss": math.fsum(losses),
        "gradient_norm_before_clip": float(gradient_norm.detach()),
        "gradient_clipped": float(gradient_norm.detach()) > float(training["gradient_clip"]),
        "gradients_finite": gradients_finite,
        "parameters_finite_after_step": parameters_finite,
        "end_to_end_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--manager-launch-record", type=Path)
    parser.add_argument("--expected-launch-sha256")
    parser.add_argument("--manager-ledger", type=Path)
    parser.add_argument("--manager-child-record", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify all tracked authorities without loading production data or CUDA",
    )
    args = parser.parse_args()
    imported_module_origins = _assert_import_origins()
    config = load_config(args.config)
    candidate, plan, selection = _validate(config)
    probe = config["probe"]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "candidate_config": str(config["candidate_config"]),
                    "variant": probe["variant"],
                    "loss_reduction": candidate["training"]["loss_reduction"],
                    "logical_batch_size_windows": probe["logical_batch_size_windows"],
                    "order": "exact frozen epoch-zero _epoch_batches + _pack_logical_batches",
                    "datasets": len(plan["datasets"]),
                    "selection_status": selection["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if (
        args.output is None
        or args.physical_gpu_index is None
        or args.manager_launch_record is None
        or args.expected_launch_sha256 is None
        or args.manager_ledger is None
        or args.manager_child_record is None
    ):
        raise ValueError(
            "GPU autotune is manager-only and requires immutable launch, ledger, output, and GPU"
        )

    source_paths = _relevant_source_paths(args.config, config)
    _require_relevant_tree_clean(source_paths)

    physical_gpu = int(args.physical_gpu_index)
    device = torch.device(str(probe["device"]))
    output = args.output.resolve()
    launch, launch_sha256, child_binding = _validate_manager_authorization(
        launch_path=args.manager_launch_record.resolve(),
        expected_launch_sha256=str(args.expected_launch_sha256),
        ledger_path=args.manager_ledger.resolve(),
        output=output,
        physical_gpu=physical_gpu,
        child_record_path=args.manager_child_record.resolve(),
    )
    requested_physical_gpu_uuid = _resolve_physical_gpu(physical_gpu, device)
    if _normalize_gpu_uuid(requested_physical_gpu_uuid) != _normalize_gpu_uuid(
        str(launch.get("physical_gpu_uuid", ""))
    ):
        raise ValueError("manager-bound physical GPU UUID changed before autotune")
    occupied = _gpu_processes(physical_gpu)
    if occupied:
        raise RuntimeError(f"physical GPU {physical_gpu} is occupied: {occupied}")
    output_template = str(config["output_template"])
    expected_output = _path(output_template.format(attempt=int(launch["attempt"])))
    if output != expected_output:
        raise ValueError("S5 throughput output differs from its unique frozen attempt template")
    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    try:
        physical_gpu_started = time.perf_counter()
        torch.cuda.set_device(device)
        runtime_physical_gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
        if _normalize_gpu_uuid(runtime_physical_gpu_uuid) != _normalize_gpu_uuid(
            requested_physical_gpu_uuid
        ):
            raise RuntimeError(
                "PyTorch logical CUDA device does not map to requested physical GPU: "
                f"requested={requested_physical_gpu_uuid}, runtime={runtime_physical_gpu_uuid}"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected GPU does not support BF16")
        load_started = time.perf_counter()
        corpora = _load_corpora(candidate, plan, selection, config)
        load_seconds = time.perf_counter() - load_started
        physical_batches = production._epoch_batches(corpora, int(config["seed"]), 0)
        logical_batches = production._pack_logical_batches(
            physical_batches, int(probe["logical_batch_size_windows"])
        )
        count = int(probe["warmup_optimizer_steps"]) + int(probe["measured_optimizer_steps"])
        if len(logical_batches) < count:
            raise ValueError("epoch zero has too few logical batches for the S5 probe")
        selected_batches = logical_batches[:count]
        replayed_windows = sum(
            len(indices) for optimizer_batch in selected_batches for _, indices in optimizer_batch
        )
        if replayed_windows != int(probe["expected_replayed_windows"]):
            raise ValueError("S5 replay window count differs from the frozen prefix")
        sequence = bytes(32)
        for optimizer_batch in selected_batches:
            sequence = _sequence_update(sequence, optimizer_batch, corpora)
        if sequence.hex() != str(probe["expected_replayed_training_sequence_sha256"]):
            raise ValueError("S5 replay order differs from the frozen epoch-zero prefix")

        torch.manual_seed(int(config["seed"]))
        student = build_student(candidate["student"])
        initialization_sha256 = _state_sha256(student)
        if initialization_sha256 != str(probe["initialization_state_sha256"]):
            raise ValueError("S5 replay initialization differs from the selected base")
        student.to(device)
        training_model = production._StudentViews(student).to(device).train()
        objective = DistillationLoss(
            LossWeights.from_mapping(candidate["training"]["loss_weights"][probe["variant"]])
        )
        optimizer = torch.optim.AdamW(
            training_model.parameters(),
            lr=float(candidate["training"]["learning_rate"]),
            weight_decay=float(candidate["training"]["weight_decay"]),
            fused=True,
        )
        measurements = []
        warmup = int(probe["warmup_optimizer_steps"])
        for index, optimizer_batch in enumerate(selected_batches):
            item = _logical_step(
                training_model=training_model,
                objective=objective,
                optimizer=optimizer,
                optimizer_batch=optimizer_batch,
                corpora=corpora,
                training=candidate["training"],
                variant=str(probe["variant"]),
                device=device,
                step=index,
            )
            if index >= warmup:
                measurements.append(item)
        elapsed = math.fsum(float(item["end_to_end_seconds"]) for item in measurements)
        windows = sum(int(item["windows"]) for item in measurements)
        if windows != int(probe["expected_measured_windows"]):
            raise ValueError("S5 measured window count differs from the frozen probe")
        aggregate = windows / elapsed
        p95_seconds = float(
            np.percentile([float(item["end_to_end_seconds"]) for item in measurements], 95)
        )
        conservative = min(
            aggregate,
            min(int(item["windows"]) for item in measurements) / p95_seconds,
        )
        variant = str(probe["variant"])
        metrics = {
            f"{variant}_conservative_windows_per_second": conservative,
            f"{variant}_aggregate_windows_per_second": aggregate,
        }
        torch.cuda.synchronize(device)
        physical_gpu_elapsed_seconds = time.perf_counter() - physical_gpu_started
        relevant_source_sha256 = {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        }
        record.extra.update(
            {
                "schema_version": 1,
                "manager_job_id": launch["job_id"],
                "manager_attempt": int(launch["attempt"]),
                "manager_launch_record": {
                    "path": str(args.manager_launch_record.resolve().relative_to(ROOT)),
                    "sha256": launch_sha256,
                },
                "manager_child_record": child_binding,
                "manager_execution_deadline_at": launch["execution_deadline_at"],
                "manager_cleanup_budget_seconds": launch["cleanup_budget_seconds"],
                "imported_module_origins": imported_module_origins,
                "candidate_config": str(config["candidate_config"]),
                "candidate_config_sha256": _sha256(_path(config["candidate_config"])),
                "production_plan": str(config["production_plan"]),
                "production_plan_sha256": _sha256(_path(config["production_plan"])),
                "selection_split_manifest_sha256": _sha256(
                    _path(config["selection_split_manifest"])
                ),
                "activation_evidence_sha256": _sha256(_path(config["activation_evidence"])),
                "training_implementation_sha256": _sha256(
                    _path(config["authority"]["training_implementation"])
                ),
                "loss_implementation_sha256": _sha256(
                    _path(config["authority"]["loss_implementation"])
                ),
                "manager_implementation_sha256": _sha256(
                    _path(config["authority"]["manager_implementation"])
                ),
                "autotune_implementation_sha256": _sha256(Path(__file__).resolve()),
                "autotune_config_sha256": _sha256(args.config.resolve()),
                "relevant_source_sha256": relevant_source_sha256,
                "output_path": str(output.relative_to(ROOT)),
                "physical_gpu_index": physical_gpu,
                "requested_physical_gpu_uuid": requested_physical_gpu_uuid,
                "runtime_physical_gpu_uuid": runtime_physical_gpu_uuid,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_cuda_device": str(device),
                "physical_gpu_count": 1,
                "physical_gpu_elapsed_seconds": physical_gpu_elapsed_seconds,
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                },
                "corpus_load_seconds": load_seconds,
                "training_semantics": {
                    "precision": "BF16 autocast with FP32 parameters and loss reductions",
                    "optimizer": "fused AdamW",
                    "loss_reduction": (
                        "per-window loss; frozen domain weight / unweighted valid windows"
                    ),
                    "zero_target_window_policy": (
                        "sequence retained; excluded from loss and denominator"
                    ),
                    "order": (
                        "exact production _epoch_batches(seed=42, epoch=0) followed by "
                        "_pack_logical_batches(..., 256)"
                    ),
                    "physical_batch_map": candidate["training"]["batch_size_by_context"],
                    "timed_unit": (
                        "CPU target-validity scan + source/cache materialization + pinned H2D + "
                        "forward + exact per-window weighted reducer + backward + clip + fused "
                        "optimizer + schedule + CUDA synchronization"
                    ),
                },
                "initialization_state_sha256": initialization_sha256,
                "replayed_training_sequence_sha256": sequence.hex(),
                "warmup_optimizer_steps": warmup,
                "measured_optimizer_steps": len(measurements),
                "replayed_optimizer_steps": len(selected_batches),
                "replayed_windows": replayed_windows,
                "measured_windows": windows,
                "measurements": measurements,
                "end_to_end_seconds": _summary(
                    [float(item["end_to_end_seconds"]) for item in measurements]
                ),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "variant_summaries": {
                    variant: {
                        "conservative_selected_windows_per_second": conservative,
                        "aggregate_windows_per_second": aggregate,
                        "stable_manage_json_pointer": (
                            f"/metrics/{variant}_conservative_windows_per_second"
                        ),
                        "all_gradients_finite": all(
                            bool(item["gradients_finite"]) for item in measurements
                        ),
                        "all_parameters_finite": all(
                            bool(item["parameters_finite_after_step"]) for item in measurements
                        ),
                    }
                },
            }
        )
        record.succeed(metrics)
    except BaseException as error:
        record.fail(f"{type(error).__name__}: {error}")
        _write_json_new(output, record.to_dict())
        raise
    _write_json_new(output, record.to_dict())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
