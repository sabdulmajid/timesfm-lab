#!/usr/bin/env python3
"""Measure S5 with its exact production logical-batch and reducer path.

This is deliberately not a synthetic tensor or homogeneous-shape benchmark.
It reconstructs the frozen epoch-zero production order, packs the selected
base's physical microbatches into 256-window optimizer batches, and executes
the per-window, domain-weighted reducer used by the production trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import train_production_student as production

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


def _gpu_processes(physical_gpu: int) -> list[dict[str, Any]]:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    mapping = {
        int(index.strip()): uuid.strip() for index, uuid in (row.split(",", 1) for row in rows)
    }
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


def _validate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority = config["authority"]
    paths = {
        "candidate_config": _path(config["candidate_config"]),
        "production_plan": _path(config["production_plan"]),
        "selection_split_manifest": _path(config["selection_split_manifest"]),
        "activation_evidence": _path(config["activation_evidence"]),
        "training_implementation": _path(authority["training_implementation"]),
        "loss_implementation": _path(authority["loss_implementation"]),
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify all tracked authorities without loading production data or CUDA",
    )
    args = parser.parse_args()
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

    physical_gpu = (
        int(args.physical_gpu_index)
        if args.physical_gpu_index is not None
        else int(probe["physical_gpu_index"])
    )
    occupied = _gpu_processes(physical_gpu)
    if occupied:
        raise RuntimeError(f"physical GPU {physical_gpu} is occupied: {occupied}")
    output = args.output.resolve() if args.output else _path(config["output"])
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
        device = torch.device(str(probe["device"]))
        torch.cuda.set_device(device)
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
        record.extra.update(
            {
                "schema_version": 1,
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
                "physical_gpu_index": physical_gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        record.write(output)
        raise
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
