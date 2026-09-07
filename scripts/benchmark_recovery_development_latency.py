#!/usr/bin/env python3
"""Measure screen-selection latency on frozen pretrain development windows.

Inputs come only from the GiftEvalPretrain development partition.  The script
accepts the hash-bound request produced by ``finalize_recovery_screens.py
prepare`` and evaluates every bound checkpoint serially on one idle GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from timesfm_lab.models import build_student

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_production_student.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_tree(root: Path) -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in sorted(root.rglob("*.py"))
    ]
    return {"files": files, "sha256": _canonical_sha256(files)}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite latency evidence: {path}")
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


def _load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("_screen_latency_validation", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_processes(physical_gpu: int) -> list[int]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    mapping = {
        int(index.strip()): uuid.strip()
        for index, uuid in (row.split(",", 1) for row in gpu_rows)
    }
    if physical_gpu not in mapping:
        raise ValueError(f"physical GPU {physical_gpu} is unavailable")
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return [
        int(pid.strip())
        for uuid, pid in (row.split(",", 1) for row in rows if "," in row)
        if uuid.strip() == mapping[physical_gpu]
    ]


def _input_sha256(contexts: list[np.ndarray], identities: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for (row, end), context in zip(identities, contexts, strict=True):
        digest.update(f"{row}:{end}".encode())
        contiguous = np.ascontiguousarray(context)
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _prepare_workloads(
    workload_config: dict[str, Any],
    request: dict[str, Any],
    plan_path: Path,
    data_root: Path,
    cache_root: Path,
    selection_path: Path,
    trainer: Any,
) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get("protocol_id") != request["data_authority"][
        "selection_split_manifest"
    ]["protocol_id"]:
        raise ValueError("selection manifest protocol mismatch")
    if selection.get("status") != "frozen_uninspected":
        raise ValueError("confirmation split is no longer frozen and uninspected")
    if selection.get("target_accessed") is not False:
        raise ValueError("selection manifest reports target access")
    entries = {str(row["dataset"]): row for row in selection["datasets"]}
    plan_items = {str(row["dataset"]): row for row in plan["datasets"]}
    first_binding = next(iter(request["model_bindings"].values()))
    model_config = _load_yaml(ROOT / first_binding["config"]["path"])
    training = model_config["training"]
    output = []
    for spec in workload_config["workloads"]:
        dataset = str(spec["dataset"])
        if dataset not in plan_items:
            raise ValueError(f"latency workload dataset is absent from the corpus: {dataset}")
        corpus = trainer._load_corpus(
            plan_items[dataset],
            data_root=data_root,
            cache_root=cache_root,
            validation_fraction=float(training["validation_fraction"]),
            validation_mode=str(training["validation_split"]),
            seed=int(selection["outer_split"]["seed"]),
            batch_sizes=training["batch_size_by_context"],
            selection_manifest=selection,
            selection_entry=entries[dataset],
            validation_partition="development",
        )
        batch = int(spec["batch"])
        indices = np.asarray(corpus.validation_indices[:batch], dtype=np.int64)
        if len(indices) != batch:
            raise ValueError(f"{dataset} has fewer than {batch} development windows")
        contexts = []
        identities = []
        for index in indices:
            row = int(corpus.row_index[index])
            end = int(corpus.context_end[index])
            contexts.append(
                np.asarray(
                    corpus.source[row][:, end - corpus.context_length : end],
                    dtype=np.float32,
                )
            )
            identities.append((row, end))
        output.append(
            {
                "name": str(spec["name"]),
                "dataset": dataset,
                "batch": batch,
                "variates": int(contexts[0].shape[0]),
                "context": int(corpus.context_length),
                "horizon": int(corpus.horizon),
                "input_shape": [
                    batch,
                    int(contexts[0].shape[0]),
                    int(corpus.context_length),
                ],
                "output_shape": [
                    batch,
                    int(contexts[0].shape[0]),
                    int(corpus.horizon),
                    9,
                ],
                "cache_indices": indices.tolist(),
                "identities": [
                    {"row_index": row, "context_end": end} for row, end in identities
                ],
                "identity_sha256": _canonical_sha256(
                    [
                        {"row_index": row, "context_end": end}
                        for row, end in identities
                    ]
                ),
                "input_sha256": _input_sha256(contexts, identities),
                "raw_contexts": contexts,
            }
        )
        output[-1]["shape_sha256"] = _canonical_sha256(
            {
                "input_shape": output[-1]["input_shape"],
                "output_shape": output[-1]["output_shape"],
            }
        )
        del corpus
    return output


def _measure_model(
    model_id: str,
    binding: dict[str, Any],
    workloads: list[dict[str, Any]],
    workload_config: dict[str, Any],
    trainer: Any,
    device: torch.device,
) -> dict[str, Any]:
    config_path = ROOT / binding["config"]["path"]
    checkpoint_path = ROOT / binding["checkpoint"]["path"]
    if _sha256(config_path) != binding["config"]["sha256"]:
        raise ValueError(f"{model_id} config changed after latency request")
    if _sha256(checkpoint_path) != binding["checkpoint"]["sha256"]:
        raise ValueError(f"{model_id} checkpoint changed after latency request")
    config = _load_yaml(config_path)
    deployment_fingerprint = _canonical_sha256(
        {"student": config["student"], "inference": config.get("inference", {})}
    )
    if deployment_fingerprint != binding["deployment_fingerprint_sha256"]:
        raise ValueError(f"{model_id} deployment fingerprint changed")
    if _source_tree(ROOT / "src/timesfm_lab") != binding["model_source"]:
        raise ValueError(f"{model_id} model source changed after latency request")
    inference_path = ROOT / binding["inference_implementation"]["path"]
    if inference_path.resolve() != TRAINER.resolve():
        raise ValueError(f"{model_id} requested a different inference implementation")
    if _sha256(inference_path) != binding["inference_implementation"]["sha256"]:
        raise ValueError(f"{model_id} inference implementation changed")
    model = build_student(config["student"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    loaded_state_sha256 = _model_state_sha256(model)
    model.to(device).eval()
    inference = dict(config.get("inference", {}))

    def call(raw_contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        prepared = np.stack(
            [trainer._timesfm_interpolate_context(value) for value in raw_contexts]
        )
        context = torch.from_numpy(np.ascontiguousarray(prepared)).pin_memory().to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = trainer._deployment_forecast(model, context, horizon, inference)
        return prediction.float().cpu().numpy()

    results = {}
    warmups = int(workload_config["warmup_calls_per_workload"])
    repetitions = int(workload_config["measured_calls_per_workload"])
    for workload in workloads:
        raw_contexts = workload["raw_contexts"]
        horizon = int(workload["horizon"])
        for _ in range(warmups):
            output = call(raw_contexts, horizon)
        if not np.isfinite(output).all():
            raise FloatingPointError(f"{model_id}.{workload['name']} produced nonfinite output")
        expected_shape = tuple(workload["output_shape"])
        if output.shape != expected_shape:
            raise ValueError(
                f"{model_id}.{workload['name']} output {output.shape} != {expected_shape}"
            )
        samples = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = call(raw_contexts, horizon)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        if not np.isfinite(output).all():
            raise FloatingPointError(f"{model_id}.{workload['name']} produced nonfinite output")
        results[workload["name"]] = {
            "p50_end_to_end_latency_ms": statistics.median(samples),
            "p95_end_to_end_latency_ms": float(np.percentile(samples, 95)),
            "end_to_end_latency_ms_samples": samples,
            "output_shape": list(output.shape),
        }
    aggregate = math.exp(
        math.fsum(math.log(row["p50_end_to_end_latency_ms"]) for row in results.values())
        / len(results)
    )
    record = {
        "checkpoint": binding["checkpoint"],
        "config": binding["config"],
        "deployment_fingerprint_sha256": deployment_fingerprint,
        "model_source": binding["model_source"],
        "inference_implementation": binding["inference_implementation"],
        "loaded_state_sha256": loaded_state_sha256,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "aggregate_end_to_end_latency_ms": aggregate,
        "workloads": results,
    }
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--selection-split-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    if request.get("status") != "awaiting_development_latency_evidence":
        raise ValueError("input is not a prepared screen-latency request")
    if request.get("confirmation_partition_accessed") is not False:
        raise ValueError("latency request reports confirmation access")
    if request.get("gift_eval_data_accessed") is not False:
        raise ValueError("latency request reports GIFT-Eval access")
    implementation = request.get("latency_implementation", {})
    implementation_path = ROOT / implementation.get("path", "")
    if implementation_path.resolve() != Path(__file__).resolve():
        raise ValueError("latency request uses a different benchmark implementation")
    if _sha256(implementation_path) != implementation.get("sha256"):
        raise ValueError("latency benchmark implementation changed after request freeze")
    authority = request.get("data_authority", {})
    expected_paths = {
        "plan": ROOT / authority["plan"]["path"],
        "data root": ROOT / authority["data_root"],
        "cache root": ROOT / authority["cache_root"],
        "selection split": ROOT / authority["selection_split_manifest"]["path"],
    }
    supplied_paths = {
        "plan": args.plan,
        "data root": args.data_root,
        "cache root": args.cache_root,
        "selection split": args.selection_split_manifest,
    }
    for label, expected_path in expected_paths.items():
        if supplied_paths[label].resolve() != expected_path.resolve():
            raise ValueError(f"{label} is not bound by the latency request")
    if _sha256(args.plan) != authority["plan"]["sha256"]:
        raise ValueError("latency-request corpus plan hash mismatch")
    cache_audit_path = ROOT / authority["cache_audit"]["path"]
    if _sha256(cache_audit_path) != authority["cache_audit"]["sha256"]:
        raise ValueError("latency-request cache audit hash mismatch")
    if _sha256(args.selection_split_manifest) != authority["selection_split_manifest"][
        "sha256"
    ]:
        raise ValueError("latency-request selection split hash mismatch")
    workload_path = ROOT / request["workload_manifest"]["path"]
    if _sha256(workload_path) != request["workload_manifest"]["sha256"]:
        raise ValueError("frozen development-latency workload changed")
    workload_config = _load_yaml(workload_path)
    if workload_config.get("protocol_id") != request.get("protocol_id"):
        raise ValueError("workload/request protocol mismatch")
    if workload_config.get("source_partition") != "development":
        raise ValueError("latency workload is not development-only")
    if os.environ.get("GIFT_EVAL"):
        raise ValueError("GIFT_EVAL must be unset for development-only latency")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise ValueError(
            f"set CUDA_VISIBLE_DEVICES={args.physical_gpu} exactly; observed {visible!r}"
        )
    occupied = _gpu_processes(args.physical_gpu)
    if occupied:
        raise RuntimeError(f"physical GPU {args.physical_gpu} is occupied by PIDs {occupied}")

    started = time.perf_counter()
    trainer = _load_trainer()
    workloads = _prepare_workloads(
        workload_config,
        request,
        args.plan,
        args.data_root,
        args.cache_root,
        args.selection_split_manifest,
        trainer,
    )
    binding_fields = tuple(request["workloads"][0])
    if len(workloads) != len(request["workloads"]):
        raise ValueError("runtime workload count differs from latency request")
    for expected, observed in zip(request["workloads"], workloads, strict=True):
        if {name: observed.get(name) for name in binding_fields} != expected:
            raise ValueError(f"runtime workload binding mismatch for {expected['name']}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_results = {}
    for model_id, binding in request["model_bindings"].items():
        foreign = [pid for pid in _gpu_processes(args.physical_gpu) if pid != os.getpid()]
        if foreign:
            raise RuntimeError(
                f"physical GPU {args.physical_gpu} became contended by PIDs {foreign}"
            )
        model_results[model_id] = _measure_model(
            model_id, binding, workloads, workload_config, trainer, device
        )
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "protocol_id": request["protocol_id"],
        "measured_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "partition": "development",
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "latency_implementation": implementation,
        "data_authority": authority,
        "workload_manifest": request["workload_manifest"],
        "latency_request": {
            "path": str(args.request.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.request),
        },
        "measurement": {
            "warmup_calls_per_workload": int(workload_config["warmup_calls_per_workload"]),
            "measured_calls_per_workload": int(workload_config["measured_calls_per_workload"]),
            "synchronize_each_sample": True,
            "single_gpu_no_contention": True,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": elapsed,
            "physical_gpu_hours": elapsed / 3600.0,
        },
        "inputs": [
            {key: value for key, value in row.items() if key != "raw_contexts"}
            for row in workloads
        ],
        "models": model_results,
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
