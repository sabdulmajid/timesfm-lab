#!/usr/bin/env python3
"""Evaluate all bound recovery checkpoints on development data only.

This intentionally has no confirmation-partition option.  It reuses the exact
corpus materialization and validation implementation used by production
training so incumbent and candidate metrics are directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
DEFAULT_REGISTRY = ROOT / "configs/performance_recovery/candidates.yaml"
DEFAULT_SELECTION_CONFIG = ROOT / "configs/performance_recovery/screen_selection.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_tree(root: Path) -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in sorted(root.rglob("*.py"))
    ]
    return {"files": files, "sha256": _canonical_sha256(files)}


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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("_recovery_validation_authority", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation authority {TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite_positive_validation(validation: dict[str, Any]) -> None:
    balanced = validation.get("balanced", {})
    required = (
        "student_normalized_median_mae",
        "student_pinball",
        "true_mv_student_normalized_median_mae",
        "true_mv_student_pinball",
        "forecast_error",
    )
    for name in required:
        value = float(balanced.get(name, float("nan")))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"development validation has invalid {name}: {value}")


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--screen-selection-config", type=Path, default=DEFAULT_SELECTION_CONFIG
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite development evidence: {args.output}")
    if args.registry.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("incumbent evaluation requires the canonical candidate registry")
    if args.screen_selection_config.resolve() != DEFAULT_SELECTION_CONFIG.resolve():
        raise ValueError("incumbent evaluation requires the canonical selection authority")
    # This process must never discover or initialize GIFT-Eval.
    os.environ.pop("GIFT_EVAL", None)

    registry = _load_yaml(args.registry)
    selection_config = _load_yaml(args.screen_selection_config)
    request = json.loads(args.request.read_text())
    protocol_id = str(registry["protocol"]["id"])
    if selection_config.get("protocol_id") != protocol_id:
        raise ValueError("screen-selection authority protocol mismatch")
    if request.get("schema_version") != 1:
        raise ValueError("unsupported common-development request schema")
    if request.get("status") != "awaiting_common_development_evaluation":
        raise ValueError("input is not a common-development evaluation request")
    if request.get("protocol_id") != protocol_id:
        raise ValueError("common-development request protocol mismatch")
    if request.get("partition") != "development":
        raise ValueError("common-development request is not development-only")
    if request.get("confirmation_partition_accessed") is not False:
        raise ValueError("common-development request reports confirmation access")
    if request.get("gift_eval_data_accessed") is not False:
        raise ValueError("common-development request reports GIFT-Eval access")

    authority = request["data_authority"]
    plan_path = ROOT / authority["plan"]["path"]
    cache_audit = ROOT / authority["cache_audit"]["path"]
    selection_path = ROOT / authority["selection_split_manifest"]["path"]
    expected_roots = {
        "data root": ROOT / authority["data_root"],
        "cache root": ROOT / authority["cache_root"],
    }
    supplied_roots = {"data root": args.data_root, "cache root": args.cache_root}
    for label, expected_path in expected_roots.items():
        if supplied_roots[label].resolve() != expected_path.resolve():
            raise ValueError(f"{label} is not bound by the evaluation request")
    corpus = registry["corpus"]
    expected_authority = {
        "dataset_revision": corpus["dataset_revision"],
        "plan": {"path": corpus["plan"], "sha256": corpus["plan_sha256"]},
        "cache_audit": {
            "path": corpus["cache_audit"],
            "sha256": corpus["cache_audit_sha256"],
        },
        "data_root": corpus["data_root"],
        "cache_root": corpus["cache_root"],
        "selection_split_manifest": {
            "path": selection_config["selection_split_manifest"],
            "sha256": selection_config["selection_split_manifest_sha256"],
            "protocol_id": selection_config["target_authority"]["protocol_id"],
        },
    }
    if authority != expected_authority:
        raise ValueError("evaluation request data authority differs from registry")
    for path, expected_hash, label in (
        (plan_path, authority["plan"]["sha256"], "corpus plan"),
        (cache_audit, authority["cache_audit"]["sha256"], "cache audit"),
        (
            selection_path,
            authority["selection_split_manifest"]["sha256"],
            "selection split",
        ),
    ):
        if _sha256(path) != expected_hash:
            raise ValueError(f"{label} hash mismatch")
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get("protocol_id") != authority["selection_split_manifest"][
        "protocol_id"
    ]:
        raise ValueError("selection manifest protocol mismatch")
    if selection.get("status") != "frozen_uninspected":
        raise ValueError("selection manifest is no longer frozen and uninspected")
    if selection.get("target_accessed") is not False:
        raise ValueError("selection manifest reports target access")
    if selection.get("teacher_output_accessed") is not False:
        raise ValueError("selection manifest reports teacher-output access")
    if selection["source"]["plan_sha256"] != _sha256(plan_path):
        raise ValueError("selection manifest corpus-plan hash mismatch")
    evaluation_authority = request["evaluation_authority"]
    implementation_path = ROOT / evaluation_authority["validation_implementation"]["path"]
    if implementation_path.resolve() != TRAINER.resolve():
        raise ValueError("request uses a different validation implementation")
    if _sha256(TRAINER) != evaluation_authority["validation_implementation"]["sha256"]:
        raise ValueError("validation implementation changed after request freeze")
    runner_path = ROOT / evaluation_authority["evaluation_runner"]["path"]
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError("request uses a different common-development runner")
    if _sha256(runner_path) != evaluation_authority["evaluation_runner"]["sha256"]:
        raise ValueError("common-development runner changed after request freeze")
    model_source = _source_tree(ROOT / "src/timesfm_lab")
    if model_source != evaluation_authority["model_source"]:
        raise ValueError("model source changed after request freeze")
    if evaluation_authority["validation_partition"] != "development":
        raise ValueError("evaluation authority is not development-only")
    trainer = _load_trainer()
    entries = {str(item["dataset"]): item for item in selection["datasets"]}
    corpora = [
        trainer._load_corpus(
            item,
            data_root=args.data_root,
            cache_root=args.cache_root,
            validation_fraction=float(selection["outer_split"]["validation_fraction"]),
            validation_mode=str(selection["outer_split"]["mode"]),
            seed=int(selection["outer_split"]["seed"]),
            batch_sizes=evaluation_authority["batch_size_by_context"],
            selection_manifest=selection,
            selection_entry=entries[str(item["dataset"])],
            validation_partition="development",
        )
        for item in plan["datasets"]
    ]
    if sum(len(corpus.validation_indices) for corpus in corpora) != int(
        selection["totals"]["development"]["count"]
    ):
        raise ValueError("materialized development count differs from frozen manifest")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_results = {}
    for model_id, binding in request["model_bindings"].items():
        config_path = ROOT / binding["config"]["path"]
        checkpoint_path = ROOT / binding["checkpoint"]["path"]
        if _sha256(config_path) != binding["config"]["sha256"]:
            raise ValueError(f"{model_id} config changed after request freeze")
        if _sha256(checkpoint_path) != binding["checkpoint"]["sha256"]:
            raise ValueError(f"{model_id} checkpoint changed after request freeze")
        model_config = _load_yaml(config_path)
        if str(model_config["dataset_revision"]) != str(plan["dataset_revision"]):
            raise ValueError(f"{model_id} config/plan dataset revision mismatch")
        if _canonical_sha256(model_config.get("inference", {})) != evaluation_authority[
            "inference_policy_sha256"
        ]:
            raise ValueError(f"{model_id} inference policy differs from common authority")
        model = build_student(model_config["student"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model_state_sha256 = _state_sha256(model)
        model.to(device)
        validation = trainer._validate(
            model,
            corpora,
            device,
            str(evaluation_authority["input_preprocessing"]),
            dict(model_config.get("inference", {})),
        )
        _finite_positive_validation(validation)
        model_results[model_id] = {
            "checkpoint": binding["checkpoint"],
            "config": binding["config"],
            "loaded_state_sha256": model_state_sha256,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "deployment_fingerprint_sha256": _canonical_sha256(
                {
                    "student": model_config["student"],
                    "inference": model_config.get("inference", {}),
                }
            ),
            "validation": validation,
        }
        model.cpu()
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started

    payload = {
        "schema_version": 2,
        "status": "succeeded",
        "protocol_id": protocol_id,
        "evaluated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "partition": "development",
        "confirmation_partition_accessed": False,
        "gift_eval_data_accessed": False,
        "development_targets_accessed": True,
        "development_teacher_outputs_accessed": True,
        "measurement": {
            "physical_gpu_count": 1,
            "elapsed_seconds": elapsed,
            "physical_gpu_hours": elapsed / 3600.0,
        },
        "evaluation_request": {
            "path": str(args.request.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.request),
        },
        "data_authority": authority,
        "evaluation_authority": evaluation_authority,
        "development_identity_sha256": selection["totals"]["development"][
            "identity_sha256"
        ],
        "models": model_results,
    }
    _atomic_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
