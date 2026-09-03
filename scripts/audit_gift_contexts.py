#!/usr/bin/env python3
"""Audit GIFT context/horizon requirements and run one teacher context study."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME, TimesFm3GiftPredictor, _metric_scalar
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _FixedEndpointTestData:
    inputs: list[Mapping[str, Any]]
    labels: list[Mapping[str, Any]]

    def __len__(self) -> int:
        return len(self.inputs)

    def __iter__(self) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        return iter(zip(self.inputs, self.labels, strict=True))

    @property
    def input(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.inputs)

    @property
    def label(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.labels)


def _truncate_at_fixed_endpoint(entry: Mapping[str, Any], context: int) -> dict[str, Any]:
    result = dict(entry)
    target = np.asarray(entry["target"], dtype=np.float32)
    original_length = int(target.shape[-1])
    if original_length < context:
        raise ValueError(f"context {context} exceeds input length {original_length}")
    result["target"] = target[..., -context:]
    result["start"] = entry["start"] + (original_length - context)
    if entry.get("past_feat_dynamic_real") is not None:
        covariates = np.asarray(entry["past_feat_dynamic_real"], dtype=np.float32)
        result["past_feat_dynamic_real"] = covariates[..., -context:]
    return result


def _distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(array, return_counts=True)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p25": int(np.percentile(array, 25)),
        "p50": int(np.percentile(array, 50)),
        "p75": int(np.percentile(array, 75)),
        "max": int(array.max()),
        "unique_counts": {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())

    import torch
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    audit_results: list[dict[str, Any]] = []
    all_contexts: list[int] = []
    all_horizons: list[int] = []
    for item in config["scope"]["datasets"]:
        dataset = Dataset(name=str(item["name"]), term=str(item["term"]), to_univariate=False)
        context_lengths = [
            int(np.asarray(entry["target"]).shape[-1]) for entry in dataset.test_data.input
        ]
        all_contexts.extend(context_lengths)
        all_horizons.extend([int(dataset.prediction_length)] * len(context_lengths))
        audit_results.append(
            {
                "configuration": f"{item['name']}/{item['term']}",
                "instances": len(context_lengths),
                "target_variates": int(dataset.target_dim),
                "horizon": int(dataset.prediction_length),
                "context": _distribution(context_lengths),
                "instances_with_at_least": {
                    str(length): sum(value >= length for value in context_lengths)
                    for length in config["study"]["contexts"]
                },
            }
        )
    print(f"audited {len(audit_results)} configurations", flush=True)

    model_cfg = config["model"]
    teacher = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(model_cfg["id"]),
            revision=str(config["model_revision"]),
            device=str(model_cfg["device"]),
            per_core_batch_size=int(model_cfg["batch_size"]),
        )
    )
    contexts = [int(value) for value in config["study"]["contexts"]]
    study_results: list[dict[str, Any]] = []
    quantiles = [float(value) for value in config["study"]["quantiles"]]
    metrics = [
        MASE(),
        MeanWeightedSumQuantileLoss(quantile_levels=quantiles),
    ]
    for task in config["study"]["tasks"]:
        dataset = Dataset(name=str(task["name"]), term=str(task["term"]), to_univariate=False)
        pairs = list(dataset.test_data)
        required_context = max(contexts)
        eligible = [
            (input_entry, label_entry)
            for input_entry, label_entry in pairs
            if np.asarray(input_entry["target"]).shape[-1] >= required_context
        ]
        limit = min(len(eligible), int(task["max_instances"]))
        selected = eligible[-limit:]
        if not selected:
            raise ValueError(f"{task['name']} has no instances supporting {required_context}")
        for context in contexts:
            test_data = _FixedEndpointTestData(
                inputs=[_truncate_at_fixed_endpoint(entry, context) for entry, _ in selected],
                labels=[label for _, label in selected],
            )
            predictor = TimesFm3GiftPredictor(
                teacher,
                prediction_length=dataset.prediction_length,
                batch_size=int(model_cfg["batch_size"]),
                quantiles=quantiles,
                univariate=False,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            frame = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=int(model_cfg["batch_size"]),
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),
            )
            torch.cuda.synchronize()
            result = {
                "configuration": f"{task['name']}/{task['term']}",
                "context": context,
                "instances": len(test_data),
                "horizon": int(dataset.prediction_length),
                "target_variates": int(dataset.target_dim),
                MASE_NAME: _metric_scalar(frame, MASE_NAME),
                MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
                "wall_seconds": time.perf_counter() - started,
            }
            study_results.append(result)
            print(result, flush=True)

    student = load_config(Path(str(config["student_config"])))
    record.extra.update(
        {
            "scope": str(config["scope"]["name"]),
            "scope_audit": audit_results,
            "aggregate_context_distribution": _distribution(all_contexts),
            "aggregate_horizon_distribution": _distribution(all_horizons),
            "support": {
                "teacher": {
                    "max_context": int(config["support"]["teacher_max_context"]),
                    "context_evidence": str(config["support"]["teacher_context_evidence"]),
                    "hard_max_horizon": None,
                    "horizon_semantics": (
                        "positive horizons are rounded to 64-point output patches and decoded "
                        "autoregressively; the public API declares no hard horizon cap"
                    ),
                },
                "student": {
                    "max_context": int(student["student"]["max_context"]),
                    "max_horizon": int(student["student"]["max_horizon"]),
                    "patch_length": int(student["student"]["patch_length"]),
                },
            },
            "context_study": {
                "method": (
                    "official GIFT metrics on the same test endpoints and labels; only the "
                    "amount of visible history changes"
                ),
                "results": study_results,
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
    )
    record.succeed(
        {
            "scope/configurations": float(len(audit_results)),
            "scope/instances": float(len(all_contexts)),
            "scope/max_horizon": float(max(all_horizons)),
            "scope/max_available_context": float(max(all_contexts)),
        }
    )
    output = args.output or Path(str(config["output"]))
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
