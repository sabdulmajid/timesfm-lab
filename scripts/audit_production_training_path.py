#!/usr/bin/env python3
"""Audit the frozen production student path without using a GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from timesfm_lab.config import load_config
from timesfm_lab.distill.losses import output_kd_loss, pinball_loss
from timesfm_lab.models import StudentConfig, TimesFMStudent, masked_mean_and_scale

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("gt", "kd", "dual_view", "cvrd")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compress(values: list[int], label: str) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            ranges.append(
                {
                    f"{label}_start": start,
                    f"{label}_end_inclusive": index - 1,
                    "windows": values[start],
                }
            )
            start = index
    return ranges


def _batch_count(windows: int, batch_size: int) -> int:
    count = math.ceil(windows / batch_size)
    return count - 1 if count > 1 and windows % batch_size == 1 else count


def _training_audit(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    records = {
        variant: json.loads(
            (result_dir / f"production-1m-student-{variant}-seed42.json").read_text()
        )
        for variant in VARIANTS
    }
    datasets = records["gt"]["extra"]["datasets"]
    total_windows = sum(int(item["training_windows"]) for item in datasets)
    total_batches = sum(
        _batch_count(int(item["training_windows"]), int(item["batch_size"])) for item in datasets
    )

    effective_weighting: dict[str, list[dict[str, Any]]] = {}
    for field in ("context", "horizon", "domain", "view_class"):
        grouped: dict[str, dict[str, int]] = defaultdict(
            lambda: {"datasets": 0, "windows": 0, "batches_per_epoch": 0}
        )
        for item in datasets:
            group = grouped[str(item[field])]
            group["datasets"] += 1
            group["windows"] += int(item["training_windows"])
            group["batches_per_epoch"] += _batch_count(
                int(item["training_windows"]), int(item["batch_size"])
            )
        effective_weighting[field] = []
        for name, values in sorted(grouped.items()):
            window_share = values["windows"] / total_windows
            step_share = values["batches_per_epoch"] / total_batches
            effective_weighting[field].append(
                {
                    field: int(name) if field in {"context", "horizon"} else name,
                    **values,
                    "window_fraction": window_share,
                    "optimizer_step_fraction": step_share,
                    "relative_per_window_weight": step_share / window_share,
                }
            )

    model = config["student"]
    patch = int(model["patch_length"])
    maximum_context = int(model["max_context"])
    position_count = maximum_context // patch
    position_exposure = [0] * position_count
    for item in datasets:
        patches = math.ceil(int(item["context"]) / patch)
        for row in range(position_count - patches, position_count):
            position_exposure[row] += int(item["training_windows"])

    maximum_horizon = int(model["max_horizon"])
    horizon_exposure = [0] * maximum_horizon
    for item in datasets:
        for row in range(int(item["horizon"])):
            horizon_exposure[row] += int(item["training_windows"])

    learning_curves = {}
    for variant, record in records.items():
        curve = record["extra"]["learning_curve"]
        values = [float(point["validation"]["student_pinball"]) for point in curve]
        best = min(range(len(values)), key=values.__getitem__)
        tail_start = max(0, len(values) - 6)
        loss_components = {
            key.removeprefix("training/"): value
            for key, value in record["metrics"].items()
            if key.startswith("training/") and key not in {"training/windows_per_second"}
        }
        learning_curves[variant] = {
            "best_step": int(curve[best]["step"]),
            "best_validation_pinball": values[best],
            "final_step": int(curve[-1]["step"]),
            "final_validation_pinball": values[-1],
            "last_50000_step_relative_change": (values[-1] - values[tail_start])
            / values[tail_start],
            "training_average_components": loss_components,
        }

    final_validation = records["gt"]["extra"]["learning_curve"][-1]["validation"]["by_dataset"]
    contributions = []
    total_loss_mass = sum(
        int(item["observed_targets"]) * float(item["student_pinball"])
        for item in final_validation.values()
    )
    for name, item in final_validation.items():
        mass = int(item["observed_targets"]) * float(item["student_pinball"])
        contributions.append(
            {
                "dataset": name,
                "observed_targets": int(item["observed_targets"]),
                "student_pinball": float(item["student_pinball"]),
                "student_loss_fraction": mass / total_loss_mass,
            }
        )
    contributions.sort(key=lambda item: item["student_loss_fraction"], reverse=True)

    return {
        "training_windows_after_split": total_windows,
        "batches_per_epoch": total_batches,
        "effective_weighting": effective_weighting,
        "position_embedding": {
            "rows": position_count,
            "patch_length": patch,
            "exposure_ranges_per_corpus_pass": _compress(position_exposure, "row"),
            "never_updated_rows": sum(value == 0 for value in position_exposure),
        },
        "output_head_horizon": {
            "rows": maximum_horizon,
            "exposure_ranges_per_corpus_pass": _compress(horizon_exposure, "horizon_row"),
            "never_updated_rows": sum(value == 0 for value in horizon_exposure),
        },
        "learning_curves": learning_curves,
        "gt_final_validation_top_loss_contributors": contributions[:10],
    }


def _gift_scope_audit(config_path: Path, data_root: Path) -> dict[str, Any]:
    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    from gift_eval.data import Dataset  # type: ignore[import-untyped]

    config = load_config(config_path)
    configurations = []
    totals = {
        "instances": 0,
        "instances_with_past_covariates": 0,
        "context_over_8192": 0,
        "context_over_teacher_15360": 0,
        "context_over_student_16384": 0,
    }
    horizon_instances: dict[str, int] = defaultdict(int)
    for item in config["evaluation"]["datasets"]:
        dataset = Dataset(name=item["name"], term=item["term"], to_univariate=False)
        lengths: list[int] = []
        covariate_instances = 0
        past_covariate_channels: set[int] = set()
        for entry in dataset.test_data.input:
            lengths.append(int(np.asarray(entry["target"]).shape[-1]))
            if entry.get("past_feat_dynamic_real") is not None:
                covariate_instances += 1
                past_covariate_channels.add(
                    int(np.atleast_2d(np.asarray(entry["past_feat_dynamic_real"])).shape[0])
                )
        count = len(lengths)
        horizon = int(dataset.prediction_length)
        horizon_instances[str(horizon)] += count
        totals["instances"] += count
        totals["instances_with_past_covariates"] += covariate_instances
        totals["context_over_8192"] += sum(length > 8192 for length in lengths)
        totals["context_over_teacher_15360"] += sum(length > 15360 for length in lengths)
        totals["context_over_student_16384"] += sum(length > 16384 for length in lengths)
        configurations.append(
            {
                "configuration": f"{item['name']}/{item['term']}",
                "instances": count,
                "target_variates": int(dataset.target_dim),
                "horizon": horizon,
                "minimum_context": min(lengths),
                "maximum_context": max(lengths),
                "instances_with_past_covariates": covariate_instances,
                "past_covariate_channel_counts": sorted(past_covariate_channels),
            }
        )
    return {
        "scope": config["evaluation"]["reportable_scope"],
        "configuration_count": len(configurations),
        **totals,
        "horizon_instance_counts": dict(
            sorted(horizon_instances.items(), key=lambda item: int(item[0]))
        ),
        "configurations": configurations,
    }


@torch.inference_mode()
def _model_probe(config: dict[str, Any], checkpoint: Path, gift_root: Path) -> dict[str, Any]:
    os.environ["GIFT_EVAL"] = str(gift_root.resolve())
    from gift_eval.data import Dataset  # type: ignore[import-untyped]

    torch.set_num_threads(1)
    model = TimesFMStudent(StudentConfig(**config["student"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    contexts = [
        np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
        for entry in Dataset(name="ett1/W", term="short", to_univariate=False).test_data.input
    ]

    def predict(arrays: list[np.ndarray]) -> torch.Tensor:
        length = max(array.shape[-1] for array in arrays)
        padded = [
            np.pad(array, ((0, 0), (length - array.shape[-1], 0)), constant_values=np.nan)
            for array in arrays
        ]
        return model(torch.from_numpy(np.stack(padded)), 8)

    reference = predict([contexts[0]])[0]
    variable_batch = predict(contexts)[0]
    unrelated = np.random.default_rng(4).normal(size=(7, 211)).astype(np.float32)
    unrelated_batch = predict([contexts[0], unrelated])[0]
    _, scale, _ = masked_mean_and_scale(torch.from_numpy(contexts[0]))
    scale = scale.squeeze(-1)[:, None, None]
    permutation = torch.tensor([6, 0, 4, 2, 5, 1, 3])
    inverse = torch.argsort(permutation)
    permuted = predict([contexts[0][permutation.numpy()]])[0][inverse]
    missing = contexts[0].copy()
    missing[:, 10:16] = np.nan
    missing_output = predict([missing])[0]

    def normalized_error(candidate: torch.Tensor) -> dict[str, float]:
        error = (candidate - reference).abs() / scale
        return {"maximum": float(error.max()), "mean": float(error.mean())}

    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "variable_length_batch_normalized_error": normalized_error(variable_batch),
        "unrelated_request_batch_normalized_error": normalized_error(unrelated_batch),
        "channel_permutation_normalized_error": normalized_error(permuted),
        "missing_context_output_finite": bool(torch.isfinite(missing_output).all()),
        "quantiles_monotonic": bool((reference[..., 1:] >= reference[..., :-1]).all()),
    }


def _learning_probe(pretrain_root: Path, cache_root: Path) -> dict[str, Any]:
    """Check loss wiring on eight fixed real windows with a deliberately tiny model."""

    from datasets import load_from_disk  # type: ignore[import-untyped]

    torch.manual_seed(7)
    shard = cache_root / "cdc_fluview_who_nrevss/shard-00000-of-00010.npz"
    with np.load(shard) as arrays:
        rows = arrays["row_index"][:8]
        ends = arrays["context_end"][:8]
        teacher = torch.from_numpy(arrays["teacher_multivariate"][:8].astype(np.float32))
    dataset = load_from_disk(str(pretrain_root / "cdc_fluview_who_nrevss"), keep_in_memory=False)
    contexts = []
    targets = []
    for row, end in zip(rows, ends, strict=True):
        values = np.atleast_2d(np.asarray(dataset[int(row)]["target"], dtype=np.float32))
        contexts.append(values[:, int(end) - 8 : int(end)])
        targets.append(values[:, int(end) : int(end) + 8])
    context = torch.from_numpy(np.stack(contexts))
    target = torch.from_numpy(np.stack(targets))
    mean, scale, _ = masked_mean_and_scale(context)
    mask = torch.isfinite(target)
    normalized_target = (torch.where(mask, target, mean) - mean) / scale
    forecast_mean = mean.unsqueeze(-1)
    forecast_scale = scale.unsqueeze(-1)
    normalized_teacher = (teacher - forecast_mean) / forecast_scale
    tiny_config = StudentConfig(
        d_model=48,
        num_layers=1,
        num_heads=3,
        ffn_dim=96,
        max_context=32,
        max_horizon=8,
    )
    initial = copy.deepcopy(TimesFMStudent(tiny_config).state_dict())
    histories: dict[str, Any] = {}
    gradients_finite = True
    for objective_name in ("ground_truth", "teacher_imitation"):
        model = TimesFMStudent(tiny_config)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
        samples = []
        for step in range(101):
            prediction = (model(context, 8) - forecast_mean) / forecast_scale
            loss = (
                pinball_loss(prediction, normalized_target, mask)
                if objective_name == "ground_truth"
                else output_kd_loss(prediction, normalized_teacher, mask)
            )
            if step in {0, 1, 10, 25, 50, 100}:
                samples.append({"step": step, "loss": float(loss.detach())})
            if step == 100:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_finite &= all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            optimizer.step()
        histories[objective_name] = samples
    return {
        "dataset": "cdc_fluview_who_nrevss",
        "cache_shard_sha256": _sha256(shard),
        "windows": 8,
        "context": 8,
        "horizon": 8,
        "variates": 4,
        "tiny_probe_parameter_count": TimesFMStudent(tiny_config).parameter_count,
        "finite_gradients": gradients_finite,
        "loss_curves": histories,
        "interpretation": "wiring probe only; not an architecture-quality experiment",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-config",
        type=Path,
        default=ROOT / "configs/distillation/production_1m.yaml",
    )
    parser.add_argument(
        "--gift-config",
        type=Path,
        default=ROOT / "configs/reproduction/gift_short_full.yaml",
    )
    parser.add_argument(
        "--multivariate-gift-config",
        type=Path,
        default=ROOT / "configs/reproduction/multivariate_short_full.yaml",
    )
    parser.add_argument("--gift-root", type=Path, default=ROOT / "data/gift-eval-full")
    parser.add_argument(
        "--pretrain-root", type=Path, default=ROOT / "data/gift-pretrain-production"
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "teacher_cache/production-1m")
    parser.add_argument(
        "--training-result-dir",
        type=Path,
        default=ROOT / "results/reproduction/distillation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/production-1m/gt/student-gt-best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/reproduction/distillation/performance-recovery-production-path-audit.json",
    )
    args = parser.parse_args()
    config = load_config(args.student_config)
    result = {
        "status": "success",
        "audited_baseline_commit": "85be358",
        "student_config": str(args.student_config.resolve()),
        "training": _training_audit(config, args.training_result_dir),
        "gift_55_scope": _gift_scope_audit(args.gift_config, args.gift_root),
        "gift_mv19_scope": _gift_scope_audit(args.multivariate_gift_config, args.gift_root),
        "model_probe": _model_probe(config, args.checkpoint, args.gift_root),
        "fixed_real_data_learning_probe": _learning_probe(args.pretrain_root, args.cache_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
