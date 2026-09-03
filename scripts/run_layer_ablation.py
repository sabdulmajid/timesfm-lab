#!/usr/bin/env python3
"""Disable the actual variate-attention branch in selected loaded TimesFM layers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME, TimesFm3GiftPredictor, _metric_scalar
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())

    import torch
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    model_cfg = config["model"]
    quantiles = tuple(float(q) for q in config["evaluation"]["quantiles"])
    run_id = f"{config['run_id']}-shard{args.shard_index}of{args.num_shards}"
    output = Path(config["output_directory"]) / f"{run_id}.json"
    record = RunRecord.start(
        run_id=run_id,
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=ROOT,
    )
    teacher = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=model_cfg["id"],
            revision=config["model_revision"],
            device=model_cfg["device"],
            per_core_batch_size=model_cfg["batch_size"],
        )
    )
    layers = teacher.model.transformer_stack.layers
    if len(layers) != model_cfg["layers"]:
        raise RuntimeError(f"expected {model_cfg['layers']} layers, loaded {len(layers)}")

    results: list[dict[str, object]] = []
    groups = config["evaluation"]["disable_groups"][args.shard_index :: args.num_shards]
    for group in groups:
        disabled = {int(index) for index in group["layers"]}
        for index, layer in enumerate(layers):
            layer.use_variate_attention = index not in disabled
        for dataset_cfg in config["evaluation"]["datasets"]:
            dataset = Dataset(
                name=dataset_cfg["name"], term=dataset_cfg["term"], to_univariate=False
            )
            predictor = TimesFm3GiftPredictor(
                teacher,
                prediction_length=dataset.prediction_length,
                batch_size=model_cfg["batch_size"],
                quantiles=quantiles,
            )
            frame = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=list(quantiles))],
                batch_size=model_cfg["batch_size"],
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),
            )
            item = {
                "configuration": f"{dataset_cfg['name']}/{dataset_cfg['term']}",
                "disabled_group": group["name"],
                "disabled_layers": sorted(disabled),
                MASE_NAME: _metric_scalar(frame, MASE_NAME),
                MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
            }
            if not np.isfinite([item[MASE_NAME], item[MWQL_NAME]]).all():
                raise RuntimeError("non-finite metric")
            results.append(item)
            print(item, flush=True)
    record.extra.update(
        {
            "results": results,
            "shard": {"index": args.shard_index, "count": args.num_shards},
            "intervention": (
                "sets MixingTransformer.use_variate_attention=False in selected layers; "
                "the branch is skipped without modifying weights"
            ),
            "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        }
    )
    record.succeed({})
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
