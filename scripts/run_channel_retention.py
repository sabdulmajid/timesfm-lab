#!/usr/bin/env python3
"""Measure target-preserving channel-retention effects on complete GIFT configurations."""

from __future__ import annotations

import argparse
import math
import os
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import (
    MASE_NAME,
    MWQL_NAME,
    _metric_scalar,
    to_gluonts_quantile_layout,
)
from timesfm_lab.interventions import retain_channel_fraction
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


class ChannelRetentionPredictor:
    """Forecast every target with only a seeded fraction of its peer channels."""

    def __init__(
        self,
        forecaster: Any,
        *,
        prediction_length: int,
        batch_size: int,
        fraction: float,
        seed: int,
        quantiles: tuple[float, ...],
    ) -> None:
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.fraction = fraction
        self.seed = seed
        self.quantiles = quantiles

    def predict(
        self,
        test_data_input: Iterable[Mapping[str, Any]],
        batch_size: int | None = None,
    ) -> list[Any]:
        from gluonts.model.forecast import QuantileForecast

        entries = list(test_data_input)
        query_contexts: list[np.ndarray] = []
        selections: list[tuple[int, int, int]] = []
        targets: list[np.ndarray] = []
        for entry_index, entry in enumerate(entries):
            target = np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
            targets.append(target)
            for target_index in range(target.shape[0]):
                intervention = retain_channel_fraction(
                    target,
                    target_index=target_index,
                    fraction=self.fraction,
                    seed=self.seed + entry_index * 100_003 + target_index,
                )
                query_contexts.append(intervention.values)
                selections.append((entry_index, target_index, intervention.target_index))

        self.forecaster.config.per_core_batch_size = batch_size or self.batch_size
        outputs = list(
            self.forecaster.predict_batch(
                contexts=query_contexts,
                horizon=self.prediction_length,
                return_quantiles=True,
                use_symmetric_averaging=True,
                make_positive=True,
                sort_quantiles=True,
                use_znorm=False,
                padding_mode="none",
            )
        )
        quantile_outputs = [
            np.empty(
                (target.shape[0], self.prediction_length, len(self.quantiles)),
                dtype=np.float32,
            )
            for target in targets
        ]
        for output, (entry_index, target_index, selected_target) in zip(
            outputs, selections, strict=True
        ):
            if output.quantiles is None:
                raise RuntimeError("TimesFM returned no quantiles")
            quantile_outputs[entry_index][target_index] = output.quantiles[selected_target]

        return [
            QuantileForecast(
                forecast_arrays=to_gluonts_quantile_layout(
                    quantile_output,
                    prediction_length=self.prediction_length,
                ),
                forecast_keys=[str(value) for value in self.quantiles],
                start_date=entry["start"] + target.shape[-1],
            )
            for entry, target, quantile_output in zip(
                entries, targets, quantile_outputs, strict=True
            )
        ]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    config = load_config(args.config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    import torch
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    model_config = config["model"]
    quantiles = tuple(float(value) for value in config["evaluation"]["quantiles"])
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
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(model_config["id"]),
            revision=str(config["model_revision"]),
            device=str(model_config["device"]),
            per_core_batch_size=int(model_config["batch_size"]),
        )
    )
    datasets = config["evaluation"]["datasets"][args.shard_index :: args.num_shards]
    for dataset_config in datasets:
        name, term = str(dataset_config["name"]), str(dataset_config["term"])
        dataset = Dataset(name=name, term=term, to_univariate=False)
        for fraction_value in config["evaluation"]["retention_fractions"]:
            fraction = float(fraction_value)
            try:
                predictor = ChannelRetentionPredictor(
                    forecaster,
                    prediction_length=dataset.prediction_length,
                    batch_size=int(model_config["batch_size"]),
                    fraction=fraction,
                    seed=int(config["seed"]),
                    quantiles=quantiles,
                )
                frame = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=[
                        MASE(),
                        MeanWeightedSumQuantileLoss(quantile_levels=list(quantiles)),
                    ],
                    batch_size=int(model_config["batch_size"]),
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(dataset.freq),
                )
                item = {
                    "configuration": f"{name}/{term}",
                    "retention_fraction_requested": fraction,
                    "channels_retained_per_target": max(
                        1, math.ceil(fraction * dataset.target_dim)
                    ),
                    "total_channels": int(dataset.target_dim),
                    "instances": len(dataset.test_data),
                    MASE_NAME: _metric_scalar(frame, MASE_NAME),
                    MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
                }
                results.append(item)
                print(item, flush=True)
            except BaseException as error:
                failures.append(
                    {
                        "configuration": f"{name}/{term}",
                        "fraction": str(fraction),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
    record.extra.update(
        {
            "results": results,
            "failures": failures,
            "shard": {"index": args.shard_index, "count": args.num_shards},
            "runtime": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
    )
    if failures:
        record.fail(f"{len(failures)} intervention evaluations failed")
    else:
        record.succeed({})
    record.write(output)
    print(output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
