#!/usr/bin/env python3
"""Measure target-preserving auxiliary temporal shifts on GIFT-Eval configurations."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME, _metric_scalar, to_gluonts_quantile_layout
from timesfm_lab.interventions import shift_auxiliary_channels
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


class ShiftPredictor:
    def __init__(
        self,
        forecaster: Any,
        *,
        prediction_length: int,
        batch_size: int,
        patch_size: int,
        offset: int,
        quantiles: tuple[float, ...],
    ) -> None:
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.offset = offset
        self.quantiles = quantiles

    def predict(
        self, test_data_input: Iterable[Mapping[str, Any]], batch_size: int | None = None
    ) -> list[Any]:
        from gluonts.model.forecast import QuantileForecast

        entries = list(test_data_input)
        targets = [np.atleast_2d(np.asarray(e["target"], dtype=np.float32)) for e in entries]
        queries: list[np.ndarray] = []
        selections: list[tuple[int, int]] = []
        for entry_index, target in enumerate(targets):
            for target_index in range(target.shape[0]):
                shifted = shift_auxiliary_channels(
                    target,
                    target_index=target_index,
                    patch_size=self.patch_size,
                    offset_patches=self.offset,
                    fill_value=0.0,
                )
                queries.append(shifted.values)
                selections.append((entry_index, target_index))
        self.forecaster.config.per_core_batch_size = batch_size or self.batch_size
        outputs = list(
            self.forecaster.predict_batch(
                contexts=queries,
                horizon=self.prediction_length,
                return_quantiles=True,
                use_symmetric_averaging=True,
                make_positive=True,
                sort_quantiles=True,
                use_znorm=False,
                padding_mode="none",
            )
        )
        combined = [
            np.empty((x.shape[0], self.prediction_length, len(self.quantiles)), np.float32)
            for x in targets
        ]
        for output, (entry_index, target_index) in zip(outputs, selections, strict=True):
            combined[entry_index][target_index] = output.quantiles[target_index]
        return [
            QuantileForecast(
                forecast_arrays=to_gluonts_quantile_layout(
                    quantiles, prediction_length=self.prediction_length
                ),
                forecast_keys=[str(q) for q in self.quantiles],
                start_date=entry["start"] + target.shape[-1],
            )
            for entry, target, quantiles in zip(entries, targets, combined, strict=True)
        ]


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
    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=model_cfg["id"],
            revision=config["model_revision"],
            device=model_cfg["device"],
            per_core_batch_size=model_cfg["batch_size"],
        )
    )
    results: list[dict[str, object]] = []
    datasets = config["evaluation"]["datasets"][args.shard_index :: args.num_shards]
    for dataset_cfg in datasets:
        dataset = Dataset(name=dataset_cfg["name"], term=dataset_cfg["term"], to_univariate=False)
        for offset in config["evaluation"]["offsets_patches"]:
            predictor = ShiftPredictor(
                forecaster,
                prediction_length=dataset.prediction_length,
                batch_size=model_cfg["batch_size"],
                patch_size=config["evaluation"]["patch_size"],
                offset=int(offset),
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
                "offset_patches": int(offset),
                "offset_steps": int(offset) * config["evaluation"]["patch_size"],
                "instances": len(dataset.test_data),
                "target_variates": int(dataset.target_dim),
                MASE_NAME: _metric_scalar(frame, MASE_NAME),
                MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
            }
            results.append(item)
            print(item, flush=True)
    record.extra.update(
        {
            "results": results,
            "shard": {"index": args.shard_index, "count": args.num_shards},
            "shift_convention": "output[t] = input[t - offset]; zero fill; target unchanged",
            "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        }
    )
    record.succeed({})
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
