#!/usr/bin/env python3
"""Evaluate complete GIFT-Eval configurations in multivariate or univariate mode."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME, TimesFm3GiftPredictor, _metric_scalar
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("multivariate", "univariate"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    config = load_config(args.config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())

    import torch
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    model_config = config["model"]
    quantiles = tuple(float(value) for value in config["evaluation"]["quantiles"])
    mode = args.mode
    run_id = f"{config['run_id']}-{mode}-seed{config['seed']}"
    output = args.output or Path(config["output_directory"]) / f"{run_id}.json"
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
    try:
        forecaster = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=str(model_config["id"]),
                revision=str(config["model_revision"]),
                device=str(model_config["device"]),
                per_core_batch_size=int(model_config["batch_size"]),
            )
        )
        for dataset_config in config["evaluation"]["datasets"]:
            name = str(dataset_config["name"])
            term = str(dataset_config["term"])
            try:
                dataset = Dataset(name=name, term=term, to_univariate=False)
                predictor = TimesFm3GiftPredictor(
                    forecaster,
                    prediction_length=dataset.prediction_length,
                    batch_size=int(model_config["batch_size"]),
                    quantiles=quantiles,
                    univariate=mode == "univariate",
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
                result = {
                    "configuration": f"{name}/{term}",
                    "mode": mode,
                    "instances": len(dataset.test_data),
                    "target_variates": int(dataset.target_dim),
                    "prediction_length": int(dataset.prediction_length),
                    MASE_NAME: _metric_scalar(frame, MASE_NAME),
                    MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
                }
                results.append(result)
                print(result, flush=True)
            except BaseException as error:
                failure = {
                    "configuration": f"{name}/{term}",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(failure, flush=True)
        record.extra.update(
            {
                "mode": mode,
                "scope": config["evaluation"]["reportable_scope"],
                "results": results,
                "failures": failures,
                "runtime": {
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                },
            }
        )
        if failures:
            record.fail(f"{len(failures)} of {len(results) + len(failures)} configurations failed")
        else:
            record.succeed(
                {
                    f"{item['configuration']}/{metric}": float(item[metric])
                    for item in results
                    for metric in (MASE_NAME, MWQL_NAME)
                }
            )
        record.write(output)
    except BaseException as error:
        record.extra["traceback"] = traceback.format_exc()
        if record.ended_at is None:
            record.fail(f"{type(error).__name__}: {error}")
        record.write(output)
        raise
    print(output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
