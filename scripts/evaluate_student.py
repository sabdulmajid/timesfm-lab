#!/usr/bin/env python3
"""Evaluate a locally trained student with the official GIFT-Eval metrics."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import MASE_NAME, MWQL_NAME, _metric_scalar
from timesfm_lab.eval.student import StudentGiftPredictor
from timesfm_lab.models import StudentConfig, TimesFMStudent
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_config", type=Path)
    parser.add_argument("evaluation_config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("gt", "kd", "relkd"), required=True)
    parser.add_argument("--mode", choices=("multivariate", "univariate"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    student_config = load_config(args.student_config)
    evaluation_config = load_config(args.evaluation_config)
    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())

    from gift_eval.data import Dataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    device = torch.device("cuda:0")
    model = TimesFMStudent(StudentConfig(**student_config["student"]))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    run_id = f"{student_config['run_id']}-{args.variant}-{args.mode}-gift-short"
    output = args.output or Path("results/reproduction/distillation") / f"{run_id}.json"
    record = RunRecord.start(
        run_id=run_id,
        config_path=f"{args.student_config};{args.evaluation_config}",
        seed=int(student_config["seed"]),
        model_revision=student_config["model_revision"],
        dataset_revision=evaluation_config["dataset_revision"],
        hardware_snapshot=student_config["hardware_snapshot"],
        repository=ROOT,
    )
    results = []
    failures = []
    for item in evaluation_config["evaluation"]["datasets"]:
        configuration = f"{item['name']}/{item['term']}"
        try:
            dataset = Dataset(name=item["name"], term=item["term"], to_univariate=False)
            predictor = StudentGiftPredictor(
                model,
                prediction_length=dataset.prediction_length,
                batch_size=16,
                device=device,
                univariate=args.mode == "univariate",
            )
            frame = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(
                        quantile_levels=[value / 10 for value in range(1, 10)]
                    ),
                ],
                batch_size=16,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),
            )
            result = {
                "configuration": configuration,
                "variant": args.variant,
                "mode": args.mode,
                "instances": len(dataset.test_data),
                "target_variates": int(dataset.target_dim),
                "prediction_length": int(dataset.prediction_length),
                MASE_NAME: _metric_scalar(frame, MASE_NAME),
                MWQL_NAME: _metric_scalar(frame, MWQL_NAME),
            }
            if not np.isfinite([result[MASE_NAME], result[MWQL_NAME]]).all():
                raise RuntimeError("non-finite metric")
            results.append(result)
            print(result, flush=True)
        except BaseException as error:
            failures.append(
                {
                    "configuration": configuration,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(failures[-1], flush=True)
    record.extra.update(
        {
            "variant": args.variant,
            "mode": args.mode,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_distributed": False,
            "parameter_count": model.parameter_count,
            "precision": "bfloat16 autocast",
            "results": results,
            "failures": failures,
            "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        }
    )
    if failures:
        record.fail(f"{len(failures)} configurations failed")
    else:
        record.succeed(
            {
                f"{result['configuration']}/{metric}": result[metric]
                for result in results
                for metric in (MASE_NAME, MWQL_NAME)
            }
        )
    record.write(output)
    print(output, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
