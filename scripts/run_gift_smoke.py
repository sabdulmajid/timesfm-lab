#!/usr/bin/env python3
"""Run the pinned two-instance TimesFM 3 GIFT-Eval protocol smoke."""

from __future__ import annotations

import argparse
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

from timesfm_lab.config import load_config
from timesfm_lab.eval.gift import run_gift_smoke
from timesfm_lab.run_record import RunRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/reproduction/teacher_smoke.yaml"


def _default_hardware_snapshot() -> Path | None:
    snapshots = sorted((REPOSITORY_ROOT / "results/environment").glob("environment_*.txt"))
    return snapshots[-1] if snapshots else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="download root for the pinned Salesforce/GiftEval snapshot (or set GIFT_EVAL)",
    )
    parser.add_argument("--hardware-snapshot", type=Path)
    parser.add_argument("--output", type=Path, help="override append-only run-record path")
    parser.add_argument("--run-id", help="override the generated unique run identifier")
    parser.add_argument(
        "--mode",
        choices=("multivariate", "univariate"),
        default="multivariate",
        help="native multivariate inference or official target-only univariate mode",
    )
    return parser.parse_args()


def _resolve_data_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.resolve()
    configured = os.environ.get("GIFT_EVAL")
    if configured:
        return Path(configured).expanduser().resolve()
    raise ValueError("provide --data-root or set GIFT_EVAL to the pinned dataset snapshot")


def main() -> int:
    args = _arguments()
    config_path = args.config.resolve()
    config = load_config(config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = args.run_id or f"{config['experiment']}-{args.mode}-{timestamp}"
    output_directory = Path(config["output"]["directory"])
    if not output_directory.is_absolute():
        output_directory = REPOSITORY_ROOT / output_directory
    output = args.output.resolve() if args.output else output_directory / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run record: {output}")
    snapshot = args.hardware_snapshot or _default_hardware_snapshot()

    record = RunRecord.start(
        run_id=run_id,
        config_path=str(config_path),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(snapshot.resolve()) if snapshot else "NOT_RECORDED",
        repository=REPOSITORY_ROOT,
    )
    record.extra.update(
        {
            "protocol_smoke_only": True,
            "reportable_benchmark": False,
            "reportability_note": str(config["evaluation"]["reportability_note"]),
        }
    )
    try:
        result = run_gift_smoke(
            config,
            data_root=_resolve_data_root(args.data_root),
            univariate=args.mode == "univariate",
        )
    except BaseException as error:
        record.extra["traceback"] = traceback.format_exc()
        record.fail(f"{type(error).__name__}: {error}")
        record.write(output)
        raise
    record.extra["evidence"] = result.evidence
    record.succeed(result.metrics)
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
