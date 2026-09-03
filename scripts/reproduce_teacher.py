#!/usr/bin/env python3
"""Run the smallest pinned TimesFM 3.0 reference forecast correctness check."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from timesfm_lab.config import load_config
from timesfm_lab.run_record import RunRecord
from timesfm_lab.teacher.smoke import run_reference_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="override output path from the config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output = args.output or Path(config["output_path"])
    record = RunRecord.start(
        run_id=str(config["run_id"]),
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=str(config["model_revision"]),
        dataset_revision=str(config["dataset_revision"]),
        hardware_snapshot=str(config["hardware_snapshot"]),
        repository=Path(__file__).resolve().parents[1],
    )
    try:
        evidence = run_reference_smoke(config)
    except BaseException as error:
        record.extra["traceback"] = traceback.format_exc()
        record.fail(f"{type(error).__name__}: {error}")
        record.write(output)
        raise
    record.extra["evidence"] = evidence
    record.succeed({})
    record.write(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
