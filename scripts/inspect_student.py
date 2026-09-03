#!/usr/bin/env python3
"""Instantiate the student and print measured size and forward-shape facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from timesfm_lab.config import load_config
from timesfm_lab.models import StudentConfig, TimesFMStudent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config["seed"]))
    model = TimesFMStudent(StudentConfig(**config["student"])).to(args.device)
    context = torch.randn(2, 7, 512, device=args.device)
    context[:, :, :13] = float("nan")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(context, horizon=48)
    facts = {
        "parameter_count": model.parameter_count,
        "parameter_size_fp32_bytes": model.parameter_count * 4,
        "input_shape": list(context.shape),
        "output_shape": list(output.shape),
        "ordered_quantiles": bool((torch.diff(output.float(), dim=-1) >= 0).all()),
        "finite": bool(torch.isfinite(output).all()),
        "dtype": str(output.dtype),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    print(json.dumps(facts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
