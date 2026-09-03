#!/usr/bin/env python3
"""Validate and ingest the recovered manual TimesFM teacher batch sweeps."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_verify(result_path: Path, log_path: Path, variates: int) -> list[dict[str, Any]]:
    results = json.loads(result_path.read_text())
    logged = [
        ast.literal_eval(line)
        for line in log_path.read_text().splitlines()
        if line.startswith("{")
    ]
    if results != logged:
        raise ValueError(f"JSON/log mismatch for V={variates}")
    expected_batches = [16, 32, 64, 128, 256, 512]
    if [int(item["batch"]) for item in results] != expected_batches:
        raise ValueError(f"unexpected batch sequence for V={variates}")
    for item in results:
        if (
            int(item["variates"]) != variates
            or int(item["context"]) != 2048
            or int(item["horizon"]) != 64
        ):
            raise ValueError(f"unexpected shape in V={variates} sweep: {item}")
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    peak = max(float(item["windows_per_second"]) for item in results)
    selected = next(
        item for item in results if float(item["windows_per_second"]) >= 0.99 * peak
    )
    last_gain = (
        float(results[-1]["windows_per_second"])
        / float(results[-2]["windows_per_second"])
        - 1.0
    )
    return {
        "measurements": results,
        "peak_windows_per_second": peak,
        "earliest_batch_within_one_percent_of_peak": int(selected["batch"]),
        "selected_windows_per_second": float(selected["windows_per_second"]),
        "relative_throughput_change_256_to_512": last_gain,
        "extend_beyond_512": abs(last_gain) >= 0.01 and last_gain > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--v8-json", type=Path, required=True)
    parser.add_argument("--v8-log", type=Path, required=True)
    parser.add_argument("--v32-json", type=Path, required=True)
    parser.add_argument("--v32-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_script.read_text()
    required_fragments = (
        'REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"',
        "CONTEXT = 2048",
        "HORIZON = 64",
        "BATCHES = [16, 32, 64, 128, 256, 512]",
        "for _ in range(2):",
        "for _ in range(4):",
        "torch.cuda.synchronize()",
        "univariate=False",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"source script failed methodology checks: {missing}")

    result = {
        "status": "audited_with_limitations",
        "source_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                args.source_script,
                args.v8_json,
                args.v8_log,
                args.v32_json,
                args.v32_log,
            )
        },
        "methodology": {
            "model": "google/timesfm-3.0-pytorch",
            "model_revision": "43046b85ec22d584a13f8098c2ed39c889e129c2",
            "input": "seed-42 synthetic dense float32 contexts",
            "context": 2048,
            "horizon": 64,
            "warmups": 2,
            "timed_repeats": 4,
            "timing": "synchronized end-to-end predict_batch wall time",
            "view": "one multivariate teacher view only",
            "limitations": [
                "not the production MV+UV paired cache unit",
                "synthetic rather than decoded GiftEvalPretrain rows",
                "only mean latency retained; no raw repeat samples",
                "GPU utilization, host preprocessing, H2D time, and software versions absent",
            ],
        },
        "conclusion": (
            "Both curves plateau before batch 512, so no larger-batch extension is justified. "
            "Use the paired-view real-data autotune for production cache batch selection."
        ),
        "sweeps": {
            "variates_8": _summarize(_load_and_verify(args.v8_json, args.v8_log, 8)),
            "variates_32": _summarize(_load_and_verify(args.v32_json, args.v32_log, 32)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
