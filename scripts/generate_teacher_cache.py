#!/usr/bin/env python3
"""Generate disjoint, checksummed TimesFM MV/UV teacher-cache shards."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_lab.config import load_config
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _windows(dataset: Any, config: dict[str, Any], seed: int) -> list[tuple[int, int]]:
    context = int(config["context_length"])
    horizon = int(config["horizon"])
    total = int(config["total_windows"])
    generators = [np.random.default_rng(seed * 10_000 + index) for index in range(len(dataset))]
    result: list[tuple[int, int]] = []
    for global_index in range(total):
        row = global_index % len(dataset)
        values = np.asarray(dataset[row]["target"], dtype=np.float32)
        values = np.atleast_2d(values)
        if values.shape[-1] < context + horizon:
            raise ValueError(f"row {row} is too short for the configured window")
        end = int(generators[row].integers(context, values.shape[-1] - horizon + 1))
        result.append((row, end))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config = load_config(args.config)

    import torch
    from datasets import load_from_disk
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    cache_cfg = config["cache"]
    dataset_path = args.data_root / cache_cfg["dataset"]
    dataset = load_from_disk(str(dataset_path))
    selected = _windows(dataset, cache_cfg, int(config["seed"]))[
        args.shard_index :: args.num_shards
    ]
    output = args.cache_dir / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.npz"
    result_path = Path("results/reproduction/distillation") / (
        f"teacher-cache-pilot-shard{args.shard_index}of{args.num_shards}.json"
    )
    record = RunRecord.start(
        run_id=f"{config['run_id']}-shard{args.shard_index}of{args.num_shards}",
        config_path=str(args.config),
        seed=int(config["seed"]),
        model_revision=config["model_revision"],
        dataset_revision=config["dataset_revision"],
        hardware_snapshot=config["hardware_snapshot"],
        repository=ROOT,
    )
    model_cfg = config["model"]
    teacher = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=model_cfg["id"],
            revision=config["model_revision"],
            device=model_cfg["device"],
            per_core_batch_size=int(model_cfg["batch_size"]),
        )
    )
    batch_size = int(model_cfg["batch_size"])
    horizon = int(cache_cfg["horizon"])
    context_length = int(cache_cfg["context_length"])
    row_indices: list[int] = []
    end_indices: list[int] = []
    teacher_mv: list[np.ndarray] = []
    teacher_uv: list[np.ndarray] = []
    max_cast_error = 0.0
    for start in range(0, len(selected), batch_size):
        batch_meta = selected[start : start + batch_size]
        contexts = []
        for row, end in batch_meta:
            target = np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
            contexts.append(target[:, end - context_length : end])
        mv_outputs = list(
            teacher.predict_batch(
                contexts=contexts,
                horizon=horizon,
                return_quantiles=True,
                sort_quantiles=True,
                univariate=False,
            )
        )
        uv_outputs = list(
            teacher.predict_batch(
                contexts=contexts,
                horizon=horizon,
                return_quantiles=True,
                sort_quantiles=True,
                univariate=True,
            )
        )
        for (row, end), mv_output, uv_output in zip(
            batch_meta, mv_outputs, uv_outputs, strict=True
        ):
            mv = np.asarray(mv_output.quantiles, dtype=np.float32)
            uv = np.asarray(uv_output.quantiles, dtype=np.float32)
            if mv.shape != uv.shape or mv.shape[-2:] != (horizon, 9):
                raise RuntimeError(f"unexpected teacher cache shape {mv.shape} vs {uv.shape}")
            if not np.isfinite(mv).all() or not np.isfinite(uv).all():
                raise RuntimeError("teacher cache contains non-finite values")
            max_cast_error = max(
                max_cast_error,
                float(np.max(np.abs(mv - mv.astype(np.float16).astype(np.float32)))),
                float(np.max(np.abs(uv - uv.astype(np.float16).astype(np.float32)))),
            )
            row_indices.append(row)
            end_indices.append(end)
            teacher_mv.append(mv.astype(np.float16))
            teacher_uv.append(uv.astype(np.float16))
        print(f"cached {len(row_indices)}/{len(selected)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        row_index=np.asarray(row_indices, dtype=np.int32),
        context_end=np.asarray(end_indices, dtype=np.int32),
        teacher_multivariate=np.stack(teacher_mv),
        teacher_univariate=np.stack(teacher_uv),
    )
    metadata = {
        "cache_path": str(output.resolve()),
        "cache_sha256": _sha256(output),
        "cache_bytes": output.stat().st_size,
        "windows": len(selected),
        "dataset": cache_cfg["dataset"],
        "context_length": context_length,
        "horizon": horizon,
        "variates": int(teacher_mv[0].shape[0]),
        "output_dtype": str(teacher_mv[0].dtype),
        "float16_max_abs_error": max_cast_error,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "sampler": "round-robin rows; independent seeded uniform context ends; modulo sharding",
        "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    record.extra.update(metadata)
    record.succeed(
        {
            "windows": float(len(selected)),
            "cache_bytes": float(output.stat().st_size),
            "float16_max_abs_error": max_cast_error,
        }
    )
    record.write(result_path)
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
