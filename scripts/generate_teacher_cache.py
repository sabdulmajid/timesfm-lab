#!/usr/bin/env python3
"""Generate disjoint, checksummed TimesFM MV/UV teacher-cache shards."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

from timesfm_lab.config import load_config
from timesfm_lab.distill.data import sample_windows
from timesfm_lab.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config = load_config(args.config)

    import torch
    from datasets import load_from_disk  # type: ignore[import-untyped]
    from timesfm3 import ModelConfig, TimesFM3Evaluator  # type: ignore[import-untyped]

    cache_cfg = config["cache"]
    dataset_name = args.dataset or cache_cfg.get("dataset")
    if not dataset_name:
        raise ValueError("dataset must be present in the config or supplied with --dataset")
    dataset_path = args.data_root / dataset_name
    dataset = load_from_disk(str(dataset_path))
    source = [
        np.atleast_2d(np.asarray(dataset[row]["target"], dtype=np.float32))
        for row in range(len(dataset))
    ]
    all_windows, sampling_report = sample_windows(
        [values.shape[-1] for values in source],
        context_length=int(cache_cfg["context_length"]),
        horizon=int(cache_cfg["horizon"]),
        total_windows=int(cache_cfg["total_windows"]),
        seed=int(config["seed"]),
        # Omitted means exact historical behavior. Production configs must opt in
        # to the no-replacement methodology explicitly.
        mode=cache_cfg.get("sampling", "with_replacement"),
    )
    if sampling_report["shortfall"]:
        print(
            "WARNING: requested "
            f"{sampling_report['requested_windows']} windows but only "
            f"{sampling_report['available_unique_windows']} distinct windows exist; "
            f"caching {sampling_report['selected_windows']} without duplication",
            flush=True,
        )
    selected = all_windows[args.shard_index :: args.num_shards]
    output = args.cache_dir / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.npz"
    dataset_slug = dataset_name.lower().replace("/", "-").replace("_", "-")
    result_path = Path("results/reproduction/distillation") / (
        f"{config['run_id']}-{dataset_slug}-shard{args.shard_index}of{args.num_shards}.json"
    )
    record = RunRecord.start(
        run_id=(
            f"{config['run_id']}-{dataset_slug}-"
            f"shard{args.shard_index}of{args.num_shards}"
        ),
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
    teacher_mv: list[npt.NDArray[np.float32]] = []
    teacher_uv: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(selected), batch_size):
        batch_meta = selected[start : start + batch_size]
        contexts = []
        for row, end in batch_meta:
            target = source[row]
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
            row_indices.append(row)
            end_indices.append(end)
            teacher_mv.append(mv)
            teacher_uv.append(uv)
        print(f"cached {len(row_indices)}/{len(selected)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    if teacher_mv:
        stacked_mv = np.stack(teacher_mv)
        stacked_uv = np.stack(teacher_uv)
    else:
        # A fully exhausted population may leave some modulo shards empty. Write
        # a valid empty shard so multi-worker cache jobs complete transparently.
        eligible_variates = next(
            values.shape[0]
            for values in source
            if values.shape[-1] >= context_length + horizon
        )
        stacked_mv = np.empty((0, eligible_variates, horizon, 9), dtype=np.float32)
        stacked_uv = np.empty_like(stacked_mv)
    half_mv = stacked_mv.astype(np.float16)
    half_uv = stacked_uv.astype(np.float16)
    fp16_safe = bool(np.isfinite(half_mv).all() and np.isfinite(half_uv).all())
    if fp16_safe:
        stored_mv = half_mv
        stored_uv = half_uv
        max_cast_error: float | None = (
            max(
                float(np.max(np.abs(stacked_mv - half_mv.astype(np.float32)))),
                float(np.max(np.abs(stacked_uv - half_uv.astype(np.float32)))),
            )
            if stacked_mv.size
            else 0.0
        )
    else:
        stored_mv = stacked_mv
        stored_uv = stacked_uv
        max_cast_error = None
    np.savez_compressed(
        output,
        row_index=np.asarray(row_indices, dtype=np.int32),
        context_end=np.asarray(end_indices, dtype=np.int32),
        teacher_multivariate=stored_mv,
        teacher_univariate=stored_uv,
    )
    metadata = {
        "cache_path": str(output.resolve()),
        "cache_sha256": _sha256(output),
        "cache_bytes": output.stat().st_size,
        "windows": len(selected),
        "dataset": dataset_name,
        "context_length": context_length,
        "horizon": horizon,
        "variates": int(stacked_mv.shape[1]),
        "requested_output_dtype": cache_cfg["output_dtype"],
        "output_dtype": str(stored_mv.dtype),
        "fp16_safe": fp16_safe,
        "float16_max_abs_error": max_cast_error,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "sampling": sampling_report,
        "sampler": (
            "round-robin rows; independent seeded uniform context ends; modulo sharding"
            if sampling_report["mode"] == "with_replacement"
            else "fair row allocation; seeded temporal strata without replacement; modulo sharding"
        ),
        "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    record.extra.update(metadata)
    metrics = {
        "windows": float(len(selected)),
        "cache_bytes": float(output.stat().st_size),
        "fp16_safe": float(fp16_safe),
    }
    if max_cast_error is not None:
        metrics["float16_max_abs_error"] = max_cast_error
    record.succeed(metrics)
    record.write(result_path)
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
