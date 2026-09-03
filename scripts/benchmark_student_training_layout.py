#!/usr/bin/env python3
"""Benchmark real-corpus single-GPU or two-rank DDP student training throughput."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from generate_production_cache import _GpuSampler
from train_production_student import (
    _epoch_batches,
    _load_corpus,
    _loss,
    _materialize,
    _StudentViews,
)

from timesfm_lab.config import load_config
from timesfm_lab.distill.losses import DistillationLoss, LossWeights
from timesfm_lab.models import StudentConfig, TimesFMStudent


def _reduce_sum(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--variant", choices=("gt", "kd", "dual_view", "cvrd"), required=True)
    parser.add_argument("--layout", choices=("single", "ddp"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--barrier-dir",
        type=Path,
        help="filesystem barrier for concurrently launched independent single-GPU workers",
    )
    parser.add_argument("--barrier-participant")
    parser.add_argument("--barrier-participants", type=int, default=2)
    args = parser.parse_args()
    distributed = args.layout == "ddp"
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size != 2:
            raise ValueError("the layout benchmark is defined for exactly two DDP ranks")
        torch.cuda.set_device(local_rank)
        physical_gpu = local_rank
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(0)
        if args.physical_gpu is None:
            raise ValueError("single-GPU mode requires --physical-gpu for utilization sampling")
        physical_gpu = args.physical_gpu
    device = torch.device(f"cuda:{local_rank}")
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text())
    training = config["training"]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    load_started = time.perf_counter()
    corpora = [
        _load_corpus(
            item,
            data_root=args.data_root,
            cache_root=args.cache_root,
            validation_fraction=float(training["validation_fraction"]),
            validation_mode=str(training["validation_split"]),
            seed=seed,
            batch_sizes=training["batch_size_by_context"],
        )
        for item in plan["datasets"]
    ]
    load_seconds = time.perf_counter() - load_started
    student = TimesFMStudent(StudentConfig(**config["student"])).to(device)
    model: Any = _StudentViews(student).to(device)
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    objective = DistillationLoss(LossWeights.from_mapping(training["loss_weights"][args.variant]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    batches = []
    epoch = 0
    needed = args.warmup_steps + args.measured_steps
    while len(batches) < needed:
        for corpus_index, indices in _epoch_batches(corpora, seed, epoch):
            # Both ranks must receive data. Skipping a possible singleton remainder
            # changes only this throughput probe, never production training.
            if not distributed or len(indices) >= world_size:
                batches.append((corpus_index, indices))
                if len(batches) == needed:
                    break
        epoch += 1

    def train_step(corpus_index: int, global_indices: np.ndarray) -> int:
        corpus = corpora[corpus_index]
        indices = global_indices[rank::world_size] if distributed else global_indices
        context, target, teacher_primary, teacher_uv = _materialize(corpus, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = _loss(
                model,
                objective,
                args.variant,
                corpus,
                context,
                target,
                teacher_primary,
                teacher_uv,
                float(config["student"]["normalization_epsilon"]),
            )
        values["loss"].backward()
        optimizer.step()
        return len(indices)

    for corpus_index, indices in batches[: args.warmup_steps]:
        train_step(corpus_index, indices)
    torch.cuda.synchronize()
    if distributed:
        torch.distributed.barrier()
    elif args.barrier_dir is not None:
        if args.barrier_participant is None:
            raise ValueError("--barrier-dir requires --barrier-participant")
        args.barrier_dir.mkdir(parents=True, exist_ok=True)
        ready = args.barrier_dir / f"ready-{args.barrier_participant}"
        ready.touch(exist_ok=False)
        deadline = time.monotonic() + 900
        while len(list(args.barrier_dir.glob("ready-*"))) < args.barrier_participants:
            if time.monotonic() >= deadline:
                raise TimeoutError("concurrent training benchmark barrier timed out")
            time.sleep(0.1)
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    torch.cuda.reset_peak_memory_stats()
    measured_local_windows = 0
    with _GpuSampler(physical_gpu) as sampler:
        started = time.perf_counter()
        for corpus_index, indices in batches[args.warmup_steps :]:
            measured_local_windows += train_step(corpus_index, indices)
        torch.cuda.synchronize()
        if distributed:
            torch.distributed.barrier()
        elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    global_windows = _reduce_sum(measured_local_windows, device, distributed)
    if rank == 0:
        utilization = [sample["utilization_percent"] for sample in sampler.samples]
        power = [sample["power_watts"] for sample in sampler.samples]
        memory = [sample["memory_used_mib"] for sample in sampler.samples]
        result = {
            "status": "succeeded",
            "layout": args.layout,
            "variant": args.variant,
            "world_size": world_size,
            "precision": "bfloat16 autocast",
            "optimizer": "fused AdamW",
            "compiled": args.compile,
            "barrier_participant": args.barrier_participant,
            "barrier_directory": (
                str(args.barrier_dir.resolve()) if args.barrier_dir is not None else None
            ),
            "barrier_participants": (
                args.barrier_participants if args.barrier_dir is not None else None
            ),
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "global_windows": int(global_windows),
            "elapsed_seconds": elapsed,
            "per_variant_windows_per_second": global_windows / elapsed,
            "estimated_seconds_for_two_equal_variants": (2 * elapsed if distributed else elapsed),
            "corpus_load_seconds": load_seconds,
            "peak_cuda_allocated_bytes_rank0": int(torch.cuda.max_memory_allocated()),
            "gpu_samples_rank0": len(sampler.samples),
            "gpu_utilization_mean_percent_rank0": (
                float(np.mean(utilization)) if utilization else None
            ),
            "gpu_utilization_max_percent_rank0": max(utilization, default=None),
            "gpu_power_mean_watts_rank0": float(np.mean(power)) if power else None,
            "gpu_power_max_watts_rank0": max(power, default=None),
            "gpu_smi_memory_max_mib_rank0": max(memory, default=None),
            "process_user_cpu_seconds_rank0": usage.ru_utime - usage_started.ru_utime,
            "process_system_cpu_seconds_rank0": usage.ru_stime - usage_started.ru_stime,
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(local_rank),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(args.output, flush=True)
    if distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
