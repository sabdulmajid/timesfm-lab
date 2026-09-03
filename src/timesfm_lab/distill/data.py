"""Deterministic window sampling and leakage-resistant cache splits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def sample_windows(
    source_lengths: Sequence[int],
    *,
    context_length: int,
    horizon: int,
    total_windows: int,
    seed: int,
    mode: str,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Select ``(row, context_end)`` pairs and describe any population shortfall.

    ``with_replacement`` reproduces the historical sampler. Production caches should
    use ``without_replacement_temporally_spaced``: it allocates windows fairly over
    rows and selects one seeded point from each of evenly spaced temporal strata.
    """
    if context_length <= 0 or horizon <= 0 or total_windows < 0:
        raise ValueError("context_length/horizon must be positive and total_windows nonnegative")
    eligible = [
        (row, length, length - context_length - horizon + 1)
        for row, length in enumerate(source_lengths)
        if length >= context_length + horizon
    ]
    if not eligible:
        raise ValueError("dataset has no rows long enough for the configured window")

    population = sum(available for _, _, available in eligible)
    if mode == "with_replacement":
        generators = {
            row: np.random.default_rng(seed * 10_000 + row) for row, _, _ in eligible
        }
        windows = []
        for global_index in range(total_windows):
            row, length, _ = eligible[global_index % len(eligible)]
            end = int(generators[row].integers(context_length, length - horizon + 1))
            windows.append((row, end))
        unique = len(set(windows))
        return windows, {
            "mode": mode,
            "requested_windows": total_windows,
            "selected_windows": len(windows),
            "unique_windows": unique,
            "duplicate_windows": len(windows) - unique,
            "available_unique_windows": population,
            "shortfall": 0,
            "temporally_spaced": False,
        }
    if mode != "without_replacement_temporally_spaced":
        raise ValueError(f"unsupported cache sampling mode: {mode}")

    selected_total = min(total_windows, population)
    capacities = {row: available for row, _, available in eligible}
    allocations = {row: 0 for row, _, _ in eligible}
    active = list(capacities)
    row_rng = np.random.default_rng(seed)
    row_rng.shuffle(active)
    remaining = selected_total
    while remaining:
        share, extra = divmod(remaining, len(active))
        proposed = [share + int(position < extra) for position in range(len(active))]
        allocated_now = 0
        next_active: list[int] = []
        for row, amount in zip(active, proposed, strict=True):
            amount = min(amount, capacities[row] - allocations[row])
            allocations[row] += amount
            allocated_now += amount
            if allocations[row] < capacities[row]:
                next_active.append(row)
        remaining -= allocated_now
        if remaining and not next_active:
            raise AssertionError("window allocation exhausted before selected_total")
        active = next_active

    by_row: dict[int, list[int]] = {}
    for row, _, available in eligible:
        count = allocations[row]
        if not count:
            continue
        rng = np.random.default_rng(seed * 10_000 + row)
        # Disjoint temporal strata guarantee uniqueness while spreading retained
        # windows across the full usable history instead of clustering overlaps.
        lower = np.floor(np.arange(count, dtype=np.float64) * available / count).astype(np.int64)
        upper = np.floor(
            np.arange(1, count + 1, dtype=np.float64) * available / count
        ).astype(np.int64)
        offsets = np.asarray(
            [rng.integers(lo, hi) for lo, hi in zip(lower, upper, strict=True)],
            dtype=np.int64,
        )
        by_row[row] = (offsets + context_length).tolist()

    # Interleave rows so modulo sharding gives each worker comparable coverage.
    windows = []
    longest = max((len(ends) for ends in by_row.values()), default=0)
    ordered_rows = list(allocations)
    for position in range(longest):
        for row in ordered_rows:
            ends = by_row.get(row, [])
            if position < len(ends):
                windows.append((row, ends[position]))
    assert len(windows) == selected_total
    assert len(set(windows)) == selected_total
    return windows, {
        "mode": mode,
        "requested_windows": total_windows,
        "selected_windows": selected_total,
        "unique_windows": selected_total,
        "duplicate_windows": 0,
        "available_unique_windows": population,
        "shortfall": total_windows - selected_total,
        "temporally_spaced": True,
        "per_row_selected": {str(row): count for row, count in allocations.items()},
    }


def split_cache_indices(
    row_index: npt.NDArray[Any],
    context_end: npt.NDArray[Any],
    *,
    context_length: int,
    horizon: int,
    validation_fraction: float,
    seed: int,
    mode: str,
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Split cached windows, optionally with strict source/time separation."""
    rows = np.asarray(row_index)
    ends = np.asarray(context_end)
    if rows.ndim != 1 or ends.shape != rows.shape or not len(rows):
        raise ValueError("row_index and context_end must be nonempty matching vectors")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)

    if mode == "random_windows":
        order = rng.permutation(len(rows))
        count = max(1, round(len(rows) * validation_fraction))
        split = {"validation": order[:count], "training": order[count:]}
        return split, {
            "mode": mode,
            "leakage_control": "none; historical compatibility only",
            "excluded_embargo_windows": 0,
        }
    if mode not in {"series_or_time", "held_out_series", "blocked_time"}:
        raise ValueError(f"unsupported validation split mode: {mode}")

    unique_rows = np.unique(rows)
    use_series = mode in {"series_or_time", "held_out_series"} and len(unique_rows) > 1
    if mode == "held_out_series" and len(unique_rows) < 2:
        raise ValueError("held_out_series requires at least two source rows")
    if use_series:
        shuffled = rng.permutation(unique_rows)
        target = max(1, round(len(rows) * validation_fraction))
        validation_rows: list[int] = []
        validation_count = 0
        for row in shuffled:
            row_count = int(np.count_nonzero(rows == row))
            if validation_rows and abs(validation_count - target) <= abs(
                validation_count + row_count - target
            ):
                continue
            if validation_count + row_count >= len(rows):
                continue
            validation_rows.append(int(row))
            validation_count += row_count
        if not validation_rows:
            validation_rows = [int(shuffled[0])]
        validation_mask = np.isin(rows, validation_rows)
        split = {
            "validation": np.flatnonzero(validation_mask),
            "training": np.flatnonzero(~validation_mask),
        }
        return split, {
            "mode": "held_out_series",
            "leakage_control": "validation source rows are absent from training",
            "validation_rows": validation_rows,
            "excluded_embargo_windows": 0,
        }

    validation_parts: list[npt.NDArray[Any]] = []
    training_parts: list[npt.NDArray[Any]] = []
    excluded = 0
    boundaries: dict[str, int] = {}
    for row in unique_rows:
        indices = np.flatnonzero(rows == row)
        ordered = indices[np.argsort(ends[indices], kind="stable")]
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation = ordered[-validation_count:]
        validation_start = int(ends[validation].min()) - context_length
        # Half-open training [end-context, end+horizon) must end no later than
        # the earliest validation context starts. Intervening windows are embargoed.
        training = ordered[ends[ordered] + horizon <= validation_start]
        validation_parts.append(validation)
        training_parts.append(training)
        excluded += len(ordered) - len(validation) - len(training)
        boundaries[str(int(row))] = validation_start
    training = np.concatenate(training_parts)
    validation = np.concatenate(validation_parts)
    if not len(training):
        raise ValueError(
            "blocked-time split left no training windows; reduce validation_fraction or "
            "cache more widely separated history"
        )
    return {"training": training, "validation": validation}, {
        "mode": "blocked_time",
        "leakage_control": (
            "training context+target intervals end before the earliest validation context"
        ),
        "validation_context_start_by_row": boundaries,
        "excluded_embargo_windows": excluded,
    }
