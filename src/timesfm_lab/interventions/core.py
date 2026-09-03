"""Deterministic interventions on multivariate observed histories.

All public functions accept values in ``(variates, time)`` order and return new arrays;
inputs are never mutated. The target history is retained byte-for-byte in its output row.
Validity masks describe structural padding introduced by an intervention, not missingness
already present in the caller's data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

NumericArray: TypeAlias = npt.NDArray[Any]
MetadataValue: TypeAlias = str | int | float | bool | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """Result shared by all cross-variate interventions.

    ``target_index`` is the target's row in ``values``. ``selected_indices`` is interpreted
    in the index space named by ``metadata["selected_index_space"]``: input channels for
    permutation/retention/shift and pool channels for replacement/injection.
    """

    values: NumericArray
    validity_mask: npt.NDArray[np.bool_]
    selected_indices: tuple[int, ...]
    target_index: int
    metadata: dict[str, MetadataValue]


def _validate_values(values: npt.ArrayLike, *, name: str = "values") -> NumericArray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (variates, time); got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have at least one variate and one time step")
    if not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have an integer or floating dtype; got {array.dtype}")
    return array


def _validate_integer(value: int, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def _validate_target(target_index: int, num_variates: int) -> int:
    target = _validate_integer(target_index, name="target_index", minimum=0)
    if target >= num_variates:
        raise IndexError(
            f"target_index {target} is out of bounds for {num_variates} variates"
        )
    return target


def _validate_seed(seed: int) -> int:
    return _validate_integer(seed, name="seed", minimum=0)


def _validate_pool(
    pool: npt.ArrayLike,
    *,
    time_steps: int,
    dtype: np.dtype[Any],
    name: str,
) -> NumericArray:
    array = _validate_values(pool, name=name)
    if array.shape[1] != time_steps:
        raise ValueError(
            f"{name} must contain exactly the {time_steps} observed time steps; "
            f"got {array.shape[1]}"
        )
    if array.dtype != dtype:
        raise TypeError(f"{name} dtype {array.dtype} must match values dtype {dtype}")
    return array


def _all_valid(shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    return np.ones(shape, dtype=np.bool_)


def permute_auxiliary_channels(
    values: npt.ArrayLike,
    *,
    target_index: int,
    seed: int,
) -> InterventionResult:
    """Derange auxiliary channel positions while leaving the target row untouched.

    For two or more auxiliaries, no auxiliary remains in its original row. A sample with
    zero or one auxiliary cannot be effectively deranged and is returned unchanged with
    ``metadata["effective"]`` set to false.
    """

    array = _validate_values(values)
    target = _validate_target(target_index, array.shape[0])
    validated_seed = _validate_seed(seed)
    auxiliaries = tuple(index for index in range(array.shape[0]) if index != target)
    permuted = np.asarray(auxiliaries, dtype=np.int64)

    if len(auxiliaries) > 1:
        rng = np.random.default_rng(validated_seed)
        original = permuted.copy()
        while np.any(permuted == original):
            permuted = rng.permutation(original)

    source_indices = list(range(array.shape[0]))
    for destination, source in zip(auxiliaries, permuted, strict=True):
        source_indices[destination] = int(source)
    selected = tuple(source_indices)
    transformed = array[np.asarray(selected, dtype=np.int64)].copy()
    return InterventionResult(
        values=transformed,
        validity_mask=_all_valid(transformed.shape),
        selected_indices=selected,
        target_index=target,
        metadata={
            "operation": "permute_auxiliary_channels",
            "seed": validated_seed,
            "target_input_index": target,
            "target_output_index": target,
            "selected_index_space": "input_channels_by_output_row",
            "effective": len(auxiliaries) > 1,
        },
    )


def replace_auxiliary_channels(
    values: npt.ArrayLike,
    replacement_pool: npt.ArrayLike,
    *,
    target_index: int,
    seed: int,
) -> InterventionResult:
    """Replace every auxiliary with a seeded channel from an observed-history pool.

    The pool must have exactly the same time dimension and dtype as ``values``. Requiring
    this observed-context-only shape prevents accidental access to a future suffix.
    """

    array = _validate_values(values)
    target = _validate_target(target_index, array.shape[0])
    validated_seed = _validate_seed(seed)
    pool = _validate_pool(
        replacement_pool,
        time_steps=array.shape[1],
        dtype=array.dtype,
        name="replacement_pool",
    )
    auxiliaries = tuple(index for index in range(array.shape[0]) if index != target)
    sample_with_replacement = pool.shape[0] < len(auxiliaries)
    rng = np.random.default_rng(validated_seed)
    chosen = rng.choice(
        pool.shape[0],
        size=len(auxiliaries),
        replace=sample_with_replacement,
    )
    selected = tuple(int(index) for index in chosen)

    transformed = array.copy()
    for output_index, pool_index in zip(auxiliaries, selected, strict=True):
        transformed[output_index] = pool[pool_index]
    return InterventionResult(
        values=transformed,
        validity_mask=_all_valid(transformed.shape),
        selected_indices=selected,
        target_index=target,
        metadata={
            "operation": "replace_auxiliary_channels",
            "seed": validated_seed,
            "target_input_index": target,
            "target_output_index": target,
            "auxiliary_output_indices": auxiliaries,
            "selected_index_space": "replacement_pool_channels",
            "sampled_with_replacement": sample_with_replacement,
            "future_values_consumed": False,
        },
    )


def shift_auxiliary_channels(
    values: npt.ArrayLike,
    *,
    target_index: int,
    patch_size: int,
    offset_patches: int,
    fill_value: float = 0.0,
) -> InterventionResult:
    """Shift all auxiliaries by an integer patch offset without wraparound.

    The convention is ``output[t] = input[t - offset_patches * patch_size]``. Positive
    offsets therefore make auxiliary values stale, while negative offsets lead them using
    only values already present in the supplied observed history. Boundary positions are
    filled with ``fill_value`` and marked false in ``validity_mask``. No value outside the
    supplied history is ever read.
    """

    array = _validate_values(values)
    target = _validate_target(target_index, array.shape[0])
    patch = _validate_integer(patch_size, name="patch_size", minimum=1)
    offset = _validate_integer(offset_patches, name="offset_patches")
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float, np.number)):
        raise TypeError("fill_value must be a finite real scalar")
    numeric_fill = float(fill_value)
    if not math.isfinite(numeric_fill):
        raise ValueError("fill_value must be finite")

    transformed = array.copy()
    validity = _all_valid(transformed.shape)
    auxiliaries = tuple(index for index in range(array.shape[0]) if index != target)
    offset_steps = offset * patch
    distance = abs(offset_steps)
    if auxiliaries and distance:
        auxiliary_index = np.asarray(auxiliaries, dtype=np.int64)
        if distance >= array.shape[1]:
            transformed[auxiliary_index, :] = fill_value
            validity[auxiliary_index, :] = False
        elif offset_steps > 0:
            transformed[auxiliary_index, :distance] = fill_value
            transformed[auxiliary_index, distance:] = array[auxiliary_index, :-distance]
            validity[auxiliary_index, :distance] = False
        else:
            transformed[auxiliary_index, -distance:] = fill_value
            transformed[auxiliary_index, :-distance] = array[auxiliary_index, distance:]
            validity[auxiliary_index, -distance:] = False

    return InterventionResult(
        values=transformed,
        validity_mask=validity,
        selected_indices=tuple(range(array.shape[0])),
        target_index=target,
        metadata={
            "operation": "shift_auxiliary_channels",
            "target_input_index": target,
            "target_output_index": target,
            "selected_index_space": "input_channels",
            "patch_size": patch,
            "offset_patches": offset,
            "offset_steps": offset_steps,
            "fill_value": numeric_fill,
            "future_values_consumed": False,
        },
    )


def retain_channel_fraction(
    values: npt.ArrayLike,
    *,
    target_index: int,
    fraction: float,
    seed: int,
) -> InterventionResult:
    """Retain a seeded fraction of all channels, always including the target.

    The retained count is ``max(1, ceil(fraction * V))``. Selected rows remain in their
    original order, and ``target_index`` in the result identifies the target's new row.
    A fraction of zero is the explicit target-only intervention.
    """

    array = _validate_values(values)
    target = _validate_target(target_index, array.shape[0])
    validated_seed = _validate_seed(seed)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float, np.number)):
        raise TypeError("fraction must be a finite number in [0, 1]")
    requested_fraction = float(fraction)
    if not math.isfinite(requested_fraction) or not 0.0 <= requested_fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")

    retained_count = max(1, math.ceil(requested_fraction * array.shape[0]))
    auxiliaries_to_keep = retained_count - 1
    auxiliary_indices = np.asarray(
        [index for index in range(array.shape[0]) if index != target],
        dtype=np.int64,
    )
    if auxiliaries_to_keep == len(auxiliary_indices):
        chosen_auxiliaries = tuple(int(index) for index in auxiliary_indices)
    else:
        rng = np.random.default_rng(validated_seed)
        chosen_auxiliaries = tuple(
            int(index)
            for index in rng.choice(
                auxiliary_indices,
                size=auxiliaries_to_keep,
                replace=False,
            )
        )
    selected = tuple(sorted((target, *chosen_auxiliaries)))
    transformed = array[np.asarray(selected, dtype=np.int64)].copy()
    output_target = selected.index(target)
    return InterventionResult(
        values=transformed,
        validity_mask=_all_valid(transformed.shape),
        selected_indices=selected,
        target_index=output_target,
        metadata={
            "operation": "retain_channel_fraction",
            "seed": validated_seed,
            "target_input_index": target,
            "target_output_index": output_target,
            "selected_index_space": "input_channels",
            "requested_fraction": requested_fraction,
            "effective_fraction": len(selected) / array.shape[0],
        },
    )


def inject_irrelevant_variates(
    values: npt.ArrayLike,
    irrelevant_pool: npt.ArrayLike,
    *,
    count: int,
    target_index: int,
    seed: int,
) -> InterventionResult:
    """Append seeded irrelevant channels drawn from a caller-supplied history pool.

    The pool must contain exactly the same observed time range and dtype as ``values``;
    longer arrays are rejected rather than silently reading or slicing a future suffix.
    Original channels stay in place, so the target index and its values are unchanged.
    """

    array = _validate_values(values)
    target = _validate_target(target_index, array.shape[0])
    injection_count = _validate_integer(count, name="count", minimum=1)
    validated_seed = _validate_seed(seed)
    pool = _validate_pool(
        irrelevant_pool,
        time_steps=array.shape[1],
        dtype=array.dtype,
        name="irrelevant_pool",
    )
    sample_with_replacement = pool.shape[0] < injection_count
    rng = np.random.default_rng(validated_seed)
    chosen = rng.choice(
        pool.shape[0],
        size=injection_count,
        replace=sample_with_replacement,
    )
    selected = tuple(int(index) for index in chosen)
    transformed = np.concatenate(
        (array, pool[np.asarray(selected, dtype=np.int64)]),
        axis=0,
    )
    return InterventionResult(
        values=transformed,
        validity_mask=_all_valid(transformed.shape),
        selected_indices=selected,
        target_index=target,
        metadata={
            "operation": "inject_irrelevant_variates",
            "seed": validated_seed,
            "target_input_index": target,
            "target_output_index": target,
            "selected_index_space": "irrelevant_pool_channels",
            "injected_count": injection_count,
            "sampled_with_replacement": sample_with_replacement,
            "future_values_consumed": False,
        },
    )
