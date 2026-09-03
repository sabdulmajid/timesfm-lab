"""Numerically stable, TimesFM-compatible reversible normalization."""

from __future__ import annotations

import torch
from torch import Tensor

# Matches timesfm/src/timesfm3/util.py at the repository revision pinned by this project.
TIMESFM_NORMALIZATION_EPSILON = 1e-6


def masked_mean_and_scale(
    values: Tensor,
    observed_mask: Tensor | None = None,
    *,
    epsilon: float = TIMESFM_NORMALIZATION_EPSILON,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return population mean, safe standard deviation, and the effective mask.

    Statistics are taken over the last dimension.  The scale calculation factors
    out the largest centered magnitude before squaring, preventing overflow on
    large-valued series.  As in TimesFM-3, only scales below ``epsilon`` are
    replaced by one; ordinary sub-unit scales are preserved.
    """

    if not torch.is_floating_point(values):
        raise TypeError("normalization values must have a floating-point dtype")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if observed_mask is None:
        observed = torch.isfinite(values)
    else:
        if observed_mask.shape != values.shape:
            raise ValueError("observed_mask must have the same shape as values")
        observed = observed_mask.bool() & torch.isfinite(values)

    count = observed.sum(dim=-1, keepdim=True).clamp_min(1)
    zero = torch.zeros((), dtype=values.dtype, device=values.device)

    # Scaling before summation also keeps the mean finite when many values are large.
    absolute_max = torch.where(observed, values.abs(), zero).amax(dim=-1, keepdim=True)
    mean_divisor = torch.where(absolute_max > 0, absolute_max, torch.ones_like(absolute_max))
    scaled_values = torch.where(observed, values / mean_divisor, zero)
    mean = (scaled_values.sum(dim=-1, keepdim=True) / count) * mean_divisor

    centered = torch.where(observed, values - mean, zero)
    amplitude = centered.abs().amax(dim=-1, keepdim=True)
    variance_divisor = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    scaled_variance = (centered / variance_divisor).square().sum(dim=-1, keepdim=True) / count
    raw_scale = variance_divisor * torch.sqrt(scaled_variance)
    scale = torch.where(raw_scale < epsilon, torch.ones_like(raw_scale), raw_scale)
    return mean, scale, observed


def normalize_context(
    values: Tensor,
    observed_mask: Tensor | None = None,
    *,
    epsilon: float = TIMESFM_NORMALIZATION_EPSILON,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize a context tensor and return normalized data, mean, scale, mask."""

    mean, scale, observed = masked_mean_and_scale(
        values, observed_mask, epsilon=epsilon
    )
    normalized = torch.where(observed, (values - mean) / scale, torch.zeros_like(values))
    return normalized, mean, scale, observed


def denormalize_forecast(values: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Invert normalization for either point forecasts or quantile forecasts."""

    while mean.ndim < values.ndim:
        mean = mean.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
    return values * scale + mean
