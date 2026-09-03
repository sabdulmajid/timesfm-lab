"""Leakage-safe interventions for probing cross-variate information."""

from timesfm_lab.interventions.core import (
    InterventionResult,
    inject_irrelevant_variates,
    permute_auxiliary_channels,
    replace_auxiliary_channels,
    retain_channel_fraction,
    shift_auxiliary_channels,
)

__all__ = [
    "InterventionResult",
    "inject_irrelevant_variates",
    "permute_auxiliary_channels",
    "replace_auxiliary_channels",
    "retain_channel_fraction",
    "shift_auxiliary_channels",
]
