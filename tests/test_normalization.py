from __future__ import annotations

import pytest
import torch

from timesfm_lab.models.normalization import (
    TIMESFM_NORMALIZATION_EPSILON,
    denormalize_forecast,
    masked_mean_and_scale,
    normalize_context,
)


@pytest.mark.parametrize(
    ("values", "expected_scale"),
    [
        ([4.0, 4.0, 4.0, 4.0], 1.0),
        ([-1e-8, 1e-8, -1e-8, 1e-8], 1.0),
        ([-1e-3, 1e-3, -1e-3, 1e-3], 1e-3),
        ([-1e-1, 1e-1, -1e-1, 1e-1], 1e-1),
        ([-2.0, -1.0, 1.0, 2.0], 2.5**0.5),
    ],
)
def test_timesfm_safe_scale_semantics(values: list[float], expected_scale: float) -> None:
    inputs = torch.tensor(values, dtype=torch.float64).view(1, 1, -1)
    normalized, mean, scale, observed = normalize_context(inputs)
    restored = denormalize_forecast(normalized, mean, scale)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(restored).all()
    assert scale.item() == pytest.approx(expected_scale, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(restored[observed], inputs[observed], rtol=1e-12, atol=1e-12)


def test_subunit_scales_are_not_clamped_to_one() -> None:
    inputs = torch.tensor([[[-0.1, 0.1], [-0.001, 0.001]]], dtype=torch.float32)
    _, scale, _ = masked_mean_and_scale(inputs)
    torch.testing.assert_close(scale.squeeze(), torch.tensor([0.1, 0.001]))
    assert (scale < 1.0).all()


def test_missing_values_and_empty_series_remain_finite() -> None:
    inputs = torch.tensor(
        [[[1.0, float("nan"), 3.0, float("inf")], [float("nan")] * 4]]
    )
    normalized, mean, scale, observed = normalize_context(inputs)
    restored = denormalize_forecast(normalized, mean, scale)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(mean).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored[observed], inputs[observed])
    assert mean[0, 0, 0].item() == 2.0
    assert mean[0, 1, 0].item() == 0.0
    assert scale[0, 1, 0].item() == 1.0


def test_large_bitcoin_like_values_do_not_overflow() -> None:
    inputs = torch.tensor(
        [[[1.0e18, 1.2e18, 0.9e18, 1.1e18], [1.0, 2.5046488e13, float("nan"), 3.0]]],
        dtype=torch.float32,
    )
    normalized, mean, scale, observed = normalize_context(inputs)
    restored = denormalize_forecast(normalized, mean, scale)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(mean).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(restored).all()
    maximum_error = (restored[observed] - inputs[observed]).abs().max()
    input_amplitude = inputs[observed].abs().max()
    assert maximum_error / input_amplitude < 2e-6


def test_epsilon_matches_pinned_timesfm_three() -> None:
    assert TIMESFM_NORMALIZATION_EPSILON == 1e-6
