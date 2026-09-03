"""Evaluation adapters for pinned, optional benchmark dependencies."""

from timesfm_lab.eval.gift import (
    GiftEvalError,
    GiftSmokeResult,
    GiftSmokeSpec,
    LimitedTestData,
    TimesFm3GiftPredictor,
    run_gift_smoke,
    to_gluonts_quantile_layout,
)

__all__ = [
    "GiftEvalError",
    "GiftSmokeResult",
    "GiftSmokeSpec",
    "LimitedTestData",
    "TimesFm3GiftPredictor",
    "run_gift_smoke",
    "to_gluonts_quantile_layout",
]
