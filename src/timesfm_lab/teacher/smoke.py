"""Small correctness probe for the official TimesFM 3.0 inference path."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from timesfm_lab.revisions import QUANTILE_LEVELS


class TeacherSmokeError(RuntimeError):
    """Raised when the pinned reference model violates the smoke contract."""


def deterministic_context(
    *, num_variates: int, context_length: int, seed: int
) -> npt.NDArray[np.float32]:
    """Build a positive, nontrivial context for protocol smoke testing only.

    The values are deterministic but are not training data and must not be reported as a
    benchmark. Each channel combines a trend, two periods, and seeded low-amplitude noise.
    """

    if num_variates < 1:
        raise ValueError("num_variates must be positive")
    if context_length < 32:
        raise ValueError("context_length must be at least one 32-step patch")
    rng = np.random.default_rng(seed)
    time = np.arange(context_length, dtype=np.float32)
    channels = []
    for variate in range(num_variates):
        phase = np.float32(variate * 0.37)
        signal = (
            np.float32(10.0 + variate)
            + np.float32(0.015 * (variate + 1)) * time
            + np.sin(time / np.float32(5.0 + variate) + phase)
            + np.float32(0.25) * np.cos(time / np.float32(13.0 + variate) - phase)
        )
        noise = rng.normal(0.0, 0.01, size=context_length).astype(np.float32)
        channels.append((signal + noise).astype(np.float32))
    return np.stack(channels, axis=0)


def array_digest(value: npt.NDArray[Any]) -> str:
    """Hash shape, dtype, and values without serializing a large output artifact."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def run_reference_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the pinned official evaluator, forecast once, and enforce output invariants."""

    try:
        timesfm3 = importlib.import_module("timesfm3")
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise TeacherSmokeError(
            "TimesFM 3.0 dependencies are unavailable; install the pinned `teacher` extra"
        ) from error

    model_config = config["model"]
    inference = config["inference"]
    expected_quantiles = np.asarray(inference["quantiles"], dtype=np.float64)
    if not np.array_equal(expected_quantiles, np.asarray(QUANTILE_LEVELS)):
        raise TeacherSmokeError("smoke config must use the official nine quantile levels")

    context = deterministic_context(
        num_variates=int(inference["num_variates"]),
        context_length=int(inference["context_length"]),
        seed=int(config["seed"]),
    )
    forecaster_config = timesfm3.ModelConfig(
        checkpoint_path=str(model_config["id"]),
        revision=str(config["model_revision"]),
        device=str(model_config["device"]),
        per_core_batch_size=int(inference["batch_size"]),
    )
    forecaster = timesfm3.TimesFM3Evaluator(forecaster_config)
    outputs = list(
        forecaster.predict_batch(
            contexts=[context],
            horizon=int(inference["horizon"]),
            return_quantiles=True,
            use_symmetric_averaging=bool(inference["use_symmetric_averaging"]),
            make_positive=bool(inference["make_positive"]),
            sort_quantiles=True,
            univariate=bool(inference["univariate"]),
        )
    )
    if len(outputs) != 1 or outputs[0].forecast is None or outputs[0].quantiles is None:
        raise TeacherSmokeError("official evaluator returned an incomplete output")

    forecast = np.asarray(outputs[0].forecast)
    quantiles = np.asarray(outputs[0].quantiles)
    horizon = int(inference["horizon"])
    expected_forecast_shape = (context.shape[0], horizon)
    expected_quantile_shape = (*expected_forecast_shape, len(QUANTILE_LEVELS))
    if forecast.shape != expected_forecast_shape:
        raise TeacherSmokeError(
            f"unexpected forecast shape {forecast.shape}; expected {expected_forecast_shape}"
        )
    if quantiles.shape != expected_quantile_shape:
        raise TeacherSmokeError(
            f"unexpected quantile shape {quantiles.shape}; expected {expected_quantile_shape}"
        )
    if not np.isfinite(forecast).all() or not np.isfinite(quantiles).all():
        raise TeacherSmokeError("reference output contains non-finite values")
    if not (np.diff(quantiles, axis=-1) >= 0).all():
        raise TeacherSmokeError("reference quantiles are not ordered")
    if not np.array_equal(forecast, quantiles[..., 4]):
        raise TeacherSmokeError("point forecast does not equal the sorted median quantile")

    model = forecaster.model
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    device = torch.device(str(model_config["device"]))
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    return {
        "protocol_smoke_only": True,
        "reportable_benchmark": False,
        "context": {
            "shape": list(context.shape),
            "dtype": str(context.dtype),
            "sha256": array_digest(context),
        },
        "forecast": {
            "shape": list(forecast.shape),
            "dtype": str(forecast.dtype),
            "finite": True,
            "sha256": array_digest(forecast),
        },
        "quantiles": {
            "shape": list(quantiles.shape),
            "dtype": str(quantiles.dtype),
            "finite": True,
            "ordered": True,
            "levels": expected_quantiles.tolist(),
            "sha256": array_digest(quantiles),
        },
        "model": {
            "parameter_count": parameter_count,
            "loaded_quantiles": list(forecaster.config.quantiles),
            "use_variate_attention": bool(forecaster.config.use_variate_attention),
            "use_sdpa": bool(forecaster.config.use_sdpa),
        },
        "runtime": {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": str(device),
            "gpu_name": str(torch.cuda.get_device_name(device_index)),
            "compute_capability": list(capability),
        },
    }
