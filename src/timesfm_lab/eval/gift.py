"""Pinned TimesFM 3 adapter for the official GIFT-Eval metric protocol.

The optional TimesFM, GIFT-Eval, and GluonTS packages are imported only by the
runtime entry points. Importing :mod:`timesfm_lab.eval.gift` therefore remains
safe in the base development environment.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from timesfm_lab.revisions import QUANTILE_LEVELS

MASE_NAME = "MASE[0.5]"
MWQL_NAME = "mean_weighted_sum_quantile_loss"
REQUIRED_METRICS = ("MASE", "MeanWeightedSumQuantileLoss")


class GiftEvalError(RuntimeError):
    """Raised when the pinned GIFT-Eval smoke contract is not satisfied."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GiftEvalError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GiftEvalError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GiftSmokeSpec:
    """Validated subset of ``teacher_smoke.yaml`` used by the runtime."""

    model_id: str
    model_revision: str
    timesfm_git_revision: str
    dataset_id: str
    dataset_revision: str
    gift_eval_git_revision: str
    dataset_name: str
    term: str
    device: str
    batch_size: int
    max_test_instances: int
    quantiles: tuple[float, ...]
    include_past_covariates: bool
    to_univariate: bool
    return_quantiles: bool
    use_symmetric_averaging: bool
    make_positive: bool
    sort_quantiles: bool
    use_znorm: bool
    padding_mode: str
    expected_instances: int
    expected_variates: int
    expected_prediction_length: int
    reportability_note: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GiftSmokeSpec:
        """Validate and extract the exact non-reportable protocol smoke settings."""

        model = _mapping(config.get("model"), "model")
        data = _mapping(config.get("data"), "data")
        inference = _mapping(config.get("inference"), "inference")
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        expected = _mapping(evaluation.get("expected"), "evaluation.expected")
        upstream = _mapping(config.get("upstream"), "upstream")

        if evaluation.get("protocol") != "gift_eval_gluonts":
            raise GiftEvalError("evaluation.protocol must be 'gift_eval_gluonts'")
        if evaluation.get("reportable") is not False:
            raise GiftEvalError("teacher GIFT smoke must be explicitly non-reportable")
        if evaluation.get("axis", "missing") is not None:
            raise GiftEvalError("official GIFT-Eval smoke requires evaluation.axis: null")
        if evaluation.get("mask_invalid_label") is not True:
            raise GiftEvalError("official GIFT-Eval smoke must mask invalid labels")
        if evaluation.get("allow_nan_forecast") is not False:
            raise GiftEvalError("official GIFT-Eval smoke must reject NaN forecasts")
        if evaluation.get("seasonality") != "gluonts.time_feature.get_seasonality":
            raise GiftEvalError("unexpected GIFT-Eval seasonality function")
        if tuple(evaluation.get("metrics", ())) != REQUIRED_METRICS:
            raise GiftEvalError(f"evaluation.metrics must be {list(REQUIRED_METRICS)!r}")

        quantiles = tuple(float(value) for value in model.get("quantiles", ()))
        if quantiles != QUANTILE_LEVELS:
            raise GiftEvalError("model.quantiles must be the official levels 0.1 through 0.9")
        if model.get("dtype") != "float32":
            raise GiftEvalError("reference GIFT-Eval smoke must use float32")
        if model.get("use_variate_attention") is not True:
            raise GiftEvalError("reference GIFT-Eval smoke requires variate attention")
        if model.get("use_sdpa") is not True:
            raise GiftEvalError("reference GIFT-Eval smoke requires checkpoint SDPA")
        if data.get("to_univariate") is not False:
            raise GiftEvalError("teacher GIFT smoke must exercise native multivariate mode")

        required_inference = {
            "return_quantiles": True,
            "use_symmetric_averaging": True,
            "make_positive": True,
            "sort_quantiles": True,
            "use_znorm": False,
            "padding_mode": "none",
        }
        for name, required_value in required_inference.items():
            if inference.get(name) != required_value:
                raise GiftEvalError(
                    f"inference.{name} must be {required_value!r} for the official smoke"
                )

        expected_instances = _positive_int(
            expected["dataset_instances"], "evaluation.expected.dataset_instances"
        )
        expected_variates = _positive_int(
            expected["target_variates"], "evaluation.expected.target_variates"
        )
        expected_prediction_length = _positive_int(
            expected["prediction_length"], "evaluation.expected.prediction_length"
        )
        if expected.get("per_instance_point_shape") != [
            expected_variates,
            expected_prediction_length,
        ]:
            raise GiftEvalError("unexpected evaluation.expected.per_instance_point_shape")
        if expected.get("per_instance_quantile_shape") != [
            expected_variates,
            expected_prediction_length,
            len(quantiles),
        ]:
            raise GiftEvalError("unexpected evaluation.expected.per_instance_quantile_shape")

        return cls(
            model_id=str(model["repo_id"]),
            model_revision=str(config["model_revision"]),
            timesfm_git_revision=str(upstream["timesfm_git_revision"]),
            dataset_id=str(data["repo_id"]),
            dataset_revision=str(config["dataset_revision"]),
            gift_eval_git_revision=str(upstream["gift_eval_git_revision"]),
            dataset_name=str(data["dataset"]),
            term=str(data["term"]),
            device=str(model["device"]),
            batch_size=_positive_int(model["per_core_batch_size"], "model.per_core_batch_size"),
            max_test_instances=_positive_int(
                data["max_test_instances"], "data.max_test_instances"
            ),
            quantiles=quantiles,
            include_past_covariates=bool(
                data.get("include_past_dynamic_real_as_past_only_covariates", True)
            ),
            to_univariate=bool(data.get("to_univariate", False)),
            return_quantiles=bool(inference["return_quantiles"]),
            use_symmetric_averaging=bool(inference["use_symmetric_averaging"]),
            make_positive=bool(inference["make_positive"]),
            sort_quantiles=bool(inference["sort_quantiles"]),
            use_znorm=bool(inference["use_znorm"]),
            padding_mode=str(inference["padding_mode"]),
            expected_instances=expected_instances,
            expected_variates=expected_variates,
            expected_prediction_length=expected_prediction_length,
            reportability_note=str(evaluation["reportability_note"]),
        )


@dataclass(frozen=True, slots=True)
class GiftSmokeResult:
    """Metrics and compact evidence suitable for embedding in a ``RunRecord``."""

    metrics: dict[str, float]
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LimitedTestData:
    """A public-iterator-only view over the first ``limit`` test pairs.

    GluonTS ``TestData`` at the pinned revision is not subscriptable. This view
    deliberately consumes only ``input``, ``label``, and ``__iter__`` instead
    of relying on the broken ``test_data[0]`` pattern in the upstream runner.
    """

    source: Any
    limit: int

    def __post_init__(self) -> None:
        _positive_int(self.limit, "limit")

    def __len__(self) -> int:
        try:
            return min(len(self.source), self.limit)
        except TypeError:
            return self.limit

    def __iter__(self) -> Iterator[Any]:
        return islice(iter(self.source), self.limit)

    @property
    def input(self) -> Iterator[Any]:
        """Iterate over model inputs through the public ``TestData.input`` API."""

        return islice(iter(self.source.input), self.limit)

    @property
    def label(self) -> Iterator[Any]:
        """Iterate over labels through the public ``TestData.label`` API."""

        return islice(iter(self.source.label), self.limit)


def to_gluonts_quantile_layout(
    quantiles: npt.ArrayLike,
    *,
    prediction_length: int,
    quantile_count: int = len(QUANTILE_LEVELS),
) -> npt.NDArray[Any]:
    """Convert TimesFM ``(V,H,Q)``/``(H,Q)`` output to GluonTS layout."""

    horizon = _positive_int(prediction_length, "prediction_length")
    array = np.asarray(quantiles)
    if array.ndim == 3:
        if array.shape[1] < horizon:
            raise GiftEvalError(
                f"quantile horizon {array.shape[1]} is shorter than requested {horizon}"
            )
        if array.shape[2] != quantile_count:
            raise GiftEvalError(
                f"expected {quantile_count} quantiles, received {array.shape[2]}"
            )
        return np.transpose(array[:, :horizon, :], (2, 1, 0))
    if array.ndim == 2:
        if array.shape[0] < horizon:
            raise GiftEvalError(
                f"quantile horizon {array.shape[0]} is shorter than requested {horizon}"
            )
        if array.shape[1] != quantile_count:
            raise GiftEvalError(
                f"expected {quantile_count} quantiles, received {array.shape[1]}"
            )
        return array[:horizon, :].T
    raise GiftEvalError(
        f"TimesFM quantiles must have shape (H,Q) or (V,H,Q), received {array.shape}"
    )


def _batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def _quantile_forecast_factory() -> Callable[..., Any]:
    try:
        module = importlib.import_module("gluonts.model.forecast")
    except ImportError as error:
        raise GiftEvalError(
            "GluonTS is unavailable; install the pinned GIFT-Eval environment"
        ) from error
    return module.QuantileForecast


class TimesFm3GiftPredictor:
    """GluonTS-compatible predictor preserving Google's TimesFM 3 semantics."""

    def __init__(
        self,
        forecaster: Any,
        *,
        prediction_length: int,
        batch_size: int,
        quantiles: Sequence[float] = QUANTILE_LEVELS,
        include_past_covariates: bool = True,
        return_quantiles: bool = True,
        use_symmetric_averaging: bool = True,
        make_positive: bool = True,
        sort_quantiles: bool = True,
        use_znorm: bool = False,
        padding_mode: str = "none",
        univariate: bool = False,
        forecast_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.prediction_length = _positive_int(prediction_length, "prediction_length")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.quantiles = tuple(float(value) for value in quantiles)
        if self.quantiles != QUANTILE_LEVELS:
            raise GiftEvalError("predictor requires the official nine quantiles")
        if not return_quantiles:
            raise GiftEvalError("GIFT-Eval probabilistic metrics require quantile output")
        self.include_past_covariates = include_past_covariates
        self.return_quantiles = return_quantiles
        self.use_symmetric_averaging = use_symmetric_averaging
        self.make_positive = make_positive
        self.sort_quantiles = sort_quantiles
        self.use_znorm = use_znorm
        self.padding_mode = padding_mode
        self.univariate = univariate
        self.forecast_factory = forecast_factory

    def predict(
        self,
        test_data_input: Iterable[Mapping[str, Any]],
        batch_size: int | None = None,
    ) -> list[Any]:
        """Forecast an iterable of GluonTS data entries in deterministic batches."""

        effective_batch_size = self.batch_size if batch_size is None else _positive_int(
            batch_size, "batch_size"
        )
        factory = self.forecast_factory or _quantile_forecast_factory()
        forecasts: list[Any] = []
        for batch in _batches(test_data_input, effective_batch_size):
            targets = [np.asarray(entry["target"], dtype=np.float32) for entry in batch]
            if self.include_past_covariates:
                past_covariates = [
                    (
                        np.asarray(entry["past_feat_dynamic_real"], dtype=np.float32)
                        if entry.get("past_feat_dynamic_real") is not None
                        else None
                    )
                    for entry in batch
                ]
            else:
                past_covariates = [None] * len(batch)

            outputs = list(
                self.forecaster.predict_batch(
                    contexts=targets,
                    horizon=self.prediction_length,
                    past_only_covariates=past_covariates,
                    return_quantiles=self.return_quantiles,
                    use_symmetric_averaging=self.use_symmetric_averaging,
                    make_positive=self.make_positive,
                    sort_quantiles=self.sort_quantiles,
                    use_znorm=self.use_znorm,
                    padding_mode=self.padding_mode,
                    univariate=self.univariate,
                )
            )
            if len(outputs) != len(batch):
                raise GiftEvalError(
                    f"TimesFM returned {len(outputs)} outputs for a batch of {len(batch)}"
                )

            for output, entry, target in zip(outputs, batch, targets, strict=True):
                if output.quantiles is None:
                    raise GiftEvalError("TimesFM returned no quantiles")
                forecast_array = to_gluonts_quantile_layout(
                    output.quantiles,
                    prediction_length=self.prediction_length,
                    quantile_count=len(self.quantiles),
                )
                forecasts.append(
                    factory(
                        forecast_arrays=forecast_array,
                        forecast_keys=[str(value) for value in self.quantiles],
                        start_date=entry["start"] + target.shape[-1],
                    )
                )
        return forecasts


def _optional_module(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise GiftEvalError(f"{name} is unavailable; {install_hint}") from error


def _metric_scalar(frame: Any, name: str) -> float:
    try:
        column = frame[name]
        value = column.iloc[0] if hasattr(column, "iloc") else column[0]
        scalar = float(value)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise GiftEvalError(f"evaluation did not return scalar metric {name!r}") from error
    if not np.isfinite(scalar):
        raise GiftEvalError(f"metric {name!r} is non-finite")
    return scalar


def run_gift_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    univariate: bool = False,
) -> GiftSmokeResult:
    """Run the two-instance, explicitly non-reportable GIFT-Eval smoke."""

    spec = GiftSmokeSpec.from_config(config)
    if data_root is not None:
        os.environ["GIFT_EVAL"] = str(data_root.resolve())

    gift_data = _optional_module(
        "gift_eval.data", "install GIFT-Eval at the revision recorded in the config"
    )
    gluonts_metrics = _optional_module(
        "gluonts.ev.metrics", "install the pinned GIFT-Eval environment"
    )
    gluonts_model = _optional_module(
        "gluonts.model", "install the pinned GIFT-Eval environment"
    )
    gluonts_time = _optional_module(
        "gluonts.time_feature", "install the pinned GIFT-Eval environment"
    )
    timesfm3 = _optional_module(
        "timesfm3", "install TimesFM from the revision recorded in the config"
    )

    dataset = gift_data.Dataset(
        name=spec.dataset_name,
        term=spec.term,
        to_univariate=spec.to_univariate,
    )
    if int(dataset.prediction_length) != spec.expected_prediction_length:
        raise GiftEvalError(
            "dataset prediction length differs from the committed smoke expectation"
        )
    if int(dataset.target_dim) != spec.expected_variates:
        raise GiftEvalError("dataset target dimension differs from the smoke expectation")

    test_data = LimitedTestData(dataset.test_data, spec.max_test_instances)
    if len(test_data) != spec.expected_instances:
        raise GiftEvalError(
            f"expected {spec.expected_instances} test instances, received {len(test_data)}"
        )

    forecaster = timesfm3.TimesFM3Evaluator(
        timesfm3.ModelConfig(
            checkpoint_path=spec.model_id,
            revision=spec.model_revision,
            device=spec.device,
            per_core_batch_size=spec.batch_size,
        )
    )
    if tuple(float(value) for value in forecaster.config.quantiles) != spec.quantiles:
        raise GiftEvalError("loaded checkpoint quantiles differ from the pinned contract")
    if forecaster.model.use_variate_attention is not True:
        raise GiftEvalError("loaded checkpoint has variate attention disabled")
    transformer_config = forecaster.model.transformer_config.transformer
    if transformer_config.use_sdpa is not True:
        raise GiftEvalError("loaded checkpoint has SDPA disabled")
    first_parameter = next(forecaster.model.parameters())
    if str(first_parameter.dtype) != "torch.float32":
        raise GiftEvalError(
            f"loaded checkpoint dtype is {first_parameter.dtype}, expected torch.float32"
        )
    predictor = TimesFm3GiftPredictor(
        forecaster,
        prediction_length=spec.expected_prediction_length,
        batch_size=spec.batch_size,
        quantiles=spec.quantiles,
        include_past_covariates=spec.include_past_covariates,
        return_quantiles=spec.return_quantiles,
        use_symmetric_averaging=spec.use_symmetric_averaging,
        make_positive=spec.make_positive,
        sort_quantiles=spec.sort_quantiles,
        use_znorm=spec.use_znorm,
        padding_mode=spec.padding_mode,
        univariate=univariate,
    )
    metrics = [
        gluonts_metrics.MASE(),
        gluonts_metrics.MeanWeightedSumQuantileLoss(
            quantile_levels=list(spec.quantiles)
        ),
    ]
    result = gluonts_model.evaluate_model(
        predictor,
        test_data=test_data,
        metrics=metrics,
        batch_size=spec.batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=gluonts_time.get_seasonality(dataset.freq),
    )
    metric_values = {
        MASE_NAME: _metric_scalar(result, MASE_NAME),
        MWQL_NAME: _metric_scalar(result, MWQL_NAME),
    }
    evidence = {
        "protocol_smoke_only": True,
        "reportable_benchmark": False,
        "reportability_note": spec.reportability_note,
        "upstream": {
            "timesfm_git_revision": spec.timesfm_git_revision,
            "gift_eval_git_revision": spec.gift_eval_git_revision,
        },
        "dataset": {
            "id": spec.dataset_id,
            "revision": spec.dataset_revision,
            "configuration": f"{spec.dataset_name}/{spec.term}",
            "instances": len(test_data),
            "target_variates": spec.expected_variates,
            "prediction_length": spec.expected_prediction_length,
            "dataset_to_univariate": spec.to_univariate,
            "inference_mode": "univariate" if univariate else "multivariate",
        },
        "model": {
            "id": spec.model_id,
            "revision": spec.model_revision,
            "device": spec.device,
            "dtype": str(first_parameter.dtype),
            "quantiles": list(spec.quantiles),
            "use_variate_attention": True,
            "use_sdpa": True,
        },
        "evaluation": {
            "adapter": "timesfm3.TimesFM3Evaluator",
            "axis": None,
            "mask_invalid_label": True,
            "allow_nan_forecast": False,
            "quantile_layout": "TimesFM (V,H,Q) -> GluonTS (Q,H,V)",
            "metrics": list(REQUIRED_METRICS),
        },
    }
    return GiftSmokeResult(metrics=metric_values, evidence=evidence)
