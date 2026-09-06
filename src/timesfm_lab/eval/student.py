"""GluonTS adapter for the native multivariate student."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from timesfm_lab.eval.gift import to_gluonts_quantile_layout


class StudentGiftPredictor:
    prediction_length: int

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        prediction_length: int,
        batch_size: int,
        device: torch.device,
        univariate: bool = False,
        include_past_covariates: bool = False,
        use_symmetric_averaging: bool = False,
        make_positive: bool = False,
        sort_quantiles: bool = True,
        shared_context_limit: int | None = None,
    ) -> None:
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.device = device
        self.univariate = univariate
        self.include_past_covariates = include_past_covariates
        self.use_symmetric_averaging = use_symmetric_averaging
        self.make_positive = make_positive
        self.sort_quantiles = sort_quantiles
        model_limit = int(model.config.max_context)
        self.context_limit = (
            model_limit
            if shared_context_limit is None
            else min(model_limit, int(shared_context_limit))
        )
        if self.context_limit <= 0:
            raise ValueError("shared_context_limit must be positive")

    @staticmethod
    def _batches(values: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for value in values:
            batch.append(value)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _linear_interpolation(values: np.ndarray) -> np.ndarray:
        """Match the pinned TimesFM forecaster's per-row NaN interpolation."""

        result = np.asarray(values, dtype=np.float32).copy()
        for row in result:
            missing = np.isnan(row)
            if not missing.any():
                continue
            valid_indices = np.flatnonzero(~missing)
            if valid_indices.size:
                row[missing] = np.interp(
                    np.flatnonzero(missing), valid_indices, row[valid_indices]
                )
            else:
                row[missing] = 0.0
        return result

    def _prepare_entry(
        self, entry: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        original = np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
        target = original.copy()
        covariates_raw = entry.get("past_feat_dynamic_real")
        covariates = (
            np.atleast_2d(np.asarray(covariates_raw, dtype=np.float32)).copy()
            if self.include_past_covariates
            and not self.univariate
            and covariates_raw is not None
            else None
        )
        all_missing = np.isnan(target).all(axis=0)
        first_valid = target.shape[-1] if all_missing.all() else int(np.argmax(~all_missing))
        if 0 < first_valid < target.shape[-1]:
            target = target[:, first_valid:]
            if covariates is not None:
                covariates = covariates[:, first_valid:]
        elif first_valid == target.shape[-1] and target.shape[-1]:
            target = np.zeros_like(target)
        target = self._linear_interpolation(target)
        if covariates is not None:
            if covariates.shape[-1] != target.shape[-1]:
                raise ValueError(
                    "past_feat_dynamic_real must align with the target history after trimming"
                )
            covariates = self._linear_interpolation(covariates)
        target = np.ascontiguousarray(target[:, -self.context_limit :])
        if covariates is not None:
            covariates = np.ascontiguousarray(covariates[:, -self.context_limit :])
        return target, covariates, original

    @staticmethod
    def _pad_targets(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        max_length = max(array.shape[-1] for array in arrays)
        values = np.zeros((len(arrays), arrays[0].shape[0], max_length), dtype=np.float32)
        observed = np.zeros_like(values, dtype=np.bool_)
        for index, array in enumerate(arrays):
            length = array.shape[-1]
            values[index, :, -length:] = array
            observed[index, :, -length:] = np.isfinite(array)
        return values, observed

    @staticmethod
    def _pad_covariates(
        arrays: list[np.ndarray | None], max_length: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        channels = max((array.shape[0] for array in arrays if array is not None), default=0)
        if channels == 0:
            return None, None
        values = np.zeros((len(arrays), channels, max_length), dtype=np.float32)
        observed = np.zeros_like(values, dtype=np.bool_)
        for index, array in enumerate(arrays):
            if array is None:
                continue
            length = array.shape[-1]
            values[index, : array.shape[0], -length:] = array
            observed[index, : array.shape[0], -length:] = np.isfinite(array)
        return values, observed

    def _model_call(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        covariates: torch.Tensor | None,
        covariate_observed: torch.Tensor | None,
    ) -> torch.Tensor:
        output = self.model(
            context,
            self.prediction_length,
            observed_mask=observed,
            past_only_covariates=covariates,
            past_only_observed_mask=covariate_observed,
        )
        return torch.sort(output, dim=-1).values if self.sort_quantiles else output

    def _forecast_target_chunks(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        covariates: torch.Tensor | None,
        covariate_observed: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mirror the pinned evaluator's deterministic 32-variate packing."""

        maximum_variates = 32
        target_count = int(context.shape[1])
        covariate_count = 0 if covariates is None else int(covariates.shape[1])

        # The official evaluator only enters its chunk-and-pad path when the
        # complete request exceeds the 32-variate limit. Padding an already
        # fitting request with duplicated targets changes variate attention and
        # therefore changes its forecast.
        if target_count + covariate_count <= maximum_variates:
            positive = self._model_call(
                context, observed, covariates, covariate_observed
            )
            if not self.use_symmetric_averaging:
                return positive
            negative = self._model_call(
                -context,
                observed,
                -covariates if covariates is not None else None,
                covariate_observed,
            )
            return (positive - negative.flip(-1)) / 2

        if covariates is not None and covariates.shape[1] > maximum_variates - 1:
            indices = np.sort(
                np.random.default_rng(42).choice(
                    covariates.shape[1], maximum_variates - 1, replace=False
                )
            )
            index = torch.as_tensor(indices, device=covariates.device)
            covariates = covariates.index_select(1, index)
            assert covariate_observed is not None
            covariate_observed = covariate_observed.index_select(1, index)
        covariate_count = 0 if covariates is None else int(covariates.shape[1])
        targets_per_chunk = maximum_variates - covariate_count
        if targets_per_chunk < 1:
            raise ValueError("past covariates leave no target slot in a 32-variate forward")

        outputs = []
        for start in range(0, target_count, targets_per_chunk):
            stop = min(start + targets_per_chunk, target_count)
            chunk = context[:, start:stop]
            chunk_observed = observed[:, start:stop]
            actual_count = stop - start
            if actual_count < targets_per_chunk:
                needed = targets_per_chunk - actual_count
                repeats = (needed + target_count - 1) // target_count
                chunk = torch.cat(
                    (chunk, context.repeat(1, repeats, 1)[:, :needed]), dim=1
                )
                chunk_observed = torch.cat(
                    (chunk_observed, observed.repeat(1, repeats, 1)[:, :needed]), dim=1
                )
            positive = self._model_call(
                chunk, chunk_observed, covariates, covariate_observed
            )
            if self.use_symmetric_averaging:
                negative = self._model_call(
                    -chunk,
                    chunk_observed,
                    -covariates if covariates is not None else None,
                    covariate_observed,
                )
                prediction = (positive - negative.flip(-1)) / 2
            else:
                prediction = positive
            outputs.append(prediction[:, :actual_count])
        return torch.cat(outputs, dim=1)

    def predict(
        self, test_data_input: Iterable[dict[str, Any]], batch_size: int | None = None
    ) -> list[Any]:
        from gluonts.model.forecast import QuantileForecast

        effective_batch = batch_size or self.batch_size
        forecasts: list[Any] = []
        for batch in self._batches(test_data_input, effective_batch):
            prepared = [self._prepare_entry(entry) for entry in batch]
            arrays = [item[0] for item in prepared]
            covariate_arrays = [item[1] for item in prepared]
            original_arrays = [item[2] for item in prepared]
            padded, observed_array = self._pad_targets(arrays)
            padded_covariates, covariate_observed_array = self._pad_covariates(
                covariate_arrays, padded.shape[-1]
            )
            context = torch.from_numpy(padded).to(self.device)
            observed = torch.from_numpy(observed_array).to(self.device)
            covariates = (
                torch.from_numpy(padded_covariates).to(self.device)
                if padded_covariates is not None
                else None
            )
            covariate_observed = (
                torch.from_numpy(covariate_observed_array).to(self.device)
                if covariate_observed_array is not None
                else None
            )
            count, variates, length = context.shape
            if self.univariate:
                context = context.reshape(count * variates, 1, length)
                observed = observed.reshape(count * variates, 1, length)
                covariates = None
                covariate_observed = None
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                quantiles = self._forecast_target_chunks(
                    context, observed, covariates, covariate_observed
                )
                if self.univariate:
                    quantiles = quantiles.reshape(
                        count, variates, self.prediction_length, 9
                    )
            predictions = quantiles.float().cpu().numpy()
            if self.make_positive:
                for prediction, original in zip(predictions, original_arrays, strict=True):
                    for variate_index, row in enumerate(original):
                        valid = row[np.isfinite(row)]
                        if valid.size and np.all(valid >= 0):
                            prediction[variate_index] = np.maximum(
                                prediction[variate_index], 0.0
                            )
            for prediction, entry, target in zip(predictions, batch, original_arrays, strict=True):
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=to_gluonts_quantile_layout(
                            prediction,
                            prediction_length=self.prediction_length,
                            quantile_count=9,
                        ),
                        forecast_keys=[str(value / 10) for value in range(1, 10)],
                        start_date=entry["start"] + target.shape[-1],
                    )
                )
        return forecasts
