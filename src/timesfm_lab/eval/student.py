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
    ) -> None:
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.device = device
        self.univariate = univariate

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

    def predict(
        self, test_data_input: Iterable[dict[str, Any]], batch_size: int | None = None
    ) -> list[Any]:
        from gluonts.model.forecast import QuantileForecast

        effective_batch = batch_size or self.batch_size
        forecasts: list[Any] = []
        for batch in self._batches(test_data_input, effective_batch):
            arrays = [np.atleast_2d(np.asarray(entry["target"], dtype=np.float32)) for entry in batch]
            max_length = max(array.shape[-1] for array in arrays)
            padded = [
                np.pad(array, ((0, 0), (max_length - array.shape[-1], 0)), constant_values=np.nan)
                for array in arrays
            ]
            context = torch.from_numpy(np.stack(padded)).to(self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if self.univariate:
                    count, variates, length = context.shape
                    quantiles = self.model(
                        context.reshape(count * variates, 1, length), self.prediction_length
                    ).reshape(count, variates, self.prediction_length, 9)
                else:
                    quantiles = self.model(context, self.prediction_length)
            predictions = quantiles.float().cpu().numpy()
            for prediction, entry, target in zip(predictions, batch, arrays, strict=True):
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
