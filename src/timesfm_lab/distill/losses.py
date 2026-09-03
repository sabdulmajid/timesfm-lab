"""Ground-truth, output-distillation, and cross-variate relational objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    safe_values = torch.where(expanded.bool(), values, torch.zeros_like(values))
    return safe_values.sum() / expanded.sum().clamp_min(1)


def pinball_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] != len(QUANTILES):
        raise ValueError("prediction must be target.shape + [9]")
    levels = prediction.new_tensor(QUANTILES)
    error = target.unsqueeze(-1) - prediction
    losses = torch.maximum(levels * error, (levels - 1.0) * error)
    return _masked_mean(losses, mask)


def output_kd_loss(student: Tensor, teacher: Tensor, mask: Tensor | None = None) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("student and teacher quantiles must have identical shapes")
    return _masked_mean(F.smooth_l1_loss(student, teacher, reduction="none"), mask)


def relational_kd_loss(
    student_multivariate: Tensor,
    student_univariate: Tensor,
    teacher_multivariate: Tensor,
    teacher_univariate: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    shapes = {
        student_multivariate.shape,
        student_univariate.shape,
        teacher_multivariate.shape,
        teacher_univariate.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all relational KD tensors must have identical shapes")
    student_response = student_multivariate - student_univariate
    teacher_response = teacher_multivariate - teacher_univariate
    return _masked_mean(
        F.smooth_l1_loss(student_response, teacher_response, reduction="none"), mask
    )


@dataclass(frozen=True, slots=True)
class LossWeights:
    ground_truth: float = 1.0
    output_kd: float = 1.0
    relational_kd: float = 1.0


class DistillationLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        student_multivariate: Tensor,
        target: Tensor,
        *,
        mask: Tensor | None = None,
        teacher_multivariate: Tensor | None = None,
        student_univariate: Tensor | None = None,
        teacher_univariate: Tensor | None = None,
    ) -> dict[str, Tensor]:
        gt = pinball_loss(student_multivariate, target, mask)
        zero = gt.new_zeros(())
        kd = (
            output_kd_loss(student_multivariate, teacher_multivariate, mask)
            if teacher_multivariate is not None
            else zero
        )
        if (student_univariate is None) != (teacher_univariate is None):
            raise ValueError("student_univariate and teacher_univariate must be supplied together")
        rel = (
            relational_kd_loss(
                student_multivariate,
                student_univariate,
                teacher_multivariate,
                teacher_univariate,
                mask,
            )
            if student_univariate is not None and teacher_multivariate is not None
            else zero
        )
        total = (
            self.weights.ground_truth * gt
            + self.weights.output_kd * kd
            + self.weights.relational_kd * rel
        )
        return {"loss": total, "ground_truth": gt, "output_kd": kd, "relational_kd": rel}
