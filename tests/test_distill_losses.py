from __future__ import annotations

import torch

from timesfm_lab.distill.losses import DistillationLoss, LossWeights, cvrd_loss, output_kd_loss


def _fixture() -> tuple[torch.Tensor, ...]:
    target = torch.tensor([[[1.0, 2.0]]])
    offsets = torch.linspace(-0.4, 0.4, 9).view(1, 1, 1, 9)
    student_mv = target.unsqueeze(-1) + offsets
    student_uv = student_mv + 0.2
    teacher_mv = student_mv - 0.1
    teacher_uv = student_uv + 0.3
    return target, student_mv, student_uv, teacher_mv, teacher_uv


def test_dual_view_contains_both_teacher_output_losses() -> None:
    target, student_mv, student_uv, teacher_mv, teacher_uv = _fixture()
    objective = DistillationLoss(
        LossWeights(ground_truth=1.0, multivariate_kd=1.0, univariate_kd=1.0, cvrd=0.0)
    )
    values = objective(
        student_mv,
        target,
        teacher_multivariate=teacher_mv,
        student_univariate=student_uv,
        teacher_univariate=teacher_uv,
    )
    expected = (
        values["ground_truth"]
        + output_kd_loss(student_mv, teacher_mv)
        + output_kd_loss(student_uv, teacher_uv)
    )
    torch.testing.assert_close(values["loss"], expected)
    assert values["cvrd"].item() > 0


def test_cvrd_adds_only_response_difference_to_dual_view() -> None:
    target, student_mv, student_uv, teacher_mv, teacher_uv = _fixture()
    common = {
        "student_multivariate": student_mv,
        "target": target,
        "teacher_multivariate": teacher_mv,
        "student_univariate": student_uv,
        "teacher_univariate": teacher_uv,
    }
    dual = DistillationLoss(
        LossWeights(ground_truth=1.0, multivariate_kd=1.0, univariate_kd=1.0, cvrd=0.0)
    )(**common)
    cvrd = DistillationLoss(
        LossWeights(ground_truth=1.0, multivariate_kd=1.0, univariate_kd=1.0, cvrd=1.0)
    )(**common)
    response = cvrd_loss(student_mv, student_uv, teacher_mv, teacher_uv)

    torch.testing.assert_close(cvrd["loss"] - dual["loss"], response)
    for key in ("ground_truth", "multivariate_kd", "univariate_kd"):
        torch.testing.assert_close(cvrd[key], dual[key])


def test_legacy_weight_names_map_to_cvrd_names() -> None:
    weights = LossWeights.from_mapping(
        {"ground_truth": 1.0, "output_kd": 2.0, "relational_kd": 3.0}
    )
    assert weights == LossWeights(
        ground_truth=1.0,
        multivariate_kd=2.0,
        univariate_kd=0.0,
        cvrd=3.0,
    )
