"""Losses and pipelines for student distillation."""

from .losses import DistillationLoss, pinball_loss, relational_kd_loss

__all__ = ["DistillationLoss", "pinball_loss", "relational_kd_loss"]
