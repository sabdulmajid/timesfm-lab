"""Losses and pipelines for student distillation."""

from .losses import DistillationLoss, cvrd_loss, output_kd_loss, pinball_loss

__all__ = ["DistillationLoss", "cvrd_loss", "output_kd_loss", "pinball_loss"]
