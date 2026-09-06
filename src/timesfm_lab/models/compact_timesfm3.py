"""Trainable compact wrapper around the pinned TimesFM-3 architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .normalization import TIMESFM_NORMALIZATION_EPSILON


@dataclass(frozen=True, slots=True)
class CompactTimesFM3Config:
    architecture: str = "compact_timesfm3"
    input_patch_length: int = 32
    output_patch_length: int = 64
    d_model: int = 384
    ffn_dim: int = 1536
    num_layers: int = 12
    num_heads: int = 6
    max_context: int = 8192
    max_horizon: int = 64
    num_quantiles: int = 9
    normalization_epsilon: float = TIMESFM_NORMALIZATION_EPSILON
    use_stitching: bool = True
    use_linear_detrending: bool = True
    use_iterative_cpm_revin: bool = True
    sort_quantiles: bool = True
    use_rope_var: bool = False

    def __post_init__(self) -> None:
        if self.architecture != "compact_timesfm3":
            raise ValueError(f"unexpected architecture {self.architecture!r}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_context % self.input_patch_length:
            raise ValueError("max_context must be divisible by input_patch_length")
        if self.output_patch_length % self.input_patch_length:
            raise ValueError("output patch length must be a multiple of input patch length")
        if self.max_horizon > self.output_patch_length:
            raise ValueError("the first compact candidate supports one output patch")
        if self.num_quantiles != 9:
            raise ValueError("compact TimesFM-3 is fixed to nine quantiles")


class CompactTimesFM3Student(nn.Module):
    """A ~30M TimesFM-3-shaped student with a differentiable decode path."""

    quantile_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, config: CompactTimesFM3Config) -> None:
        super().__init__()
        from timesfm3.configs import (
            ResidualBlockConfig,
            StackedTransformersConfig,
            TransformerConfig,
        )
        from timesfm3.model import TimesFM3Torch

        self.config = config
        self.backbone = TimesFM3Torch(
            input_patch_len=config.input_patch_length,
            output_patch_len=config.output_patch_length,
            quantiles=list(self.quantile_levels),
            residual_block_config=ResidualBlockConfig(
                hidden_dims=config.d_model,
                output_dims=config.d_model,
                use_bias=False,
                activation="relu",
            ),
            transformer_config=StackedTransformersConfig(
                num_layers=config.num_layers,
                transformer=TransformerConfig(
                    model_dims=config.d_model,
                    hidden_dims=config.ffn_dim,
                    num_heads=config.num_heads,
                    attention_norm="rms",
                    feedforward_norm="rms",
                    qk_norm="rms",
                    use_bias=False,
                    use_rope_seq=True,
                    use_rope_var=config.use_rope_var,
                    ff_activation="relu",
                    deterministic=True,
                    causal_attention=True,
                    use_memory_efficient_attention=True,
                    use_sdpa=True,
                ),
            ),
            use_variate_attention=True,
            use_stitching=config.use_stitching,
            use_linear_detrending=config.use_linear_detrending,
            use_iterative_cpm_revin=config.use_iterative_cpm_revin,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        context: Tensor,
        horizon: int,
        observed_mask: Tensor | None = None,
        past_only_covariates: Tensor | None = None,
        past_only_observed_mask: Tensor | None = None,
    ) -> Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, variate, time]")
        if not 1 <= horizon <= self.config.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.config.max_horizon}]")
        if context.shape[-1] > self.config.max_context:
            context = context[..., -self.config.max_context :]
            if observed_mask is not None:
                observed_mask = observed_mask[..., -self.config.max_context :]
            if past_only_covariates is not None:
                past_only_covariates = past_only_covariates[..., -self.config.max_context :]
            if past_only_observed_mask is not None:
                past_only_observed_mask = past_only_observed_mask[..., -self.config.max_context :]
        if observed_mask is None:
            observed_mask = torch.isfinite(context)
        if observed_mask.shape != context.shape:
            raise ValueError("observed_mask must match context")
        if past_only_covariates is not None:
            if (
                past_only_covariates.ndim != 3
                or past_only_covariates.shape[0] != context.shape[0]
                or past_only_covariates.shape[-1] != context.shape[-1]
            ):
                raise ValueError(
                    "past_only_covariates must have shape [batch, covariate, time] "
                    "with batch/time matching context"
                )
            if past_only_observed_mask is None:
                past_only_observed_mask = torch.isfinite(past_only_covariates)
            if past_only_observed_mask.shape != past_only_covariates.shape:
                raise ValueError("past_only_observed_mask must match past_only_covariates")

        # The upstream method is decorated with torch.no_grad because that package
        # is inference-oriented. Calling the preserved implementation gives the
        # same decode semantics while retaining autograd for student training.
        decode = self.backbone.decode.__wrapped__
        output: Tensor = decode(
            self.backbone,
            context,
            horizon=horizon,
            past_only_covariates=past_only_covariates,
            target_mask=~observed_mask,
            past_only_mask=(
                ~past_only_observed_mask if past_only_observed_mask is not None else None
            ),
        )
        output = output[:, : context.shape[1]]
        return torch.sort(output, dim=-1).values if self.config.sort_quantiles else output
