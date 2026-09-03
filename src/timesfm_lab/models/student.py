"""Compact factorized temporal/cross-variate forecasting transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class StudentConfig:
    patch_length: int = 32
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    ffn_dim: int = 1536
    max_context: int = 16_384
    max_horizon: int = 256
    num_quantiles: int = 9
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_context % self.patch_length:
            raise ValueError("max_context must be divisible by patch_length")
        if self.num_quantiles != 9:
            raise ValueError("the initial student is fixed to quantiles 0.1 through 0.9")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_value = nn.Linear(d_model, hidden * 2)
        self.output = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.gate_value(inputs).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class FactorizedMixingBlock(nn.Module):
    """Temporal attention per variate followed by variate attention per patch."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        kwargs = {
            "embed_dim": config.d_model,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.temporal_norm = nn.LayerNorm(config.d_model)
        self.temporal_attention = nn.MultiheadAttention(**kwargs)
        self.temporal_ffn_norm = nn.LayerNorm(config.d_model)
        self.temporal_ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)
        self.variate_norm = nn.LayerNorm(config.d_model)
        self.variate_attention = nn.MultiheadAttention(**kwargs)
        self.variate_ffn_norm = nn.LayerNorm(config.d_model)
        self.variate_ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, states: Tensor, valid_patches: Tensor) -> Tensor:
        batch, variates, patches, width = states.shape
        temporal = states.reshape(batch * variates, patches, width)
        temporal_mask = ~valid_patches.reshape(batch * variates, patches)
        normalized = self.temporal_norm(temporal)
        attended, _ = self.temporal_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=temporal_mask,
            need_weights=False,
        )
        temporal = temporal + self.dropout(attended)
        temporal = temporal + self.dropout(self.temporal_ffn(self.temporal_ffn_norm(temporal)))
        states = temporal.reshape(batch, variates, patches, width)
        states = states.masked_fill(~valid_patches.unsqueeze(-1), 0.0)

        variate = states.permute(0, 2, 1, 3).reshape(batch * patches, variates, width)
        variate_mask = ~valid_patches.permute(0, 2, 1).reshape(batch * patches, variates)
        empty_patch = variate_mask.all(dim=-1)
        if empty_patch.any():
            variate_mask = variate_mask.clone()
            variate_mask[empty_patch, 0] = False
        normalized = self.variate_norm(variate)
        attended, _ = self.variate_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=variate_mask,
            need_weights=False,
        )
        variate = variate + self.dropout(attended)
        variate = variate + self.dropout(self.variate_ffn(self.variate_ffn_norm(variate)))
        states = variate.reshape(batch, patches, variates, width).permute(0, 2, 1, 3)
        return states.masked_fill(~valid_patches.unsqueeze(-1), 0.0)


class TimesFMStudent(nn.Module):
    """Non-autoregressive native-multivariate quantile forecaster."""

    quantile_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, config: StudentConfig = StudentConfig()) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(config.patch_length * 2, config.d_model)
        self.time_embedding = nn.Parameter(
            torch.empty(config.max_context // config.patch_length, config.d_model)
        )
        self.layers = nn.ModuleList(
            FactorizedMixingBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.max_horizon * config.num_quantiles)
        nn.init.normal_(self.time_embedding, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _patchify(
        self, context: Tensor, observed_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if context.ndim != 3 or observed_mask.shape != context.shape:
            raise ValueError("context and observed_mask must both have shape [batch, variate, time]")
        if context.shape[-1] > self.config.max_context:
            context = context[..., -self.config.max_context :]
            observed_mask = observed_mask[..., -self.config.max_context :]
        observed_mask = observed_mask.bool() & torch.isfinite(context)
        clean = torch.where(observed_mask, context, torch.zeros_like(context))
        count = observed_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = clean.sum(dim=-1, keepdim=True) / count
        centered = torch.where(observed_mask, clean - mean, torch.zeros_like(clean))
        scale = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(1e-5)
        normalized = torch.where(observed_mask, centered / scale, torch.zeros_like(clean))

        padding = (-normalized.shape[-1]) % self.config.patch_length
        if padding:
            normalized = F.pad(normalized, (padding, 0))
            observed_mask = F.pad(observed_mask, (padding, 0), value=False)
        patches = normalized.unfold(-1, self.config.patch_length, self.config.patch_length)
        mask_patches = observed_mask.unfold(-1, self.config.patch_length, self.config.patch_length)
        valid_patches = mask_patches.any(dim=-1)
        features = torch.cat((patches, mask_patches.to(patches.dtype)), dim=-1)
        return features, valid_patches, mean, scale

    def forward(
        self,
        context: Tensor,
        horizon: int,
        observed_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ordered quantiles with shape ``[batch, variate, horizon, 9]``."""

        if not 1 <= horizon <= self.config.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.config.max_horizon}]")
        if observed_mask is None:
            observed_mask = torch.isfinite(context)
        features, valid_patches, mean, scale = self._patchify(context, observed_mask)
        if (~valid_patches).all(dim=-1).any():
            raise ValueError("each series must contain at least one observed context value")
        patches = features.shape[-2]
        states = self.patch_projection(features)
        states = states + self.time_embedding[-patches:].view(1, 1, patches, -1)
        states = states.masked_fill(~valid_patches.unsqueeze(-1), 0.0)
        for layer in self.layers:
            states = layer(states, valid_patches)

        indices = torch.arange(patches, device=states.device).view(1, 1, patches)
        last_index = torch.where(valid_patches, indices, -1).amax(dim=-1)
        gather_index = last_index[..., None, None].expand(-1, -1, 1, states.shape[-1])
        pooled = states.gather(dim=2, index=gather_index).squeeze(2)
        raw = self.output_head(self.final_norm(pooled))
        raw = raw.view(*raw.shape[:-1], self.config.max_horizon, 9)[..., :horizon, :]
        median = raw[..., 4:5]
        lower_gaps = F.softplus(raw[..., :4])
        upper_gaps = F.softplus(raw[..., 5:])
        lower = median - torch.cumsum(lower_gaps, dim=-1).flip(-1)
        upper = median + torch.cumsum(upper_gaps, dim=-1)
        normalized_quantiles = torch.cat((lower, median, upper), dim=-1)
        return normalized_quantiles * scale.unsqueeze(-1) + mean.unsqueeze(-1)
