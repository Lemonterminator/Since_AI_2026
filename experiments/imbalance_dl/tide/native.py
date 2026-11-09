"""Native PyTorch port of the TensorFlow TiDE reference model.

The port follows ``tide/reference/upstream/tide/models.py``.  It keeps the
upstream layer order, Keras dense orientation, residual blocks, feature
projection, static and categorical embeddings, temporal final decoder, and
linear history skip.  It does not import TensorFlow or the ignored reference
tree at runtime.

Copyright 2026 The Google Research Authors.  This derivative is distributed
under the Apache License, Version 2.0.  See ``tide/LICENSE``.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NativeTiDEOutput:
    """Native forward intermediates used by the task bridge and by tests."""

    prediction: torch.Tensor  # [B, H]
    encoding: torch.Tensor  # [B, width]
    decoder: torch.Tensor  # [B, decoder_output_dim, H]
    future_features: torch.Tensor  # [B, temporal_width, H]
    history_residual: torch.Tensor  # [B, H], normalized-domain linear skip


def _dense(in_features: int, out_features: int) -> nn.Linear:
    """Keras-compatible Dense layer: Glorot uniform kernel and zero bias."""
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLPResidual(nn.Module):
    """TiDE residual block with the upstream operation order.

    ``hidden = relu(lin_a(x))``, ``out = dropout(lin_b(hidden))``,
    ``out = out + lin_res(x)``, then optional Keras layer normalization.
    Dropout is applied only to the ``lin_b`` output.

    ``pre_norm=True`` switches to the training-safe order: normalize the block
    input, activate with GELU, and keep the residual path unnormalized.  A
    post-normalized block fed a common-mode input saturates instead.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-3,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.lin_a = _dense(input_dim, hidden_dim)
        self.lin_b = _dense(hidden_dim, output_dim)
        self.lin_res = _dense(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        normalized = input_dim if pre_norm else output_dim
        self.lnorm = (
            nn.LayerNorm(normalized, eps=layer_norm_eps)
            if layer_norm
            else nn.Identity()
        )
        self.layer_norm = layer_norm
        self.pre_norm = pre_norm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            normed = self.lnorm(inputs) if self.layer_norm else inputs
            hidden = F.gelu(self.lin_a(normed))
            return self.lin_res(inputs) + self.dropout(self.lin_b(hidden))
        hidden = torch.relu(self.lin_a(inputs))
        out = self.dropout(self.lin_b(hidden))
        out = out + self.lin_res(inputs)
        if self.layer_norm:
            return self.lnorm(out)
        return out


def _make_dnn_residual(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    layer_norm: bool = False,
    dropout_rate: float = 0.0,
    layer_norm_eps: float = 1e-3,
    pre_norm: bool = False,
) -> nn.Module:
    """Build the upstream ``_make_dnn_residual`` stack.

    The TensorFlow helper lazily builds its first Dense layer.  This port takes
    the input width explicitly.  With fewer than two hidden dimensions the
    upstream code returns one plain Dense layer, not a residual block.
    """
    dims = list(hidden_dims)
    if len(dims) < 2:
        return _dense(input_dim, dims[-1])
    layers = []
    current = input_dim
    for index, hidden_dim in enumerate(dims[:-1]):
        layers.append(
            MLPResidual(
                current,
                hidden_dim,
                dims[index + 1],
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
                layer_norm_eps=layer_norm_eps,
                pre_norm=pre_norm,
            )
        )
        current = dims[index + 1]
    return nn.Sequential(*layers)


def _as_target(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3 and value.shape[-1] == 1:
        return value[..., 0]
    if value.dim() == 2:
        return value
    raise ValueError("past_target must have shape [B, L] or [B, L, 1]")


class NativeTiDE(nn.Module):
    """Self-contained PyTorch TiDE.

    Args:
        config: TiDE configuration.
        feature_channels: Optional numeric known-feature count for the future
            path. The default is ``config.known_channels``. The task bridge
            passes ``3 * C`` because it augments every known channel with a
            mask and an age.
        history_feature_channels: Optional numeric history-feature count for the
            past path. The default is ``feature_channels``. The task bridge
            passes known history plus observed values, masks, ages, and target
            presence/age as per-step channels.
    """

    def __init__(
        self,
        config,
        *,
        feature_channels: int | None = None,
        history_feature_channels: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.feature_channels = (
            int(config.known_channels) if feature_channels is None
            else int(feature_channels)
        )
        self.history_feature_channels = (
            self.feature_channels if history_feature_channels is None
            else int(history_feature_channels)
        )
        if self.feature_channels < 0 or self.history_feature_channels < 0:
            raise ValueError("feature channel counts must be non-negative")

        self.L = int(config.history)
        self.H = int(config.horizon)
        self.p = int(config.decoder_output_dim)
        self.temporal_width = int(config.temporal_width)
        self.series_embedding = int(config.series_embedding)
        self.layer_norm_eps = float(config.layer_norm_eps)

        categorical_width = (
            len(config.categorical_sizes) * int(config.categorical_embedding)
        )
        self.feature_dim = max(1, self.feature_channels + categorical_width)
        self.history_feature_dim = max(
            1, self.history_feature_channels + categorical_width
        )

        self.time_encoder = _make_dnn_residual(
            self.feature_dim,
            [int(config.time_encoder_width), self.temporal_width],
            layer_norm=config.layer_norm,
            dropout_rate=config.dropout,
            layer_norm_eps=self.layer_norm_eps,
            pre_norm=config.pre_norm,
        )
        self.history_time_encoder = None
        if self.history_feature_channels != self.feature_channels:
            self.history_time_encoder = _make_dnn_residual(
                self.history_feature_dim,
                [int(config.time_encoder_width), self.temporal_width],
                layer_norm=config.layer_norm,
                dropout_rate=config.dropout,
                layer_norm_eps=self.layer_norm_eps,
                pre_norm=config.pre_norm,
            )

        encoder_input_dim = (
            self.L
            + self.temporal_width * self.L
            + self.temporal_width * self.H
            + self.series_embedding
        )
        self.encoder = _make_dnn_residual(
            encoder_input_dim,
            [int(config.width)] * int(config.layers),
            layer_norm=config.layer_norm,
            dropout_rate=config.dropout,
            layer_norm_eps=self.layer_norm_eps,
            pre_norm=config.pre_norm,
        )
        decoder_dims = [int(config.width)] * (int(config.layers) - 1)
        decoder_dims.append(self.p * self.H)
        self.decoder = _make_dnn_residual(
            int(config.width),
            decoder_dims,
            layer_norm=config.layer_norm,
            dropout_rate=config.dropout,
            layer_norm_eps=self.layer_norm_eps,
            pre_norm=config.pre_norm,
        )
        self.final_decoder = MLPResidual(
            self.p + self.temporal_width,
            int(config.final_decoder_hidden),
            1,
            layer_norm=config.layer_norm,
            dropout_rate=config.dropout,
            layer_norm_eps=self.layer_norm_eps,
            pre_norm=config.pre_norm,
        )
        self.linear = _dense(self.L, self.H)

        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(int(size), int(config.categorical_embedding))
                for size in config.categorical_sizes
            ]
        )
        for embedding in self.cat_embs:
            nn.init.uniform_(embedding.weight, -0.05, 0.05)
        self.ts_embs = nn.Embedding(int(config.num_series), self.series_embedding)
        nn.init.uniform_(self.ts_embs.weight, -0.05, 0.05)

        if config.reversible_norm:
            self.affine_weight = nn.Parameter(torch.ones(int(config.num_series)))
            self.affine_bias = nn.Parameter(torch.zeros(int(config.num_series)))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    @property
    def reversible_norm_enabled(self) -> bool:
        return self.affine_weight is not None

    def _known_to_time_major(
        self,
        known: torch.Tensor | None,
        batch_size: int,
        time_steps: int,
        name: str,
        channels: int | None = None,
    ) -> torch.Tensor:
        channels = self.feature_channels if channels is None else channels
        if known is None:
            return torch.zeros(
                batch_size,
                time_steps,
                channels,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
        if known.dim() == 2:
            if known.shape[0] != time_steps:
                raise ValueError(f"{name} must be [T, C] or [B, T, C]")
            known = known.unsqueeze(0)
        if known.dim() != 3:
            raise ValueError(f"{name} must be [T, C] or [B, T, C]")
        if known.shape[1] != time_steps:
            raise ValueError(f"{name} has the wrong time dimension")
        if known.shape[-1] != channels:
            raise ValueError(
                f"{name} has {known.shape[-1]} channels; expected {channels}"
            )
        if known.shape[0] == 1:
            known = known.expand(batch_size, -1, -1)
        elif known.shape[0] != batch_size:
            raise ValueError(f"{name} has the wrong batch dimension")
        return known

    def _categorical_to_batch(
        self,
        categorical: Sequence[torch.Tensor] | None,
        batch_size: int,
        time_steps: int,
    ) -> list[torch.Tensor]:
        if not self.cat_embs:
            if categorical:
                raise ValueError("categorical covariates were provided but not configured")
            return []
        if categorical is None or len(categorical) != len(self.cat_embs):
            raise ValueError("one categorical tensor is required per configured category")
        prepared = []
        for cat, embedding in zip(categorical, self.cat_embs):
            if cat.dim() == 1:
                if cat.shape[0] != time_steps:
                    raise ValueError("categorical covariate has the wrong time dimension")
                cat = cat.unsqueeze(0)
            if cat.dim() != 2:
                raise ValueError("categorical covariate must be [T] or [B, T]")
            if cat.shape[1] != time_steps:
                raise ValueError("categorical covariate has the wrong time dimension")
            if cat.shape[0] == 1:
                cat = cat.expand(batch_size, -1)
            elif cat.shape[0] != batch_size:
                raise ValueError("categorical covariate has the wrong batch dimension")
            if cat.min() < 0 or cat.max() >= embedding.num_embeddings:
                raise ValueError("categorical covariate is out of range")
            prepared.append(cat.long())
        return prepared

    def _assemble_features(
        self,
        numeric: torch.Tensor,
        categorical: Sequence[torch.Tensor],
        *,
        encoder: nn.Module,
        channels: int,
    ) -> torch.Tensor:
        """Return [B, temporal_width, T] in upstream feature-major order."""
        batch_size, time_steps, _ = numeric.shape
        parts = []
        if channels > 0:
            parts.append(numeric.transpose(1, 2))
        for cat, embedding in zip(categorical, self.cat_embs):
            parts.append(embedding(cat).transpose(1, 2))
        if not parts:
            parts.append(
                torch.zeros(
                    batch_size,
                    1,
                    time_steps,
                    device=numeric.device,
                    dtype=numeric.dtype,
                )
            )
        feats = torch.cat(parts, dim=1)  # [B, F, T]
        projected = encoder(feats.transpose(1, 2))  # [B, T, tw]
        return projected.transpose(1, 2)  # [B, tw, T]

    @staticmethod
    def _flatten_feature_major(features: torch.Tensor) -> torch.Tensor:
        # [B, tw, T] -> [B, tw * T]; row-major flatten is feature-major.
        return features.reshape(features.shape[0], -1)

    def _series_ids(
        self, series_ids: torch.Tensor | None, batch_size: int, device
    ) -> torch.Tensor:
        if series_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if series_ids.dim() == 0:
            series_ids = series_ids.expand(batch_size)
        if series_ids.dim() != 1 or series_ids.shape[0] != batch_size:
            raise ValueError("series_ids must be scalar or [B]")
        if series_ids.min() < 0 or series_ids.max() >= self.ts_embs.num_embeddings:
            raise ValueError("series_ids are out of range")
        return series_ids.long()

    def forward(
        self,
        past_target: torch.Tensor,
        known_history: torch.Tensor | None = None,
        known_future: torch.Tensor | None = None,
        *,
        series_ids: torch.Tensor | None = None,
        past_categorical: Sequence[torch.Tensor] | None = None,
        future_categorical: Sequence[torch.Tensor] | None = None,
        encoding_extra: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ):
        """Forecast future target values.

        ``known_history`` accepts ``[B, L, C]`` or a shared ``[L, C]``.
        ``known_future`` accepts ``[B, H, C]`` or a shared ``[H, C]``.  A shared
        covariate tensor is broadcast across the batch, which reproduces the
        upstream global-covariate path.  ``encoding_extra`` is an additive
        ``[B, width]`` side channel to the encoding; it leaves the decoder
        width untouched and defaults to the native path.
        """
        past_target = _as_target(past_target)
        batch_size, history = past_target.shape
        if history != self.L:
            raise ValueError(f"past_target has {history} steps; expected {self.L}")
        device = past_target.device
        dtype = past_target.dtype

        numeric_history = self._known_to_time_major(
            known_history, batch_size, self.L, "known_history",
            channels=self.history_feature_channels,
        ).to(device=device, dtype=dtype)
        numeric_future = self._known_to_time_major(
            known_future, batch_size, self.H, "known_future"
        ).to(device=device, dtype=dtype)
        cat_history = self._categorical_to_batch(
            past_categorical, batch_size, self.L
        )
        cat_future = self._categorical_to_batch(
            future_categorical, batch_size, self.H
        )

        history_encoder = (
            self.time_encoder
            if self.history_time_encoder is None
            else self.history_time_encoder
        )
        past_features = self._flatten_feature_major(
            self._assemble_features(
                numeric_history,
                cat_history,
                encoder=history_encoder,
                channels=self.history_feature_channels,
            )
        )
        future_features = self._assemble_features(
            numeric_future,
            cat_future,
            encoder=self.time_encoder,
            channels=self.feature_channels,
        )
        future_flat = self._flatten_feature_major(future_features)

        ids = self._series_ids(series_ids, batch_size, device)
        ts_embedding = self.ts_embs(ids)

        transformed_target = past_target
        batch_mean = None
        batch_std = None
        if self.reversible_norm_enabled:
            affine_weight = self.affine_weight[ids].unsqueeze(-1)
            affine_bias = self.affine_bias[ids].unsqueeze(-1)
            batch_mean = past_target.mean(dim=1, keepdim=True)
            batch_std = past_target.std(dim=1, keepdim=True, unbiased=False)
            batch_std = torch.where(
                batch_std == 0.0, torch.ones_like(batch_std), batch_std
            )
            transformed_target = (past_target - batch_mean) / batch_std
            transformed_target = affine_weight * transformed_target + affine_bias

        skip = transformed_target
        lags = int(self.config.history_residual_lags) or self.L
        if lags < self.L:
            # Only the most recent lags reach the skip, so distant spike echoes
            # cannot leak into the median.
            skip = torch.cat((transformed_target.new_zeros(batch_size, self.L - lags),
                              transformed_target[:, self.L - lags:]), dim=1)
        if self.config.history_skip_asinh:
            skip = torch.asinh(skip)
        history_residual = self.linear(skip)
        encoder_parts = [
            transformed_target,
            past_features,
            future_flat,
            ts_embedding,
        ]
        encoding = self.encoder(torch.cat(encoder_parts, dim=1))
        if encoding_extra is not None:
            if encoding_extra.shape != encoding.shape:
                raise ValueError(
                    f"encoding_extra has shape {tuple(encoding_extra.shape)}; "
                    f"expected {tuple(encoding.shape)}"
                )
            encoding = encoding + encoding_extra

        decoder_flat = self.decoder(encoding)
        decoder_out = decoder_flat.reshape(batch_size, self.p, self.H)
        final_input = torch.cat((decoder_out, future_features), dim=1)
        temporal = self.final_decoder(final_input.transpose(1, 2)).squeeze(-1)
        prediction = temporal + history_residual

        if self.reversible_norm_enabled:
            affine_weight = self.affine_weight[ids].unsqueeze(-1)
            affine_bias = self.affine_bias[ids].unsqueeze(-1)
            prediction = (prediction - affine_bias) / (
                affine_weight + self.config.norm_eps
            )
            prediction = prediction * batch_std + batch_mean

        if return_intermediates:
            return NativeTiDEOutput(
                prediction=prediction,
                encoding=encoding,
                decoder=decoder_out,
                future_features=future_features,
                history_residual=history_residual,
            )
        return prediction

    def forward_reference(
        self,
        past_data,
        future_features,
        series_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call the native model with the upstream tensor layout.

        ``past_data`` is ``(past_target, past_numeric, past_categorical)``.
        Numeric covariates use the upstream ``[F, T]`` layout and categorical
        covariates use ``[n_cat, T]``.  ``future_features`` is
        ``(future_numeric, future_categorical)``.
        """
        past_target, past_numeric, past_categorical = past_data
        future_numeric, future_categorical = future_features

        def numeric_time_major(value):
            if value is None:
                return None
            if value.dim() == 2:
                return value.transpose(0, 1)
            if value.dim() == 3:
                return value.transpose(1, 2)
            raise ValueError("numeric covariates must be [F, T] or [B, F, T]")

        def categorical_time_major(value, time_steps):
            if value is None:
                return None
            if value.dim() == 2:
                return [value[index] for index in range(value.shape[0])]
            if value.dim() == 3:
                return [
                    value[:, index, :] for index in range(value.shape[1])
                ]
            raise ValueError(
                "categorical covariates must be [n_cat, T] or [B, n_cat, T]"
            )

        return self.forward(
            past_target,
            numeric_time_major(past_numeric),
            numeric_time_major(future_numeric),
            series_ids=series_ids,
            past_categorical=categorical_time_major(past_categorical, self.L),
            future_categorical=categorical_time_major(
                future_categorical, self.H
            ),
        )
