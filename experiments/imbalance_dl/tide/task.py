"""Task bridge from ``ForecastBatch`` to the shared ``Forecast`` contract."""

from dataclasses import replace

import torch
from torch import nn

from experiments.imbalance_dl.shared.contracts import Forecast, ForecastBatch, Targets
from experiments.imbalance_dl.shared.heads import TaskHeads

from .config import TiDEConfig
from .context import DailyContext
from .market import MarketRuleTaskHeads
from ..shared.losses import task_loss
from .native import MLPResidual, NativeTiDE, NativeTiDEOutput


def _expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=like.device, dtype=like.dtype)
    while mask.dim() < like.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(like)


def _sequence_tensor(
    value: torch.Tensor,
    batch_size: int,
    time_steps: int,
    name: str,
) -> torch.Tensor:
    if value.dim() == 3:
        if value.shape[-1] != 1:
            raise ValueError(f"{name} must have one channel")
        value = value[..., 0]
    if value.dim() != 2:
        raise ValueError(f"{name} must be [B, T] or [B, T, 1]")
    if value.shape != (batch_size, time_steps):
        raise ValueError(f"{name} has the wrong shape")
    return value


def _channel_tensor(
    value: torch.Tensor,
    batch_size: int,
    time_steps: int,
    channels: int,
    name: str,
) -> torch.Tensor:
    if channels == 0:
        return value.new_zeros(batch_size, time_steps, 0)
    if value.dim() == 2:
        value = value.unsqueeze(-1)
    if value.dim() != 3:
        raise ValueError(f"{name} must be [B, T, C]")
    if value.shape[:2] != (batch_size, time_steps):
        raise ValueError(f"{name} has the wrong time or batch dimension")
    if value.shape[-1] == 1 and channels != 1:
        value = value.expand(batch_size, time_steps, channels)
    if value.shape[-1] != channels:
        raise ValueError(f"{name} has {value.shape[-1]} channels; expected {channels}")
    return value


def _normalized_age(age: torch.Tensor) -> torch.Tensor:
    """Compress unbounded hours-to-publication before the per-step projection."""
    age = torch.nan_to_num(age, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.log1p(age.clamp_min(0.0))


class TiDEForecaster(nn.Module):
    """TiDE backbone with the shared quantile, direction, and spike heads.

    The native point head is replaced by the task heads.  The native linear
    history skip enters only the median quantile path.  The final temporal
    decoder input and future projected covariates form a per-horizon task
    representation.  History-only observations, target presence, and ages enter
    the native per-step feature projection as additional channels.
    """

    def __init__(self, config: TiDEConfig):
        super().__init__()
        if config.reversible_norm:
            raise ValueError(
                "TiDEForecaster expects normalized spread targets; use "
                "NativeTiDE for the reversible-normalization reference path"
            )
        self.config = config
        known_channels = 3 * config.known_channels
        history_channels = (
            known_channels + 3 * config.observed_channels + 2
        )
        native_config = (replace(config, history=config.short_history, short_history=0)
                         if config.short_history else config)
        self.native = NativeTiDE(
            native_config,
            feature_channels=known_channels,
            history_feature_channels=history_channels,
        )
        self.horizon_projection = MLPResidual(
            config.decoder_output_dim + config.temporal_width,
            config.final_decoder_hidden, config.width,
            layer_norm=config.layer_norm, dropout_rate=config.dropout,
            layer_norm_eps=config.layer_norm_eps, pre_norm=config.pre_norm,
        )
        # The task temporal decoder replaces this native scalar point head.
        self.native.final_decoder.requires_grad_(False)
        self.heads = (MarketRuleTaskHeads(config.width, config.learn_volume_weights)
                      if config.use_volume_rule_head else TaskHeads(
                          config.width, use_market_rule=config.use_market_rule_head,
                          direct_rule_gaps=config.direct_rule_gaps))
        if config.rule_gate_bias is not None:
            nn.init.zeros_(self.heads.rule_gate.weight)
            nn.init.constant_(self.heads.rule_gate.bias, config.rule_gate_bias)
        self.daily_context = (DailyContext(config.history, history_channels, config.width,
                                           config.dropout, config.context_patch_size)
                              if config.short_history else None)
        self.asof_norm = None
        self.asof_projection = None
        if config.use_asof_token:
            state_width = 4 * (1 + config.observed_channels)
            self.asof_norm = nn.LayerNorm(state_width, eps=config.layer_norm_eps)
            self.asof_projection = nn.Linear(state_width, config.width)
            nn.init.zeros_(self.asof_projection.weight)
            nn.init.zeros_(self.asof_projection.bias)

    def _target_history(self, batch: ForecastBatch, batch_size: int, device) -> torch.Tensor:
        history = batch.target_history.to(device=device)
        mask = _expand_mask(batch.target_mask, history)
        history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
        return (history * mask)[..., 0]

    def _known_features(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        age: torch.Tensor,
        batch_size: int,
        time_steps: int,
        name: str,
    ) -> torch.Tensor:
        channels = self.config.known_channels
        values = _channel_tensor(
            values.to(device=values.device), batch_size, time_steps, channels, name
        )
        mask = _channel_tensor(
            mask.to(device=values.device), batch_size, time_steps, channels, f"{name}_mask"
        )
        age = _channel_tensor(
            age.to(device=values.device), batch_size, time_steps, channels, f"{name}_age"
        )
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mask = mask.to(values.dtype)
        age = _normalized_age(age.to(values.dtype))
        return torch.cat((values * mask, mask, age), dim=-1)

    def _history_features(
        self, batch: ForecastBatch, batch_size: int, device
    ) -> torch.Tensor:
        observed = batch.observed_history.to(device=device)
        mask = _expand_mask(batch.observed_mask, observed)
        age = _channel_tensor(
            batch.observed_age.to(device=device),
            batch_size,
            self.config.history,
            self.config.observed_channels,
            "observed_age",
        ).to(observed.dtype)
        observed = torch.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
        observed_features = torch.cat(
            (observed * mask, mask, _normalized_age(age)), dim=-1
        )

        target_mask = _sequence_tensor(
            batch.target_mask.to(device=device),
            batch_size,
            self.config.history,
            "target_mask",
        ).unsqueeze(-1)
        target_age = _sequence_tensor(
            batch.target_age.to(device=device),
            batch_size,
            self.config.history,
            "target_age",
        ).unsqueeze(-1)
        return torch.cat(
            (
                observed_features,
                target_mask.to(observed.dtype),
                _normalized_age(target_age),
            ),
            dim=-1,
        )

    @staticmethod
    def _state_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-channel ``[latest, mean of 8, change, present]`` from ``[B, C, A]``."""
        window = values.shape[-1]
        position = torch.arange(window, device=values.device).expand_as(mask)
        valid = torch.where(mask, position, torch.full_like(position, -1))
        latest = valid.max(-1).values
        present = latest >= 0
        latest_value = values.gather(-1, latest.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        latest_value = torch.where(present, latest_value, torch.zeros_like(latest_value))
        previous_position = torch.where(position < latest.unsqueeze(-1), valid,
                                        torch.full_like(valid, -1)).max(-1).values
        previous = values.gather(-1, previous_position.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        change = torch.where(previous_position >= 0, latest_value - previous,
                             torch.zeros_like(latest_value))
        tail = min(8, window)
        recent_mask = mask[..., window - tail:]
        mean = (values[..., window - tail:] * recent_mask).sum(-1) / recent_mask.sum(-1).clamp_min(1)
        return torch.cat((latest_value, mean, change, present.to(values.dtype)), dim=-1)

    def _asof_encoding(self, batch: ForecastBatch, batch_size: int, device) -> torch.Tensor:
        """Publication-aligned state across target and observed history."""
        steps = min(int(self.config.asof_state_channels), self.config.history)

        def prepared(values, mask, channels, name):
            values = _channel_tensor(torch.nan_to_num(values.to(device=device), nan=0.0,
                                                      posinf=0.0, neginf=0.0),
                                     batch_size, self.config.history, channels, name)
            expanded = _expand_mask(mask.to(device=device), values).bool()
            # Slice the time axis; the trailing axis holds channels.
            return (values[:, -steps:].transpose(1, 2),
                    expanded[:, -steps:].transpose(1, 2))

        target_values, target_mask = prepared(batch.target_history, batch.target_mask, 1,
                                             "target_history")
        observed_values, observed_mask = prepared(batch.observed_history, batch.observed_mask,
                                                  self.config.observed_channels,
                                                  "observed_history")
        values = torch.cat((target_values, observed_values), dim=1)
        mask = torch.cat((target_mask, observed_mask), dim=1)
        token = self._state_token(values, mask)
        return self.asof_projection(self.asof_norm(token))

    def forward(self, batch: ForecastBatch, return_aux: bool = False):
        sample = batch.target_history
        batch_size = sample.shape[0]
        device = sample.device
        target = self._target_history(batch, batch_size, device)
        known_history = self._known_features(
            batch.known_history,
            batch.known_history_mask,
            batch.known_history_age,
            batch_size,
            self.config.history,
            "known_history",
        )
        known_future = self._known_features(
            batch.known_future,
            batch.known_future_mask,
            batch.known_future_age,
            batch_size,
            self.config.horizon,
            "known_future",
        )
        history_features = torch.cat(
            (
                known_history,
                self._history_features(batch, batch_size, device),
            ),
            dim=-1,
        ).to(sample.dtype)

        encoding_extra = (
            self._asof_encoding(batch, batch_size, device) if self.asof_projection else None
        )
        if self.daily_context is not None:
            context = self.daily_context(target, history_features)
            encoding_extra = context if encoding_extra is None else context + encoding_extra
            target = target[:, -self.config.short_history:]
            history_features = history_features[:, -self.config.short_history:]
        native_output: NativeTiDEOutput = self.native(
            target,
            history_features,
            known_future,
            series_ids=batch.series_ids,
            encoding_extra=encoding_extra,
            return_intermediates=True,
        )
        horizon_features = torch.cat(
            (native_output.decoder, native_output.future_features), dim=1
        ).transpose(1, 2)
        hidden = self.horizon_projection(horizon_features)
        kwargs = ({"spread_offset": batch.spread_center / batch.spread_scale}
                  if self.config.use_volume_rule_head or self.config.center_rule_anchor else {})
        return self.heads(hidden, median_residual=native_output.history_residual,
                          return_aux=return_aux, **kwargs)

    def objective(
        self,
        batch: ForecastBatch,
        targets: Targets,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the shared task objective dictionary."""
        if self.config.use_market_rule_head:
            forecast, aux = self(batch, return_aux=True)
        else:
            forecast, aux = self(batch), None
        if aux is not None:
            if not self.config.supervise_afrr:
                aux = aux[:2]
            if self.config.rule_aux_weight == 0.0:
                aux = None
        return task_loss(forecast, targets, batch, sample_weight=sample_weight,
                         aux_components=aux, aux_weight=self.config.rule_aux_weight)
