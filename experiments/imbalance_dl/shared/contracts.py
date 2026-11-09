"""Tensor contracts. Prices are normalized spreads until export."""
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass
class ForecastBatch:
    target_history: Tensor  # [B,L,1]
    observed_history: Tensor  # [B,L,C_obs]
    known_history: Tensor  # [B,L,C_known]
    known_future: Tensor  # [B,H,C_known]
    target_mask: Tensor
    observed_mask: Tensor
    known_history_mask: Tensor
    known_future_mask: Tensor
    target_age: Tensor
    observed_age: Tensor
    known_history_age: Tensor
    known_future_age: Tensor
    spot: Tensor  # Raw EUR/MWh [B,H]. Never edit this anchor.
    spread_center: Tensor  # [B,1]
    spread_scale: Tensor  # Positive [B,1]
    origins: tuple[str, ...] = ()
    data_cutoffs: tuple[str, ...] = ()
    observed_editable: Tensor | None = None  # [C_obs], continuous fields only.
    known_editable: Tensor | None = None  # [C_known], exclude calendars.
    series_ids: Tensor | None = None  # Long [B] static series identifiers.

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                                for f in fields(self) if isinstance(getattr(self, f.name), Tensor)})


@dataclass
class Targets:
    spread: Tensor  # Normalized [B,H]. Unavailable labels have a false mask.
    direction: Tensor  # Long [B,H], up=0, down=1, none=2.
    price_mask: Tensor
    direction_mask: Tensor
    spikes: Tensor  # Bool [B,H,2], high/low from original prices before scaling.
    up_spread: Tensor | None = None  # Non-negative mFRR up-regulation spread, sealed by scale.
    down_spread: Tensor | None = None  # Non-negative down-regulation spread magnitude.
    afrr_up_spread: Tensor | None = None  # Non-negative aFRR up-regulation spread, sealed by scale.
    afrr_down_spread: Tensor | None = None  # Non-negative aFRR down-regulation spread magnitude.
    # Bool [B,H,4] validity in reserve order: mfrr_up, mfrr_down, afrr_up, afrr_down.
    reserve_mask: Tensor | None = None

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                                for f in fields(self) if isinstance(getattr(self, f.name), Tensor)})


@dataclass
class Forecast:
    quantiles: Tensor  # Normalized spread [B,H,3], levels .1/.5/.9.
    direction: Tensor  # Probabilities [B,H,3], up/down/none.
    spikes: Tensor  # Probabilities [B,H,3], normal/high/low.
    samples: Tensor | None = None  # Normalized spread [B,K,H].


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype).expand_as(values)
    return torch.where(mask.bool(), values, 0).sum() / mask.sum().clamp_min(1)


def observed_context(batch: ForecastBatch) -> Tensor:
    """Flatten history-only values, presence, and age for a task bridge."""
    return torch.cat((batch.observed_history * batch.observed_mask,
                      batch.observed_mask.to(batch.observed_history.dtype),
                      batch.observed_age), dim=-1).flatten(1)
