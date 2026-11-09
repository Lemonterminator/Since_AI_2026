"""Task objectives in normalized spread units. No parameter updates."""
import torch
from torch.nn import functional as F

from .contracts import Forecast, ForecastBatch, Targets

AUX_COMPONENT_CLAMP = 6.0  # normalized units; keeps a single spike from dominating
# Auxiliary labels in reserve order: mfrr_up, mfrr_down, afrr_up, afrr_down.
AUX_LABELS = ("up_spread", "down_spread", "afrr_up_spread", "afrr_down_spread")


def _weighted(values, mask, weight=None):
    """Masked mean that also honours an optional per-sample weight ``[B]``."""
    weights = mask.to(values.dtype).expand_as(values)
    if weight is not None:
        weight = weight.to(values.dtype)
        while weight.dim() < values.dim():
            weight = weight.unsqueeze(-1)
        weights = weights * weight
    return (values * weights).sum() / weights.sum().clamp_min(1)


def _aux_rule_loss(targets, aux_components, sample_weight):
    """Conditional component L1 for a two- or four-component auxiliary head.

    Even components are up-regulation magnitudes and odd components are
    down-regulation magnitudes.  Each term is supervised only on quarters with
    the matching direction, a valid main label, and (when available) a valid
    reserve label for that component.  An all-empty mask contributes zero.
    """
    components = tuple(aux_components)
    if len(components) not in (2, 4):
        raise ValueError("Auxiliary components must contain 2 or 4 tensors")
    labels = [getattr(targets, name) for name in AUX_LABELS[:len(components)]]
    if any(label is None for label in labels):
        return None
    total = None
    for index, (component, label) in enumerate(zip(components, labels)):
        mask = targets.direction_mask & targets.price_mask
        if targets.reserve_mask is not None:
            mask = mask & targets.reserve_mask[..., index]
        side = 0 if index % 2 == 0 else 1
        term = _weighted((component - label.clamp_max(AUX_COMPONENT_CLAMP)).abs(),
                         mask & (targets.direction == side), sample_weight)
        total = term if total is None else total + term
    return total


def task_loss(forecast: Forecast, targets: Targets, batch: ForecastBatch,
              quantile_weight=1.0, direction_weight=1.0, spike_weight=1.0,
              sample_weight=None, aux_components=None, aux_weight=0.15):
    levels = forecast.quantiles.new_tensor([.1, .5, .9])
    error = targets.spread[..., None] - forecast.quantiles
    quantile = _weighted(torch.maximum(levels * error, (levels - 1) * error),
                         targets.price_mask[..., None], sample_weight)
    direction_target = F.one_hot(targets.direction.clamp(0, 2), 3)
    direction = _weighted((forecast.direction - direction_target).square().sum(-1),
                          targets.direction_mask, sample_weight)
    spike = _weighted((forecast.spikes[..., 1:] - targets.spikes.to(forecast.spikes.dtype)).square(),
                      targets.price_mask[..., None], sample_weight)
    losses = {"total": quantile_weight * quantile + direction_weight * direction + spike_weight * spike,
              "quantile": quantile, "direction": direction, "spike": spike}
    if aux_components is not None:
        aux_rule = _aux_rule_loss(targets, aux_components, sample_weight)
        if aux_rule is not None:
            losses["aux_rule"] = aux_rule
            losses["total"] = losses["total"] + aux_weight * aux_rule
    return losses
