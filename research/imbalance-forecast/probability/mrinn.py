"""MRINN rule operators and explicitly separate Austrian/Finnish task models.

Original equations: https://arxiv.org/html/2605.09061v1 (III and IV).
Independent implementation, not copied author code or a benchmark reproduction.
Inputs must be publication-masked, training-scaled historical features. No data
loader here may look up future realized components. Constants use supplied
training-only unit-group (median, IQR) statistics, as in IV-F.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import torch
from torch import nn
from torch.nn import functional as F


def latent_max(a, b):
    return b + F.softplus(a - b)


def latent_min(a, b):
    return -latent_max(-a, -b)


def latent_abs(a, eps=1e-7):
    return torch.sqrt(a.square() + eps)


def latent_div(a, b, eps=1e-7):
    """Paper Eq.24; near-negative epsilon denominators remain a native risk."""
    return a / (b + eps)


class ConditionalSelect(nn.Module):
    """Learned soft branch selection, Eq.26–28 (not hard rule enforcement)."""
    def __init__(self, width: int, branches: int, conditions: int):
        super().__init__()
        self.branches = branches
        self.conditions = conditions
        self.gate = nn.Linear(width * conditions, branches)

    def forward(self, candidates, conditions):
        if len(candidates) != self.branches or len(conditions) != self.conditions:
            raise ValueError("branch/condition arity does not match configured gate")
        weights = self.gate(torch.cat(conditions, dim=-1)).softmax(-1)
        return (torch.stack(candidates, -2) * weights.unsqueeze(-1)).sum(-2)


class OrderedHead(nn.Module):
    def __init__(self, width, quantiles):
        super().__init__()
        levels = tuple(float(q) for q in quantiles)
        if not levels or any(not 0 < q < 1 for q in levels) or sorted(set(levels)) != list(levels) or .5 not in levels:
            raise ValueError("quantiles must be unique, increasing and include 0.5")
        self.register_buffer("levels", torch.tensor(levels))
        self.mid = levels.index(.5)
        self.proj = nn.Linear(width, len(levels))

    def forward(self, h):
        raw = self.proj(h)
        med = raw[..., self.mid:self.mid + 1]
        lo = med - F.softplus(raw[..., :self.mid].flip(-1)).cumsum(-1)
        hi = med + F.softplus(raw[..., self.mid + 1:]).cumsum(-1)
        return torch.cat((lo.flip(-1), med, hi), -1)


class RuleEncoder(nn.Module):
    """Project a fixed historical window independently per physical feature."""
    features: tuple[str, ...] = ()

    def __init__(self, history_length, width, unit_stats: Mapping[str, tuple[float, float]]):
        super().__init__()
        if history_length < 1 or width < 1:
            raise ValueError("positive history_length and width required")
        self.history_length, self.width = history_length, width
        self.project = nn.ModuleDict({k: nn.Linear(history_length, width) for k in self.features})
        for unit in ("price", "volume"):
            center, scale = unit_stats[unit]
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("unit statistics must be finite with positive IQR")
            self.register_buffer(unit + "_stats", torch.tensor([center, scale], dtype=torch.float32))

    def constant(self, value, unit="dimensionless"):
        reference = self.price_stats
        if unit == "dimensionless":
            return reference.new_tensor(value).expand(self.width)
        stats = getattr(self, unit + "_stats")
        return ((stats.new_tensor(value) - stats[0]) / stats[1]).expand(self.width)

    def encode(self, history):
        if set(history) != set(self.features):
            raise ValueError(f"expected exactly these scaled history features: {self.features}")
        shapes = {tuple(x.shape) for x in history.values()}
        if len(shapes) != 1 or next(iter(shapes))[-1:] != (self.history_length,):
            raise ValueError("all feature histories must share (..., history_length) shape")
        if any(not torch.isfinite(x).all() for x in history.values()):
            raise ValueError("mask/impute missing observations before projection")
        return {k: self.project[k](history[k]) for k in self.features}

    def activation(self, f, side):
        ea, em = f['afrr_e_' + side], f['mfrr_e_' + side]
        return latent_div(ea * f['afrr_p_' + side] + em * f['mfrr_p_' + side], ea + em)


class AustrianRuleEncoder(RuleEncoder):
    """Native-style Austrian rule graph; unit scaling and temporal projection
    are explicit. Gate details unspecified by the paper are linear softmax here.
    Requires intraday and avoided-activation signals absent from our FI panel.
    """
    features = ("imbalance", "afrr_e_up", "mfrr_e_up", "afrr_p_up", "mfrr_p_up",
                "afrr_e_down", "mfrr_e_down", "afrr_p_down", "mfrr_p_down",
                "voaa_up", "voaa_down", "id15", "id60", "spot", "liquidity15", "liquidity60")

    def __init__(self, history_length, width, unit_stats):
        super().__init__(history_length, width, unit_stats)
        self.balance_gate = ConditionalSelect(width, 6, 3)
        self.ramp_gate = ConditionalSelect(width, 3, 3)
        self.scarcity_gate = ConditionalSelect(width, 3, 3)
        self.final_gate = ConditionalSelect(width, 2, 1)

    def forward(self, history):
        f = self.encode(history)
        v = f['imbalance']
        up, down = self.activation(f, 'up'), self.activation(f, 'down')
        ep = f['afrr_e_up'] + f['mfrr_e_up']
        en = f['afrr_e_down'] + f['mfrr_e_down']
        bal = self.balance_gate([up, down, up, down, f['voaa_up'], f['voaa_down']], [ep, en, v])
        one = self.constant(1)
        c4 = self.constant(50, 'volume')
        minus_c4 = self.constant(-50, 'volume')
        ramp = self.ramp_gate([-one, latent_div(v, c4), one], [v, minus_c4, c4])
        w15 = latent_min(one, latent_div(f['liquidity15'], self.constant(200, 'volume')))
        w60 = latent_min(one - w15, latent_div(f['liquidity60'], self.constant(200, 'volume')))
        weights = (w15, w60, one - w15 - w60)
        prices = (f['id15'], f['id60'], f['spot'])
        adjusted = [p + ramp * latent_max(self.constant(c, 'price'), .1 * latent_abs(p))
                    for p, c in zip(prices, (5, 10, 15))]
        market = sum(w * p for w, p in zip(weights, adjusted))
        base = sum(w * p for w, p in zip(weights, prices))
        c7, c8, c9 = [self.constant(c, 'volume') for c in (200, 800, 1000)]
        magnitude = latent_abs(v)
        term = self.constant(1000, 'price') * torch.tanh(v)
        middle = base + term * latent_div(magnitude - c7, c9 - c7).pow(3)
        capped = base + term * latent_div(c8 - c7, c9 - c7).pow(3)
        scarcity = self.scarcity_gate([base, middle, capped], [magnitude, c7, c8])
        low = latent_min(latent_min(bal, market), scarcity)
        high = latent_max(latent_max(bal, market), scarcity)
        return self.final_gate([low, high], [v])


class FinnishRuleEncoder(RuleEncoder):
    """FI June-rule inspired latent adaptation, not an exact settlement map.
    Does not infer direction from price signs or force spot for argmax none.
    """
    features = ("direction", "afrr_e_up", "mfrr_e_up", "afrr_p_up", "mfrr_p_up",
                "afrr_e_down", "mfrr_e_down", "afrr_p_down", "mfrr_p_down", "spot")

    def __init__(self, history_length, width, unit_stats):
        super().__init__(history_length, width, unit_stats)
        self.up_gate = ConditionalSelect(width, 3, 2)
        self.down_gate = ConditionalSelect(width, 3, 2)
        self.direction_gate = ConditionalSelect(width, 3, 1)

    def forward(self, history):
        f = self.encode(history)
        candidates = []
        for side, gate in [('up', self.up_gate), ('down', self.down_gate)]:
            candidates.append(gate([self.activation(f, side), f['mfrr_p_' + side], f['afrr_p_' + side]],
                                   [f['mfrr_e_' + side], f['afrr_e_' + side]]))
        return self.direction_gate([*candidates, f['spot']], [f['direction']])


class MRINNForecaster(nn.Module):
    """Multi-horizon task adapter: history -> rule latent -> known-future decoder.
    Output is (spread quantiles, direction logits, spike logits), matching DL's
    tuple convention. Default 19 levels; paper's 7-level head is configurable.
    This decoder is an adaptation, not a claim of exact author architecture parity.
    Raw-unit reconstruction requires adding the known delivery spot externally.
    """
    def __init__(self, encoder: RuleEncoder, future_dim: int, horizon=40,
                 quantiles=tuple(i / 20 for i in range(1, 20))):
        super().__init__()
        if future_dim < 1 or horizon < 1:
            raise ValueError("positive future_dim and horizon required")
        self.encoder, self.horizon, self.future_dim = encoder, horizon, future_dim
        width = encoder.width
        self.future = nn.Linear(future_dim, width)
        self.step = nn.Embedding(horizon, width)
        self.decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU())
        self.head = OrderedHead(width, quantiles)
        self.direction, self.spike = nn.Linear(width, 3), nn.Linear(width, 3)

    def forward(self, history, known_future):
        if known_future.ndim != 3 or known_future.shape[1:] != (self.horizon, self.future_dim):
            raise ValueError("known_future must be (batch, horizon, future_dim)")
        if not torch.isfinite(known_future).all():
            raise ValueError("known_future must be finite and publication-masked")
        state = self.encoder(history)
        if state.ndim != 2 or state.shape[0] != known_future.shape[0]:
            raise ValueError("history must produce one state per batch item")
        steps = torch.arange(self.horizon, device=known_future.device)
        h = self.decoder(state[:, None] + self.future(known_future) + self.step(steps))
        return self.head(h), self.direction(h), self.spike(h)
