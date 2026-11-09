"""Ordered spread quantiles and competition probability heads."""
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import Forecast


class TaskHeads(nn.Module):
    """Ordered quantiles and the two three-way competition heads.

    ``use_market_rule=True`` adds the reserve-spread mixture.  The median blends
    a directional expectation ``p_up * up_spread - p_down * down_spread`` with
    the direct median through a learned gate, and each gap is widened by the
    side whose activation can actually reach it.
    """

    def __init__(self, width, use_market_rule=False, direct_rule_gaps=False):
        super().__init__()
        self.quantile = nn.Linear(width, 3)
        self.direction = nn.Linear(width, 3)
        self.spike = nn.Linear(width, 3)
        self.use_market_rule = use_market_rule
        self.direct_rule_gaps = direct_rule_gaps
        self._last_aux = None
        if use_market_rule:
            self.up_comp = nn.Linear(width, 3)
            self.down_comp = nn.Linear(width, 3)
            self.rule_gate = nn.Linear(width, 1)

    @property
    def aux_components(self):
        """``(up_spread, down_spread)`` magnitudes from the latest forward pass."""
        return self._last_aux

    def forward(self, hidden, median_residual=None, return_aux=False, spread_offset=None):
        raw = self.quantile(hidden)
        median = raw[..., 0]
        if median_residual is not None:
            median = median + median_residual
        lo_gap = F.softplus(raw[..., 1])
        hi_gap = F.softplus(raw[..., 2])
        direction = self.direction(hidden).softmax(-1)
        aux = None
        if self.use_market_rule:
            residual = 0.0 if median_residual is None else median_residual
            u_raw = self.up_comp(hidden)
            d_raw = self.down_comp(hidden)
            u50, u10, d50, d10 = (F.softplus(u_raw[..., 0]), F.softplus(u_raw[..., 1]),
                                  F.softplus(d_raw[..., 0]), F.softplus(d_raw[..., 1]))
            u90 = u50 + F.softplus(u_raw[..., 2])
            d90 = d50 + F.softplus(d_raw[..., 2])
            p_up, p_dn = direction[..., 0], direction[..., 1]
            gate = torch.sigmoid(self.rule_gate(hidden)).squeeze(-1)
            rule_median = p_up * u50 - p_dn * d50 + residual
            if spread_offset is not None:
                rule_median = rule_median - spread_offset
            median = gate * rule_median + (1.0 - gate) * median
            if not self.direct_rule_gaps:
                lo_gap = gate * (p_dn * d90 + p_up * u10) + (1.0 - gate) * lo_gap
                hi_gap = gate * (p_up * u90 + p_dn * d10) + (1.0 - gate) * hi_gap
            aux = (u50, d50)
            self._last_aux = aux
        forecast = Forecast(
            torch.stack((median - lo_gap, median, median + hi_gap), dim=-1),
            direction,
            self.spike(hidden).softmax(-1),
        )
        if return_aux:
            return forecast, aux
        return forecast
