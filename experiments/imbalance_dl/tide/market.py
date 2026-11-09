"""A differentiable reserve-price anchor for the learned quantile head."""

import torch
from torch import nn
from torch.nn import functional as F

from ..shared.contracts import Forecast
from ..shared.heads import TaskHeads


class MarketRuleTaskHeads(TaskHeads):
    """Predict four reserve spreads and positive relative activation volumes.

    The direction-weighted price is an anchor, not a mixture-distribution
    median. Pinball loss trains a gated direct median and ordered gaps around
    it. Future realized reserve prices and volumes are never forward inputs.
    """

    def __init__(self, width, learn_volume_weights=True):
        super().__init__(width)
        self.reserve = nn.Linear(width, 4)
        self.activation = nn.Linear(width, 4) if learn_volume_weights else None
        self.rule_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.rule_gate.weight)
        nn.init.constant_(self.rule_gate.bias, -2.0)

    @staticmethod
    def reserve_anchor(components, volumes):
        """Order: mFRR up/down, aFRR up/down; magnitudes share spread units."""
        mf_up, mf_down, af_up, af_down = components.unbind(-1)
        vm_up, vm_down, va_up, va_down = volumes.unbind(-1)
        w_up = vm_up / (vm_up + va_up)
        w_down = vm_down / (vm_down + va_down)
        return (w_up * mf_up + (1.0 - w_up) * af_up,
                w_down * mf_down + (1.0 - w_down) * af_down)

    def forward(self, hidden, median_residual=None, return_aux=False, spread_offset=None):
        direct = super().forward(hidden, median_residual)
        components = F.softplus(self.reserve(hidden))
        # Positive denominators preserve the convex combination at zero activation.
        volumes = (F.softplus(self.activation(hidden)) + 1e-4
                   if self.activation is not None else torch.ones_like(components))
        up, down = self.reserve_anchor(components, volumes)
        anchor = direct.direction[..., 0] * up - direct.direction[..., 1] * down
        if spread_offset is not None:
            anchor = anchor - spread_offset
        if median_residual is not None:
            anchor = anchor + median_residual
        gate = torch.sigmoid(self.rule_gate(hidden)).squeeze(-1)
        median = gate * anchor + (1.0 - gate) * direct.quantiles[..., 1]
        lower = direct.quantiles[..., 1] - direct.quantiles[..., 0]
        upper = direct.quantiles[..., 2] - direct.quantiles[..., 1]
        forecast = Forecast(torch.stack((median - lower, median, median + upper), -1),
                            direct.direction, direct.spikes)
        aux = tuple(components.unbind(-1))
        self._last_aux = aux
        return (forecast, aux) if return_aux else forecast
