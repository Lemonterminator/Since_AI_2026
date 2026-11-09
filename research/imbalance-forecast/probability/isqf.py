"""Incremental quantile functions with linear splines and exponential tails.

This is an independently written, experimental implementation of the constructions
in Park et al. (2022), sections 3--4. GPD tails and analytical CRPS are not
implemented. The quadrature score below is an approximation, not a CRPS bound.
No author-code parity or empirical performance has been established.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SplineQuantileDistribution:
    """A continuous distribution per batch element.

    ``levels`` and ``values`` have shape ``batch_shape + (n_knots,)`` and are
    strictly increasing. Tail scales have shape ``batch_shape``. Query methods
    accept a scalar, a batch-shaped tensor, or a global one-dimensional grid;
    the latter returns ``batch_shape + (grid_size,)``. A trailing query dimension
    may also be supplied explicitly as ``batch_shape + (grid_size,)``.
    """

    levels: Tensor
    values: Tensor
    left_scale: Tensor
    right_scale: Tensor

    def _query(self, value: Tensor | float) -> tuple[Tensor, bool]:
        x = torch.as_tensor(value, device=self.values.device, dtype=self.values.dtype)
        batch = self.values.shape[:-1]
        if x.ndim == 0:
            return x.expand(batch).unsqueeze(-1), True
        if tuple(x.shape) == tuple(batch):
            return x.unsqueeze(-1), True
        if x.ndim == 1:
            return x.reshape((1,) * len(batch) + (-1,)).expand(*batch, -1), False
        if x.ndim == len(batch) + 1 and tuple(x.shape[:-1]) == tuple(batch):
            return x, False
        raise ValueError("query must be scalar, batch-shaped, a 1-D grid, or batch-shaped with a query axis")

    def quantile(self, alpha: Tensor | float) -> Tensor:
        a, squeeze = self._query(alpha)
        if bool(((a <= 0) | (a >= 1) | ~torch.isfinite(a)).any()):
            raise ValueError("quantile levels must be finite and strictly between 0 and 1")
        lv, qv = self.levels.unsqueeze(-2), self.values.unsqueeze(-2)
        # Sum of clipped positive increments is continuous at every knot.
        frac = ((a.unsqueeze(-1) - lv[..., :-1]) / (lv[..., 1:] - lv[..., :-1])).clamp(0, 1)
        middle = qv[..., 0] + (frac * (qv[..., 1:] - qv[..., :-1])).sum(-1)
        left = qv[..., 0] + self.left_scale.unsqueeze(-1) * torch.log(a / lv[..., 0])
        right = qv[..., -1] - self.right_scale.unsqueeze(-1) * torch.log((1 - a) / (1 - lv[..., -1]))
        out = torch.where(a < lv[..., 0], left, torch.where(a > lv[..., -1], right, middle))
        return out.squeeze(-1) if squeeze else out

    def cdf(self, value: Tensor | float) -> Tensor:
        x, squeeze = self._query(value)
        lv, qv = self.levels.unsqueeze(-2), self.values.unsqueeze(-2)
        frac = ((x.unsqueeze(-1) - qv[..., :-1]) / (qv[..., 1:] - qv[..., :-1])).clamp(0, 1)
        middle = lv[..., 0] + (frac * (lv[..., 1:] - lv[..., :-1])).sum(-1)
        # Clamp the unused tail branch too, so torch.where cannot create inf gradients.
        left = lv[..., 0] * torch.exp(((x - qv[..., 0]) / self.left_scale.unsqueeze(-1)).clamp(max=0))
        right = 1 - (1 - lv[..., -1]) * torch.exp((-(x - qv[..., -1]) / self.right_scale.unsqueeze(-1)).clamp(max=0))
        out = torch.where(x < qv[..., 0], left, torch.where(x > qv[..., -1], right, middle))
        return out.squeeze(-1) if squeeze else out

    def spike_probabilities(self, low: Tensor | float, high: Tensor | float) -> tuple[Tensor, Tensor]:
        """Return ``P(Y < low), P(Y > high)`` in the distribution's units.

        This implementation has continuous, strictly increasing quantiles and
        therefore no point masses; strict and non-strict CDF boundaries coincide.
        Callers using scaled spread must scale the +/-50 EUR thresholds first.
        """
        return self.cdf(low), 1 - self.cdf(high)

    def quadrature_crps(self, target: Tensor, points: int = 199) -> Tensor:
        """Midpoint-rule approximation to 2*integral(pinball), per batch item."""
        if points < 1:
            raise ValueError("points must be positive")
        if tuple(target.shape) != tuple(self.values.shape[:-1]):
            raise ValueError("target must have the distribution batch shape")
        tau = (torch.arange(points, device=self.values.device, dtype=self.values.dtype) + 0.5) / points
        d = target.unsqueeze(-1) - self.quantile(tau)
        return 2 * torch.maximum(tau * d, (tau - 1) * d).mean(dim=-1)


class IncrementalSplineQuantileHead(nn.Module):
    """Learn ordered anchors and, optionally, internal spline knots and tail scales.

    ``mode='iqf'`` uses straight interpolation and exponential tail scales
    determined by the outer two anchor values. ``mode='isqf'`` learns internal
    knot locations/heights and two positive exponential scales. ``segments`` is
    the number of linear pieces per anchor interval in ISQF mode.
    """

    def __init__(self, in_dim: int, anchor_levels: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                 mode: str = "isqf", segments: int = 4, min_increment: float = 1e-4):
        super().__init__()
        if mode not in ("iqf", "isqf"):
            raise ValueError("mode must be 'iqf' or 'isqf'; GPD tails are not implemented")
        if len(anchor_levels) < 2 or any(not 0 < x < 1 for x in anchor_levels) or any(a >= b for a, b in zip(anchor_levels, anchor_levels[1:])):
            raise ValueError("anchor levels must be strictly increasing inside (0,1)")
        if segments < 1 or min_increment <= 0:
            raise ValueError("segments and min_increment must be positive")
        self.mode, self.segments, self.min_increment = mode, (1 if mode == "iqf" else segments), min_increment
        self.register_buffer("anchor_levels", torch.tensor(anchor_levels))
        self.anchor_proj = nn.Linear(in_dim, len(anchor_levels))
        if mode == "isqf":
            size = (len(anchor_levels) - 1) * self.segments
            self.level_proj = nn.Linear(in_dim, size)
            self.height_proj = nn.Linear(in_dim, size)
            self.tail_proj = nn.Linear(in_dim, 2)

    def forward(self, h: Tensor) -> SplineQuantileDistribution:
        raw = self.anchor_proj(h)
        anchors = torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(F.softplus(raw[..., 1:]) + self.min_increment, -1)], -1)
        a = self.anchor_levels.to(dtype=h.dtype)
        k, s = a.numel(), self.segments
        shape = h.shape[:-1] + (k - 1, s)
        if self.mode == "iqf":
            level_weights = height_weights = h.new_ones(shape)
            left_scale = (anchors[..., 1] - anchors[..., 0]) / torch.log(a[1] / a[0])
            right_scale = (anchors[..., -1] - anchors[..., -2]) / torch.log((1 - a[-2]) / (1 - a[-1]))
        else:
            # Positive normalized widths/heights ensure each interval ends at its anchor.
            level_weights = F.softmax(self.level_proj(h).reshape(shape), dim=-1).clamp_min(self.min_increment)
            height_weights = F.softmax(self.height_proj(h).reshape(shape), dim=-1).clamp_min(self.min_increment)
            level_weights = level_weights / level_weights.sum(-1, keepdim=True)
            height_weights = height_weights / height_weights.sum(-1, keepdim=True)
            tails = F.softplus(self.tail_proj(h)) + self.min_increment
            left_scale, right_scale = tails.unbind(-1)
        level_cum = torch.cumsum(level_weights, -1)
        height_cum = torch.cumsum(height_weights, -1)
        internal_levels = a[:-1, None] + (a[1:] - a[:-1])[:, None] * level_cum
        internal_values = anchors[..., :-1, None] + (anchors[..., 1:] - anchors[..., :-1])[..., None] * height_cum
        first_level = a[0].expand(h.shape[:-1] + (1,))
        levels = torch.cat([first_level, internal_levels.flatten(-2)], -1)
        values = torch.cat([anchors[..., :1], internal_values.flatten(-2)], -1)
        return SplineQuantileDistribution(levels, values, left_scale, right_scale)
