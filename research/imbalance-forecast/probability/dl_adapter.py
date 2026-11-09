"""Opt-in IQF/ISQF head matching the existing DL head's output contract.

Nothing in this module changes default checkpoints or the existing trainer.
Use build_spline_architecture explicitly to construct an experimental model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .isqf import IncrementalSplineQuantileHead, SplineQuantileDistribution


class SplineProbHead(nn.Module):
    """Return ``(19 scaled-spread quantiles, direction logits, spike logits)``.

    Constructor arguments mirror ``HierarchicalProbHead``. The quantile prior
    initializes anchor biases via interpolation of its 0.05--0.95 grid; feature
    weights retain normal PyTorch initialization. The independent spike logits
    exist solely for compatibility and need not agree with the distribution.
    For coherent spike probabilities use ``forward_with_distribution`` and
    ``distribution.spike_probabilities`` with thresholds in the same units.
    """

    def __init__(self, in_dim: int, raw_dim: int, q_init: np.ndarray,
                 dir_log_prior: np.ndarray, spike_log_prior: np.ndarray,
                 *, mode: str = "isqf", segments: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.distribution_head = IncrementalSplineQuantileHead(in_dim, mode=mode, segments=segments)
        self.dir_proj, self.dir_highway = nn.Linear(in_dim, 3), nn.Linear(raw_dim, 3, bias=False)
        self.spk_proj, self.spk_highway = nn.Linear(in_dim, 3), nn.Linear(raw_dim, 3, bias=False)
        self.register_buffer("grid", torch.arange(1, 20, dtype=torch.float32) / 20)
        init = np.asarray(q_init, dtype=np.float64)
        if init.shape != (19,) or not np.isfinite(init).all():
            raise ValueError("q_init must contain 19 finite values")
        anchors = np.interp(self.distribution_head.anchor_levels.cpu().numpy(), np.arange(1, 20) / 20, init)
        increments = np.maximum(np.diff(anchors) - self.distribution_head.min_increment, 1e-3)
        # Stable inverse softplus, including large initial spacings.
        inv = increments + np.log(-np.expm1(-increments))
        with torch.no_grad():
            self.distribution_head.anchor_proj.bias.copy_(torch.as_tensor(np.r_[anchors[0], inv], dtype=self.distribution_head.anchor_proj.bias.dtype))
            self.dir_proj.bias.copy_(torch.as_tensor(dir_log_prior, dtype=self.dir_proj.bias.dtype))
            self.spk_proj.bias.copy_(torch.as_tensor(spike_log_prior, dtype=self.spk_proj.bias.dtype))

    def forward_with_distribution(self, h: Tensor, raw_in: Tensor) -> tuple[Tensor, Tensor, Tensor, SplineQuantileDistribution]:
        h = self.norm(h)
        distribution = self.distribution_head(h)
        quantiles = distribution.quantile(self.grid.to(dtype=h.dtype))
        direction = self.dir_proj(h) + self.dir_highway(raw_in)
        spike = self.spk_proj(h) + self.spk_highway(raw_in)
        return quantiles, direction, spike, distribution

    def forward(self, h: Tensor, raw_in: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, direction, spike, _ = self.forward_with_distribution(h, raw_in)
        return q, direction, spike


def build_spline_architecture(arch, ds, dh, q_init, dir_lp, spike_lp,
                              *, mode="isqf", segments=4):
    """Build an existing pure-DL backbone with the opt-in spline head.

    Call before creating an optimizer. This creates a new checkpoint schema;
    existing released checkpoints must still use their original factory. The
    ordinary trainer consumes the 19-level tuple and its existing pinball loss;
    selecting this builder does not silently change the loss to CRPS.
    """
    from dl.models import PURE_DL_ARCHS, build_architecture
    if arch not in PURE_DL_ARCHS:
        raise ValueError("spline heads support the four pure-DL architectures only")
    model = build_architecture(arch, ds, dh, q_init, dir_lp, spike_lp)
    old = model.head
    head = SplineProbHead(old.q_proj.in_features, old.q_highway.in_features,
                          q_init, dir_lp, spike_lp, mode=mode, segments=segments)
    # Keep the independently initialized classification recipe of the backbone.
    head.dir_proj.load_state_dict(old.dir_proj.state_dict())
    head.dir_highway.load_state_dict(old.dir_highway.state_dict())
    head.spk_proj.load_state_dict(old.spk_proj.state_dict())
    head.spk_highway.load_state_dict(old.spk_highway.state_dict())
    model.head = head
    return model
