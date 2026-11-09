"""Task-2 Frontier Deep Learning Architectures (Pure-GPU, Zero CPU Contention).

Addresses multi-task tail-gradient interference observed in Task-1 by decoupling
the L1 Median (P50) representation from the Non-Crossing Quantile-Spacing &
Classification representations (mirroring LightGBM's separate P50 vs Prob models),
and introduces four frontier architectures from the EPF literature:

6. ``kan_spline_net``       : Kolmogorov-Arnold / Learnable RBF-Spline Network (DL_for_EPF_review)
7. ``patch_tst_horizon``    : PatchTST Multi-Scale Horizon Transformer (Nie et al., 2023)
8. ``itransformer_variate`` : iTransformer Inverted Variate-Attention (Liu et al., 2024; PriceFM)
9. ``decoupled_tide_l1``    : Task-Decoupled Dual-Tower TiDE (pure L1 P50 tower + Distributional tower)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GatedResBlock, MEDIAN_IDX, N_QUANTILES, N_STEPS, _inv_softplus


class DecoupledDualHead(nn.Module):
    """Decouples P50 (pure L1 MAE objective) from Non-Crossing Spacings & Regime Probabilities.

    Eliminates gradient interference from extreme 0.05/0.95 tail quantiles onto P50!
    """

    def __init__(
        self,
        med_dim: int,
        dist_dim: int,
        raw_dim: int,
        q_init: np.ndarray,
        dir_lp: np.ndarray,
        spk_lp: np.ndarray,
    ):
        super().__init__()
        # Dedicated P50 head + highway (receives ONLY |y - p50| L1 gradients!)
        self.med_norm = nn.LayerNorm(med_dim)
        self.med_proj = nn.Linear(med_dim, 1)
        self.med_highway = nn.Linear(raw_dim, 1, bias=False)

        # Dedicated Distributional Spacing + Classification head (detached from P50 gradient!)
        self.dist_norm = nn.LayerNorm(dist_dim)
        self.up_proj = nn.Linear(dist_dim, N_QUANTILES - 1 - MEDIAN_IDX)  # 9 upper spacings
        self.lo_proj = nn.Linear(dist_dim, MEDIAN_IDX)                    # 9 lower spacings
        self.up_highway = nn.Linear(raw_dim, N_QUANTILES - 1 - MEDIAN_IDX, bias=False)
        self.lo_highway = nn.Linear(raw_dim, MEDIAN_IDX, bias=False)

        self.dir_proj = nn.Linear(dist_dim, 3)
        self.dir_highway = nn.Linear(raw_dim, 3, bias=False)
        self.spk_proj = nn.Linear(dist_dim, 3)
        self.spk_highway = nn.Linear(raw_dim, 3, bias=False)

        for layer in (self.med_proj, self.med_highway, self.up_proj, self.lo_proj,
                      self.up_highway, self.lo_highway, self.dir_proj, self.dir_highway,
                      self.spk_proj, self.spk_highway):
            nn.init.normal_(layer.weight, 0.0, 1e-3)

        with torch.no_grad():
            self.med_proj.bias.fill_(float(q_init[MEDIAN_IDX]))
            self.up_proj.bias.copy_(torch.from_numpy(_inv_softplus(np.diff(q_init[MEDIAN_IDX:])).astype(np.float32)))
            # lo_proj[..., k] for k=0..8 ordered outward from median
            lo_diffs_out = np.diff(q_init[: MEDIAN_IDX + 1])[::-1].copy()
            self.lo_proj.bias.copy_(torch.from_numpy(_inv_softplus(lo_diffs_out).astype(np.float32)))
            self.dir_proj.bias.copy_(torch.from_numpy(dir_lp))
            self.spk_proj.bias.copy_(torch.from_numpy(spk_lp))

    def forward(self, h_med: torch.Tensor, h_dist: torch.Tensor, raw_in: torch.Tensor):
        hm = self.med_norm(h_med)
        median = self.med_proj(hm) + self.med_highway(raw_in)  # (B, 40, 1)

        # Detach median anchor when building outer quantiles so tail pinball gradients NEVER pull P50!
        med_anchor = median.detach()
        hd = self.dist_norm(h_dist)
        up_inc = F.softplus(self.up_proj(hd) + self.up_highway(raw_in)) + 1e-4
        lo_inc_out = F.softplus(self.lo_proj(hd) + self.lo_highway(raw_in)) + 1e-4

        up_q = med_anchor + torch.cumsum(up_inc, dim=-1)
        lo_q = torch.flip(med_anchor - torch.cumsum(lo_inc_out, dim=-1), dims=[-1])
        q_scaled = torch.cat([lo_q, median, up_q], dim=-1)

        dir_logits = self.dir_proj(hd) + self.dir_highway(raw_in)
        spk_logits = self.spk_proj(hd) + self.spk_highway(raw_in)
        return q_scaled, dir_logits, spk_logits


class RBFSplineLayer(nn.Module):
    """Fast Kolmogorov-Arnold (KAN) Learnable Gaussian RBF-Spline univariate basis expansion."""

    def __init__(self, in_dim: int, out_dim: int, n_knots: int = 10, grid_range: tuple[float, float] = (-2.5, 2.5)):
        super().__init__()
        knots = torch.linspace(grid_range[0], grid_range[1], n_knots)
        self.register_buffer("knots", knots.view(1, 1, n_knots))
        self.inv_width = n_knots / (grid_range[1] - grid_range[0])
        self.spline_linear = nn.Linear(in_dim * n_knots, out_dim)
        self.base_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        diff = (x.unsqueeze(-1) - self.knots) * self.inv_width
        basis = torch.exp(-0.5 * diff * diff).flatten(start_dim=-2)
        return self.base_linear(F.silu(x)) + self.spline_linear(basis)


class KANSplineModel(nn.Module):
    """Arch 6: Kolmogorov-Arnold RBF-Spline Network (DL_for_EPF_review) with Decoupled Dual Towers."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 128, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        # Global horizon summary via lightweight spline pooling so each step sees intraday context
        self.ctx_proj = nn.Linear(dh, 16)
        in_full = raw_dim + 32  # + mean & max-min context of 40-step horizon

        self.med_tower = nn.Sequential(
            RBFSplineLayer(in_full, d_model, n_knots=10),
            nn.Dropout(dropout),
            GatedResBlock(d_model, d_model * 2, dropout),
            GatedResBlock(d_model, d_model * 2, dropout),
        )
        self.dist_tower = nn.Sequential(
            RBFSplineLayer(in_full, d_model, n_knots=10),
            nn.Dropout(dropout),
            GatedResBlock(d_model, d_model * 2, dropout),
            GatedResBlock(d_model, d_model * 2, dropout),
        )
        self.head = DecoupledDualHead(d_model, d_model, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)
        c = F.silu(self.ctx_proj(x_step))  # (B, 40, 16)
        c_summary = torch.cat([c.mean(dim=1, keepdim=True), c.std(dim=1, keepdim=True)], dim=-1).expand(-1, N_STEPS, -1)
        aug_in = torch.cat([raw_in, c_summary], dim=-1)
        return self.head(self.med_tower(aug_in), self.dist_tower(aug_in), raw_in)


class PatchTSTHorizonModel(nn.Module):
    """Arch 7: PatchTST Multi-Scale Hourly Patch Transformer across 40 quarter-hour steps."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 112, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        # 10 non-overlapping 1-hour patches (4 steps each) + 1 origin-state token = 11 tokens
        self.n_patches = 10
        self.patch_len = 4
        self.state_tok = nn.Linear(ds, d_model)
        self.patch_emb = nn.Linear(self.patch_len * dh, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.med_enc = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.dist_enc = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)

        self.med_unpatch = nn.Sequential(
            nn.Linear(d_model, self.patch_len * d_model),
            nn.SiLU(),
        )
        self.dist_unpatch = nn.Sequential(
            nn.Linear(d_model, self.patch_len * d_model),
            nn.SiLU(),
        )
        self.med_refine = GatedResBlock(d_model + raw_dim, d_model * 2, dropout, out_dim=d_model)
        self.dist_refine = GatedResBlock(d_model + raw_dim, d_model * 2, dropout, out_dim=d_model)
        self.head = DecoupledDualHead(d_model, d_model, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        bsz = x_state.size(0)
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)

        patches = x_step.reshape(bsz, self.n_patches, self.patch_len * x_step.size(-1))
        toks = torch.cat([self.state_tok(x_state).unsqueeze(1), self.patch_emb(patches)], dim=1) + self.pos_emb

        m_p = self.med_enc(toks)[:, 1:, :]  # (B, 10, d_model)
        d_p = self.dist_enc(toks)[:, 1:, :]

        d_model = m_p.size(-1)
        m_step = self.med_unpatch(m_p).reshape(bsz, N_STEPS, d_model)
        d_step = self.dist_unpatch(d_p).reshape(bsz, N_STEPS, d_model)

        hm = self.med_refine(torch.cat([m_step, raw_in], dim=-1))
        hd = self.dist_refine(torch.cat([d_step, raw_in], dim=-1))
        return self.head(hm, hd, raw_in)


class iTransformerVariateModel(nn.Module):
    """Arch 8: iTransformer Inverted Variate-Token Attention (Liu et al., 2024; PriceFM)."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 112, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        # Each horizon channel k in 1..Dh has a 40-step profile -> embedded as a variate token!
        # Plus each state channel j in 1..Ds embedded via scalar-to-token projection -> (Ds + Dh) variate tokens!
        self.state_var_proj = nn.Parameter(torch.randn(ds, d_model) * 0.05)
        self.state_var_bias = nn.Parameter(torch.zeros(ds, d_model))
        self.step_var_emb = nn.Linear(N_STEPS, d_model)
        self.var_id_emb = nn.Parameter(torch.randn(1, ds + dh, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.var_attn = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)

        # Project attended horizon variate tokens back to 40 steps
        self.to_med_steps = nn.Linear(d_model, N_STEPS)
        self.to_dist_steps = nn.Linear(d_model, N_STEPS)

        self.med_post = GatedResBlock(dh + raw_dim, 160, dropout, out_dim=112)
        self.dist_post = GatedResBlock(dh + raw_dim, 160, dropout, out_dim=112)
        self.head = DecoupledDualHead(112, 112, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)

        # Variate tokens: (B, Ds, d_model) and (B, Dh, d_model)
        v_state = x_state.unsqueeze(-1) * self.state_var_proj.unsqueeze(0) + self.state_var_bias.unsqueeze(0)
        v_step = self.step_var_emb(x_step.transpose(1, 2))  # (B, Dh, d_model)
        v_all = torch.cat([v_state, v_step], dim=1) + self.var_id_emb

        v_out = self.var_attn(v_all)
        v_step_out = v_out[:, x_state.size(-1) :, :]  # (B, Dh, d_model)

        m_traj = self.to_med_steps(v_step_out).transpose(1, 2)   # (B, 40, Dh)
        d_traj = self.to_dist_steps(v_step_out).transpose(1, 2)  # (B, 40, Dh)

        hm = self.med_post(torch.cat([m_traj, raw_in], dim=-1))
        hd = self.dist_post(torch.cat([d_traj, raw_in], dim=-1))
        return self.head(hm, hd, raw_in)


class _TiDETower(nn.Module):
    def __init__(self, ds: int, dh: int, proj_dim: int = 12, enc_dim: int = 160, dec_step_dim: int = 40,
                 out_dim: int = 112, dropout: float = 0.18):
        super().__init__()
        self.cov_proj = GatedResBlock(dh, 64, dropout, out_dim=proj_dim)
        self.enc_in = nn.Linear(ds + N_STEPS * proj_dim, enc_dim)
        self.enc = nn.Sequential(GatedResBlock(enc_dim, enc_dim * 2, dropout), GatedResBlock(enc_dim, enc_dim * 2, dropout))
        self.dec = GatedResBlock(enc_dim, enc_dim * 2, dropout, out_dim=N_STEPS * dec_step_dim)
        self.dec_step_dim = dec_step_dim
        self.step_emb = nn.Embedding(N_STEPS, 16)
        self.temp_dec = GatedResBlock(dec_step_dim + proj_dim + dh + ds + 16, 160, dropout, out_dim=out_dim)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor) -> torch.Tensor:
        bsz = x_state.size(0)
        cp = self.cov_proj(x_step)
        e = self.enc(F.silu(self.enc_in(torch.cat([x_state, cp.reshape(bsz, -1)], dim=-1))))
        g = self.dec(e).view(bsz, N_STEPS, self.dec_step_dim)
        s_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        sid = self.step_emb(torch.arange(N_STEPS, device=x_state.device).unsqueeze(0).expand(bsz, -1))
        return self.temp_dec(torch.cat([g, cp, x_step, s_rep, sid], dim=-1))


class DecoupledTiDEModel(nn.Module):
    """Arch 9: Task-Decoupled Dual-Tower TiDE (Pure L1 Median TiDE + Distributional TiDE)."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        self.med_tide = _TiDETower(ds, dh, out_dim=112, dropout=dropout)
        self.dist_tide = _TiDETower(ds, dh, out_dim=112, dropout=dropout)
        self.head = DecoupledDualHead(112, 112, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)
        return self.head(self.med_tide(x_state, x_step), self.dist_tide(x_state, x_step), raw_in)


FRONTIER_ARCHITECTURES = ("kan_spline_net", "patch_tst_horizon", "itransformer_variate", "decoupled_tide_l1")


def build_frontier_architecture(
    arch: str,
    ds: int,
    dh: int,
    q_init: np.ndarray,
    dir_lp: np.ndarray,
    spike_lp: np.ndarray,
) -> nn.Module:
    if arch == "kan_spline_net":
        return KANSplineModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "patch_tst_horizon":
        return PatchTSTHorizonModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "itransformer_variate":
        return iTransformerVariateModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "decoupled_tide_l1":
        return DecoupledTiDEModel(ds, dh, q_init, dir_lp, spike_lp)
    raise ValueError(f"Unknown frontier architecture: {arch}")
