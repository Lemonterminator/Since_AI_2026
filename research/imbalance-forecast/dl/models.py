"""Five controlled Deep Learning architectures for 40-step probabilistic imbalance forecasting.

All architectures consume strictly the exact same controlled feature tensors
``(X_state [B, Ds], X_step [B, 40, Dh])`` produced from ``submission/panel.py``:

1. ``tabular_resnet``  : Pointwise Tabular ResNet + Fourier Embeddings + Hierarchical Head
                         (strict row-wise control: zero cross-step interaction, identical view to GBDT).
2. ``tide``            : TiDE Trajectory Encoder-Decoder (Das et al., 2023) + Hierarchical Head
                         (encodes the full 40-step future spot/wind/net-load context jointly, decodes per step).
3. ``conv_bigru``      : Multi-Scale 1D Conv + Origin-FiLM Modulated BiGRU + Hierarchical Head
                         (extracts local intraday ramps/curvatures + bidirectional horizon propagation).
4. ``transformer_moe`` : Horizon Self-Attention + PriceFM/Review Regime-Gated Mixture-of-Experts (M=4)
                         (cross-step attention over [STATE, step_1..step_40] + soft regime expert routing).
5. ``lgbm_tide_corrector``: Wärtsilä Hybrid GBDT + TiDE Trajectory Residual Corrector
                         (learns cross-step median, non-crossing quantile-spacing, and logit corrections
                         on top of walk-forward OOF LightGBM predictions; zero-initialized).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_STEPS = 40
N_QUANTILES = 19
MEDIAN_IDX = 9  # QGRID[9] == 0.50


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-3)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


class GatedResBlock(nn.Module):
    """Pre-Norm Residual MLP block with SiLU gating and dropout (TiDE basic building block)."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.15, out_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        out_dim = out_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if dim == out_dim else nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return self.skip(x) + h


class FourierFeatureEmbed(nn.Module):
    """Learnable periodic feature embedding (Gorishniy et al., NeurIPS 2022) for tabular inputs."""

    def __init__(self, in_dim: int, n_freqs: int = 8, out_dim: int = 128):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(in_dim, n_freqs) * 0.5)
        self.proj = nn.Linear(in_dim * (1 + 2 * n_freqs), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.coeffs  # (..., D, F)
        emb = torch.cat([x.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)
        return F.silu(self.proj(emb.flatten(start_dim=-2)))


class HierarchicalProbHead(nn.Module):
    """OrderFusion / PriceFM Hierarchical Non-Crossing 19-Quantile + Multi-Task Classification Head.

    Guarantees q_05 <= q_10 <= ... <= q_50 <= ... <= q_90 <= q_95 identically (AQCR == 0%).
    Includes a direct linear highway from raw step inputs for stable optimization.
    """

    def __init__(
        self,
        in_dim: int,
        raw_dim: int,
        q_init: np.ndarray,
        dir_log_prior: np.ndarray,
        spike_log_prior: np.ndarray,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.q_proj = nn.Linear(in_dim, N_QUANTILES)
        self.q_highway = nn.Linear(raw_dim, N_QUANTILES, bias=False)

        self.dir_proj = nn.Linear(in_dim, 3)
        self.dir_highway = nn.Linear(raw_dim, 3, bias=False)

        self.spk_proj = nn.Linear(in_dim, 3)
        self.spk_highway = nn.Linear(raw_dim, 3, bias=False)

        self._init_priors(q_init, dir_log_prior, spike_log_prior)

    def _init_priors(self, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray) -> None:
        for layer in (self.q_proj, self.q_highway, self.dir_proj, self.dir_highway, self.spk_proj, self.spk_highway):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        b = np.zeros(N_QUANTILES, dtype=np.float32)
        b[MEDIAN_IDX] = float(q_init[MEDIAN_IDX])
        # Upper spacings j = 10..18
        b[MEDIAN_IDX + 1 :] = _inv_softplus(np.diff(q_init[MEDIAN_IDX:]))
        # Lower spacings j = 0..8 (each represents q[j+1] - q[j])
        b[:MEDIAN_IDX] = _inv_softplus(np.diff(q_init[: MEDIAN_IDX + 1]))

        with torch.no_grad():
            self.q_proj.bias.copy_(torch.from_numpy(b))
            self.dir_proj.bias.copy_(torch.from_numpy(dir_lp))
            self.spk_proj.bias.copy_(torch.from_numpy(spk_lp))

    @staticmethod
    def raw_to_quantiles(raw_q: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained ``(..., 19)`` tensor into strictly monotone quantiles."""
        median = raw_q[..., MEDIAN_IDX : MEDIAN_IDX + 1]  # (..., 1)
        up_inc = F.softplus(raw_q[..., MEDIAN_IDX + 1 :]) + 1e-4  # (..., 9)
        up_q = median + torch.cumsum(up_inc, dim=-1)

        # Lower branch: raw_q[..., 8] is closest to median, raw_q[..., 0] is farthest
        lo_inc_outward = F.softplus(torch.flip(raw_q[..., :MEDIAN_IDX], dims=[-1])) + 1e-4
        lo_q_outward = median - torch.cumsum(lo_inc_outward, dim=-1)
        lo_q = torch.flip(lo_q_outward, dims=[-1])
        return torch.cat([lo_q, median, up_q], dim=-1)

    def forward(self, h: torch.Tensor, raw_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_n = self.norm(h)
        raw_q = self.q_proj(h_n) + self.q_highway(raw_in)
        q_scaled = self.raw_to_quantiles(raw_q)
        dir_logits = self.dir_proj(h_n) + self.dir_highway(raw_in)
        spk_logits = self.spk_proj(h_n) + self.spk_highway(raw_in)
        return q_scaled, dir_logits, spk_logits


class TabularResNetModel(nn.Module):
    """Arch 1: Strict row-wise Tabular ResNet + Fourier Embeddings + Hierarchical Head."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 128, n_blocks: int = 3, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        self.embed = FourierFeatureEmbed(raw_dim, n_freqs=6, out_dim=d_model)
        self.blocks = nn.Sequential(*[GatedResBlock(d_model, d_model * 2, dropout) for _ in range(n_blocks)])
        self.head = HierarchicalProbHead(d_model, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)  #Processed strictly independently per step!
        h = self.blocks(self.embed(raw_in))
        return self.head(h, raw_in)


class TiDETrajectoryModel(nn.Module):
    """Arch 2: Google Research TiDE adapted for 40-step known-covariate trajectory forecasting."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 proj_dim: int = 12, enc_dim: int = 192, dec_step_dim: int = 48, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        # 1. TiDE Feature Projection for dynamic future covariates across all 40 steps
        self.cov_proj = GatedResBlock(dh, hidden_dim=64, dropout=dropout, out_dim=proj_dim)
        # 2. TiDE Dense Encoder over [origin_state ; flattened projected 40-step covariates]
        flat_dim = ds + N_STEPS * proj_dim
        self.enc_in = nn.Linear(flat_dim, enc_dim)
        self.encoder = nn.Sequential(
            GatedResBlock(enc_dim, enc_dim * 2, dropout),
            GatedResBlock(enc_dim, enc_dim * 2, dropout),
        )
        # 3. TiDE Dense Decoder -> reshaped to (B, 40, dec_step_dim)
        self.dense_dec = GatedResBlock(enc_dim, enc_dim * 2, dropout, out_dim=N_STEPS * dec_step_dim)
        self.dec_step_dim = dec_step_dim
        self.step_emb = nn.Embedding(N_STEPS, 16)
        # 4. TiDE Temporal Decoder combining per-step decoded latent + local step covariates + origin state
        temp_in = dec_step_dim + proj_dim + dh + ds + 16
        self.temporal_dec = GatedResBlock(temp_in, hidden_dim=160, dropout=dropout, out_dim=128)
        self.head = HierarchicalProbHead(128, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        bsz = x_state.size(0)
        cov_p = self.cov_proj(x_step)  # (B, 40, proj_dim)
        flat = torch.cat([x_state, cov_p.reshape(bsz, -1)], dim=-1)
        e = self.encoder(F.silu(self.enc_in(flat)))
        g = self.dense_dec(e).view(bsz, N_STEPS, self.dec_step_dim)

        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        step_ids = torch.arange(N_STEPS, device=x_state.device).unsqueeze(0).expand(bsz, -1)
        s_emb = self.step_emb(step_ids)

        raw_in = torch.cat([state_rep, x_step], dim=-1)
        dec_in = torch.cat([g, cov_p, x_step, state_rep, s_emb], dim=-1)
        h = self.temporal_dec(dec_in)
        return self.head(h, raw_in)


class ConvBiGRUModel(nn.Module):
    """Arch 3: Multi-Scale 1D Horizon Conv + Origin-FiLM Modulated BiGRU (Wärtsilä Memo Arch #3)."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 96, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        self.state_enc = nn.Sequential(
            nn.Linear(ds, d_model),
            nn.SiLU(),
            GatedResBlock(d_model, d_model * 2, dropout),
        )
        # Multi-scale 1D convolutions along the 40-step horizon to capture local ramps & curvature
        self.conv3 = nn.Conv1d(dh, d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dh, d_model // 2, kernel_size=5, padding=2)
        # FiLM conditioning from origin imbalance state
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        self.h0_proj = nn.Linear(d_model, 4 * (d_model // 2))  # 2 layers x 2 directions

        self.bigru = nn.GRU(
            input_size=d_model + ds,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.post = GatedResBlock(d_model * 2, d_model * 2, dropout, out_dim=128)
        self.head = HierarchicalProbHead(128, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        bsz = x_state.size(0)
        s_lat = self.state_enc(x_state)  # (B, d_model)

        xt = x_step.transpose(1, 2)  # (B, Dh, 40)
        c = F.silu(torch.cat([self.conv3(xt), self.conv5(xt)], dim=1)).transpose(1, 2)  # (B, 40, d_model)
        gamma = 1.0 + 0.5 * torch.tanh(self.film_gamma(s_lat)).unsqueeze(1)
        beta = self.film_beta(s_lat).unsqueeze(1)
        c_mod = gamma * c + beta

        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        gru_in = torch.cat([c_mod, state_rep], dim=-1)
        h0 = torch.tanh(self.h0_proj(s_lat)).view(bsz, 4, -1).permute(1, 0, 2).contiguous()
        self.bigru.flatten_parameters()
        seq_out, _ = self.bigru(gru_in, h0)  # (B, 40, d_model)

        h = self.post(torch.cat([seq_out, c_mod], dim=-1))
        raw_in = torch.cat([state_rep, x_step], dim=-1)
        return self.head(h, raw_in)


class HorizonTransformerMoEModel(nn.Module):
    """Arch 4: Horizon Self-Attention + Regime-Gated Mixture-of-Experts (PriceFM + Review + Memo #4)."""

    def __init__(self, ds: int, dh: int, q_init: np.ndarray, dir_lp: np.ndarray, spk_lp: np.ndarray,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 2, n_experts: int = 4, dropout: float = 0.18):
        super().__init__()
        raw_dim = ds + dh
        self.state_tok = nn.Linear(ds, d_model)
        self.step_tok = nn.Linear(ds + dh, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + N_STEPS, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        # PriceFM / DL-Review inspired Shared Mixture-of-Experts with soft regime router
        self.n_experts = n_experts
        self.router = nn.Sequential(
            nn.LayerNorm(d_model + raw_dim),
            nn.Linear(d_model + raw_dim, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model + raw_dim),
                nn.Linear(d_model + raw_dim, d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_experts)
        ])
        self.head = HierarchicalProbHead(d_model, raw_dim, q_init, dir_lp, spk_lp)

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor | None = None):
        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        raw_in = torch.cat([state_rep, x_step], dim=-1)

        t0 = self.state_tok(x_state).unsqueeze(1)  # (B, 1, d_model)
        th = self.step_tok(raw_in)                 # (B, 40, d_model)
        tokens = torch.cat([t0, th], dim=1) + self.pos_emb
        attn_out = self.transformer(tokens)[:, 1:, :]  # (B, 40, d_model)

        joint = torch.cat([attn_out, raw_in], dim=-1)
        weights = F.softmax(self.router(joint), dim=-1)  # (B, 40, M)
        exp_stack = torch.stack([exp(joint) for exp in self.experts], dim=-2)  # (B, 40, M, d_model)
        moe_out = attn_out + (weights.unsqueeze(-1) * exp_stack).sum(dim=-2)
        return self.head(moe_out, raw_in)


class TiDEResidualCorrectorModel(nn.Module):
    """Arch 5: Wärtsilä Hybrid GBDT + TiDE Trajectory Residual & Calibration Corrector.

    ``extra_step`` shape ``(B, 40, 25)`` contains strictly out-of-fold / test LightGBM predictions:
      - ``extra_step[..., 0:19]``  : LightGBM sorted 19 quantiles in scaled spread units
      - ``extra_step[..., 19:22]`` : LightGBM direction log-probabilities (3)
      - ``extra_step[..., 22:25]`` : LightGBM spike log-probabilities (3)

    Zero-initialized output layer guarantees exact identity with LightGBM at initialization,
    learning only genuine cross-step trajectory and calibration residuals.
    """

    def __init__(self, ds: int, dh: int, d_extra: int = 25, proj_dim: int = 14,
                 enc_dim: int = 160, dec_step_dim: int = 32, dropout: float = 0.22):
        super().__init__()
        step_in_dim = dh + d_extra + 1  # +1 for local cross-step LGBM median ramp
        self.cov_proj = GatedResBlock(step_in_dim, hidden_dim=64, dropout=dropout, out_dim=proj_dim)
        flat_dim = ds + N_STEPS * proj_dim
        self.enc_in = nn.Linear(flat_dim, enc_dim)
        self.encoder = GatedResBlock(enc_dim, enc_dim * 2, dropout)
        self.dense_dec = GatedResBlock(enc_dim, enc_dim * 2, dropout, out_dim=N_STEPS * dec_step_dim)
        self.dec_step_dim = dec_step_dim

        temp_in = dec_step_dim + proj_dim + step_in_dim + ds
        self.temporal_dec = nn.Sequential(
            GatedResBlock(temp_in, hidden_dim=128, dropout=dropout, out_dim=96),
            nn.LayerNorm(96),
        )
        # 1 (median shift) + 9 (upper log-scale) + 9 (lower log-scale) + 3 (dir logit) + 3 (spk logit) = 25
        self.delta_out = nn.Linear(96, 25)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        self.gate = nn.Parameter(torch.tensor(0.15))

    def forward(self, x_state: torch.Tensor, x_step: torch.Tensor, extra_step: torch.Tensor):
        bsz = x_state.size(0)
        q_base = extra_step[..., :19]
        dir_lp_base = extra_step[..., 19:22]
        spk_lp_base = extra_step[..., 22:25]

        med_base = q_base[..., MEDIAN_IDX : MEDIAN_IDX + 1]  # (B, 40, 1)
        med_ramp = torch.cat([torch.zeros_like(med_base[:, :1, :]), med_base[:, 1:, :] - med_base[:, :-1, :]], dim=1)
        step_aug = torch.cat([x_step, torch.arcsinh(q_base), dir_lp_base / 3.0, spk_lp_base / 3.0, med_ramp], dim=-1)

        cov_p = self.cov_proj(step_aug)
        flat = torch.cat([x_state, cov_p.reshape(bsz, -1)], dim=-1)
        e = self.encoder(F.silu(self.enc_in(flat)))
        g = self.dense_dec(e).view(bsz, N_STEPS, self.dec_step_dim)

        state_rep = x_state.unsqueeze(1).expand(-1, N_STEPS, -1)
        h = self.temporal_dec(torch.cat([g, cov_p, step_aug, state_rep], dim=-1))
        deltas = self.gate * self.delta_out(h)

        d_med = deltas[..., 0:1]
        d_up = deltas[..., 1:10]
        d_lo = deltas[..., 10:19]
        d_dir = deltas[..., 19:22]
        d_spk = deltas[..., 22:25]

        # Non-crossing hierarchical spacing update around LightGBM's sorted quantiles
        up_diffs = torch.clamp_min(q_base[..., MEDIAN_IDX + 1 :] - q_base[..., MEDIAN_IDX:-1], 1e-4)
        lo_diffs_out = torch.flip(
            torch.clamp_min(q_base[..., 1 : MEDIAN_IDX + 1] - q_base[..., :MEDIAN_IDX], 1e-4),
            dims=[-1],
        )

        new_med = med_base + d_med
        new_up = new_med + torch.cumsum(up_diffs * torch.exp(0.6 * torch.tanh(d_up)), dim=-1)
        new_lo_out = new_med - torch.cumsum(lo_diffs_out * torch.exp(0.6 * torch.tanh(d_lo)), dim=-1)
        new_lo = torch.flip(new_lo_out, dims=[-1])
        q_scaled = torch.cat([new_lo, new_med, new_up], dim=-1)

        return q_scaled, dir_lp_base + d_dir, spk_lp_base + d_spk


PURE_DL_ARCHS = ("tabular_resnet", "tide", "conv_bigru", "transformer_moe")
ARCHITECTURES = (*PURE_DL_ARCHS, "lgbm_tide_corrector")


def build_architecture(
    arch: str,
    ds: int,
    dh: int,
    q_init: np.ndarray,
    dir_lp: np.ndarray,
    spike_lp: np.ndarray,
) -> nn.Module:
    if arch == "tabular_resnet":
        return TabularResNetModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "tide":
        return TiDETrajectoryModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "conv_bigru":
        return ConvBiGRUModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "transformer_moe":
        return HorizonTransformerMoEModel(ds, dh, q_init, dir_lp, spike_lp)
    if arch == "lgbm_tide_corrector":
        return TiDEResidualCorrectorModel(ds, dh)
    raise ValueError(f"Unknown architecture: {arch}")
