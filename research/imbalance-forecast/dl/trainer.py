"""GPU-resident trainer, strict temporal two-pass calibration, and OOF hybrid pipeline."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (
    N_STEPS,
    ControlledScaler,
    TrajectoryBatch,
    build_masked_trajectories,
)
from .models import MEDIAN_IDX, N_QUANTILES, build_architecture

import make_submission as M  # loaded via dl.features sys.path
import panel as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def compute_multitask_loss(
    q_scaled: torch.Tensor,      # (B, 40, 19)
    dir_logits: torch.Tensor,    # (B, 40, 3)
    spk_logits: torch.Tensor,    # (B, 40, 3)
    y_scaled: torch.Tensor,      # (B, 40)
    y_dir: torch.Tensor,         # (B, 40) int64
    y_spk: torch.Tensor,         # (B, 40) int64
    mask: torch.Tensor,          # (B, 40) bool
    taus: torch.Tensor,          # (19,)
) -> torch.Tensor:
    """Strictly proper composite loss aligned with official score.py weights."""
    valid = mask.reshape(-1)
    if not valid.any():
        return q_scaled.sum() * 0.0

    qv = q_scaled.reshape(-1, N_QUANTILES)[valid]
    yv = y_scaled.reshape(-1, 1)[valid]
    dv = dir_logits.reshape(-1, 3)[valid]
    sv = spk_logits.reshape(-1, 3)[valid]
    yd = y_dir.reshape(-1)[valid]
    ys = y_spk.reshape(-1)[valid]

    # 1. Dense 19-quantile pinball + explicit score.py P50 (MAE) and P10/P90 terms
    diff = yv - qv  # (N_val, 19)
    pin_all = torch.maximum(taus * diff, (taus - 1.0) * diff)
    loss_grid = pin_all.mean()
    loss_p50 = (yv.squeeze(-1) - qv[:, MEDIAN_IDX]).abs().mean()
    loss_p10_90 = 0.5 * (pin_all[:, 1].mean() + pin_all[:, 17].mean())
    loss_q = 0.45 * loss_grid + 0.35 * loss_p50 + 0.20 * loss_p10_90

    # 2. Direction 3-class: 0.5 CE + 0.5 Brier (range 0..2)
    dir_probs = F.softmax(dv, dim=-1)
    dir_oh = F.one_hot(yd, num_classes=3).float()
    brier_dir = ((dir_probs - dir_oh) ** 2).sum(dim=-1).mean()
    ce_dir = F.cross_entropy(dv, yd)
    loss_dir = 0.35 * ce_dir + 0.65 * brier_dir

    # 3. Spike 3-class: 0.5 CE + 0.5 mean(Brier_high, Brier_low) matching score.py
    spk_probs = F.softmax(sv, dim=-1)
    spk_oh = F.one_hot(ys, num_classes=3).float()
    brier_spk = 0.5 * (((spk_probs[:, 1] - spk_oh[:, 1]) ** 2).mean()
                       + ((spk_probs[:, 2] - spk_oh[:, 2]) ** 2).mean())
    ce_spk = F.cross_entropy(sv, ys)
    loss_spk = 0.35 * ce_spk + 1.50 * brier_spk

    return loss_q + 0.35 * loss_dir + 0.45 * loss_spk


class EMAWeights:
    """Exponential moving average of model weights for variance reduction on noisy tails."""

    def __init__(self, model: nn.Module, decay: float = 0.985):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def clone_model(self, template: nn.Module) -> nn.Module:
        m = copy.deepcopy(template)
        m.load_state_dict(self.shadow)
        m.eval()
        return m


def _to_gpu(batch: TrajectoryBatch, target_scale: float, device: torch.device) -> dict[str, torch.Tensor | None]:
    has_any = batch.mask.any(axis=1) if batch.mask.any() else np.ones(len(batch.origins), dtype=bool)
    return {
        "x_state": torch.from_numpy(batch.X_state[has_any]).to(device),
        "x_step": torch.from_numpy(batch.X_step[has_any]).to(device),
        "extra_step": (
            torch.from_numpy(batch.extra_step[has_any]).to(device)
            if batch.extra_step is not None else None
        ),
        "y_scaled": torch.from_numpy(batch.y_spread[has_any] / np.float32(target_scale)).to(device),
        "y_dir": torch.from_numpy(batch.y_dir[has_any]).to(device),
        "y_spk": torch.from_numpy(batch.y_spike[has_any]).to(device),
        "mask": torch.from_numpy(batch.mask[has_any]).to(device),
    }


def _train_single_seed(
    arch: str,
    scaler: ControlledScaler,
    tr_gpu: dict[str, torch.Tensor | None],
    val_gpu: dict[str, torch.Tensor | None] | None,
    seed: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds = tr_gpu["x_state"].shape[-1]
    dh = tr_gpu["x_step"].shape[-1]
    model = build_architecture(
        arch, ds, dh, scaler.target_q_init, scaler.dir_log_prior, scaler.spike_log_prior
    ).to(device)

    n_orig = tr_gpu["x_state"].shape[0]
    batch_size = 32 if n_orig < 3500 else 128
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Keep T_max=max_epochs so Pass-2 follows the exact same LR curve up to best_epoch as Pass-1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    ema = EMAWeights(model, decay=0.98 if n_orig < 3500 else 0.99)
    taus = torch.tensor(M.QGRID, dtype=torch.float32, device=device).view(1, -1)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)

    val_losses: list[float] = []
    best_val = float("inf")
    best_state = None
    min_ep = min(6, epochs)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_orig, generator=gen, device=device)
        for i in range(0, n_orig, batch_size):
            idx = perm[i : i + batch_size]
            ex = tr_gpu["extra_step"][idx] if tr_gpu["extra_step"] is not None else None
            q_s, d_l, s_l = model(tr_gpu["x_state"][idx], tr_gpu["x_step"][idx], ex)
            loss = compute_multitask_loss(
                q_s, d_l, s_l,
                tr_gpu["y_scaled"][idx], tr_gpu["y_dir"][idx], tr_gpu["y_spk"][idx],
                tr_gpu["mask"][idx], taus,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()

        if val_gpu is not None:
            eval_m = ema.clone_model(model)
            with torch.no_grad():
                q_v, d_v, s_v = eval_m(val_gpu["x_state"], val_gpu["x_step"], val_gpu["extra_step"])
                vl = float(compute_multitask_loss(
                    q_v, d_v, s_v,
                    val_gpu["y_scaled"], val_gpu["y_dir"], val_gpu["y_spk"],
                    val_gpu["mask"], taus,
                ).item())
            val_losses.append(vl)
            if ep >= min_ep and vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(ema.shadow)

    if best_state is not None and fixed_epochs is None:
        model.load_state_dict(best_state)
        model.eval()
        return model, val_losses
    return ema.clone_model(model), val_losses


@torch.no_grad()
def predict_models_raw(
    models: list[nn.Module],
    scaler: ControlledScaler,
    batch: TrajectoryBatch,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(grid_spread [N,40,19], dir_probs [N,40,3], spk_probs [N,40,3], p50_epistemic_std [N,40])``."""
    xs = torch.from_numpy(batch.X_state).to(device)
    xh = torch.from_numpy(batch.X_step).to(device)
    ex = torch.from_numpy(batch.extra_step).to(device) if batch.extra_step is not None else None

    q_list, d_list, s_list = [], [], []
    for m in models:
        m.eval()
        qs, dl, sl = m(xs, xh, ex)
        q_list.append(qs.cpu().numpy() * scaler.target_scale)
        d_list.append(F.softmax(dl, dim=-1).cpu().numpy())
        s_list.append(F.softmax(sl, dim=-1).cpu().numpy())

    q_stack = np.stack(q_list, axis=0)  # (S, N, 40, 19)
    q_mean = q_stack.mean(axis=0)       # convex sum of monotone vectors -> strictly monotone!
    p50_std = q_stack[..., MEDIAN_IDX].std(axis=0) if len(models) > 1 else np.zeros_like(q_mean[..., 0])
    d_mean = np.mean(d_list, axis=0)
    s_mean = np.mean(s_list, axis=0)
    return q_mean, d_mean, s_mean, p50_std


def assemble_prediction_df(
    df_rows: pd.DataFrame,
    grid_spread: np.ndarray,   # (N, 40, 19)
    dir_probs: np.ndarray,     # (N, 40, 3)
    spk_probs: np.ndarray,     # (N, 40, 3)
    scale: dict[tuple[int, int], tuple[float, float]],
    spike_source: str = "mean",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert raw trajectory arrays to exact ``make_submission.py`` prediction DataFrame + price grid."""
    flat_grid = grid_spread.reshape(-1, N_QUANTILES)
    flat_dir = dir_probs.reshape(-1, 3)
    flat_spk = spk_probs.reshape(-1, 3)

    q50 = flat_grid[:, MEDIAN_IDX]
    sl_row = np.ones(len(df_rows), dtype=np.float64)
    sh_row = np.ones(len(df_rows), dtype=np.float64)
    for (a, b), (sl, sh) in scale.items():
        sel = df_rows["step"].between(a, b).to_numpy()
        sl_row[sel], sh_row[sel] = sl, sh

    dev = flat_grid - q50[:, None]
    scaled_spread_grid = q50[:, None] + np.where(dev < 0.0, dev * sl_row[:, None], dev * sh_row[:, None])

    p10_s = scaled_spread_grid[:, 1]
    p50_s = scaled_spread_grid[:, MEDIAN_IDX]
    p90_s = scaled_spread_grid[:, 17]

    if spike_source in ("cdf", "mean"):
        cdf_hi = np.array([np.interp(P.SPIKE_MARGIN, g, M.QGRID) for g in scaled_spread_grid])
        cdf_lo = np.array([np.interp(-P.SPIKE_MARGIN, g, M.QGRID) for g in scaled_spread_grid])
        p_hi = np.where(scaled_spread_grid[:, -1] < P.SPIKE_MARGIN, 0.025, 1.0 - cdf_hi)
        p_lo = np.where(scaled_spread_grid[:, 0] > -P.SPIKE_MARGIN, 0.025, cdf_lo)
        cdf_probs = np.column_stack([np.clip(1.0 - p_hi - p_lo, 0.0, 1.0), p_hi, p_lo])
        if spike_source == "cdf":
            flat_spk = cdf_probs
        else:
            flat_spk = 0.5 * (flat_spk + cdf_probs)

    spot = df_rows["spot_level"].to_numpy(dtype=np.float64)
    price_grid = scaled_spread_grid + spot[:, None]
    out = pd.DataFrame(
        {
            "p10": spot + p10_s,
            "p50": spot + p50_s,
            "p90": spot + p90_s,
            "p_up": flat_dir[:, 0],
            "p_down": flat_dir[:, 1],
            "p_none": flat_dir[:, 2],
            "p_spike_high": flat_spk[:, 1],
            "p_spike_low": flat_spk[:, 2],
        },
        index=df_rows.index,
    )
    return out, price_grid


def fit_scale_from_holdout(
    hold_df: pd.DataFrame,
    grid_spread: np.ndarray,
    hold_mask: np.ndarray,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Exact same 14-day inner holdout interval width grid search as ``make_submission.fit_interval_scale``."""
    flat_mask = hold_mask.reshape(-1)
    flat_grid = grid_spread.reshape(-1, N_QUANTILES)
    spot = hold_df["spot_level"].to_numpy(dtype=np.float64)
    y = hold_df["imbalance_price"].to_numpy(dtype=np.float64)
    steps = hold_df["step"].to_numpy()

    p50 = spot + flat_grid[:, MEDIAN_IDX]
    lo = flat_grid[:, MEDIAN_IDX] - flat_grid[:, 1]
    hi = flat_grid[:, 17] - flat_grid[:, MEDIAN_IDX]

    grid_s = np.linspace(0.6, 2.5, 39)
    scale: dict[tuple[int, int], tuple[float, float]] = {}
    for a, b in M.BANDS:
        sel = flat_mask & (steps >= a) & (steps <= b) & (~np.isnan(y))
        if sel.sum() < 40:
            scale[(a, b)] = (1.0, 1.0)
            continue
        sl = grid_s[int(np.argmin([M.pinball(y[sel], p50[sel] - s * lo[sel], 0.1) for s in grid_s]))]
        sh = grid_s[int(np.argmin([M.pinball(y[sel], p50[sel] + s * hi[sel], 0.9) for s in grid_s]))]
        scale[(a, b)] = (float(sl), float(sh))
    return scale


# ---------------------------------------------------------------------------
# Walk-forward Out-of-Fold LightGBM features for Arch 5 (Hybrid Corrector)
# ---------------------------------------------------------------------------

def extract_lgbm_extra_features(
    fc: M.Forecaster,
    df_sorted: pd.DataFrame,
    target_scale: float,
) -> np.ndarray:
    """Extract ``(N_orig, 40, 25)`` trajectory features from a fitted LightGBM ``Forecaster``."""
    X = df_sorted[fc.features]
    Xp = df_sorted[fc.prob_features]
    pd_dir = np.clip(fc.direction.predict(Xp), 1e-5, 1.0)
    pd_dir = pd_dir / pd_dir.sum(axis=1, keepdims=True)

    raw_grid = np.sort(
        np.column_stack([fc.q[a].predict(X if a == 0.5 else Xp) for a in M.QGRID]),
        axis=1,
    )
    cdf_hi = np.array([np.interp(P.SPIKE_MARGIN, g, M.QGRID) for g in raw_grid])
    cdf_lo = np.array([np.interp(-P.SPIKE_MARGIN, g, M.QGRID) for g in raw_grid])
    p_hi = np.where(raw_grid[:, -1] < P.SPIKE_MARGIN, 0.025, 1.0 - cdf_hi)
    p_lo = np.where(raw_grid[:, 0] > -P.SPIKE_MARGIN, 0.025, cdf_lo)
    pd_spk = np.column_stack([np.clip(1.0 - p_hi - p_lo, 1e-5, 1.0), p_hi, p_lo])
    pd_spk = pd_spk / pd_spk.sum(axis=1, keepdims=True)

    feat_2d = np.column_stack([
        raw_grid / target_scale,
        np.log(np.clip(pd_dir, 1e-5, 1.0)),
        np.log(np.clip(pd_spk, 1e-5, 1.0)),
    ]).astype(np.float32)
    n_orig = len(df_sorted) // N_STEPS
    return feat_2d.reshape(n_orig, N_STEPS, 25)


def build_walkforward_lgbm_oof(
    panel_df: pd.DataFrame,
    features: list[str],
    train_origins: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    n_blocks: int = 5,
    rounds: int = 200,
) -> tuple[pd.DatetimeIndex, pd.DataFrame, M.Forecaster]:
    """Generate strictly forward-in-time out-of-sample LightGBM raw predictions across blocks 1..K-1,
    plus fit the full LightGBM model at ``cutoff``.
    """
    sorted_orig = train_origins.sort_values()
    blocks = np.array_split(sorted_orig, n_blocks)
    oof_parts = []
    oof_origins_list = []

    for k in range(1, n_blocks):
        blk_orig = pd.DatetimeIndex(blocks[k])
        if len(blk_orig) == 0:
            continue
        blk_cut = blk_orig[0]
        tr_k = M.training_rows(panel_df, blk_cut)
        tr_k = tr_k.loc[tr_k["origin"] >= sorted_orig[0]]
        if len(tr_k) < 2000:
            continue
        fc_k = M.Forecaster(features, "grid", rounds, None, spike_source="cdf").fit(tr_k, blk_cut)
        blk_df = panel_df.loc[panel_df["origin"].isin(blk_orig)].sort_values(["origin", "step"]).copy()

        # Store raw unscaled spread grid + probs in a compact table keyed by (origin, step)
        X = blk_df[fc_k.features]
        Xp = blk_df[fc_k.prob_features]
        raw_grid = np.sort(
            np.column_stack([fc_k.q[a].predict(X if a == 0.5 else Xp) for a in M.QGRID]),
            axis=1,
        )
        pd_dir = fc_k.direction.predict(Xp)
        pd_spk = fc_k.spike.predict(Xp)
        for j in range(N_QUANTILES):
            blk_df[f"_lgb_q{j}"] = raw_grid[:, j]
        for j in range(3):
            blk_df[f"_lgb_d{j}"] = pd_dir[:, j]
            blk_df[f"_lgb_s{j}"] = pd_spk[:, j]
        oof_parts.append(blk_df[["origin", "step"]
                                + [f"_lgb_q{j}" for j in range(N_QUANTILES)]
                                + [f"_lgb_d{j}" for j in range(3)]
                                + [f"_lgb_s{j}" for j in range(3)]])
        oof_origins_list.append(blk_orig)

    oof_table = pd.concat(oof_parts, ignore_index=True)
    oof_origins = oof_origins_list[0]
    for o in oof_origins_list[1:]:
        oof_origins = oof_origins.append(o)

    full_tr = M.training_rows(panel_df, cutoff)
    full_tr = full_tr.loc[full_tr["origin"] >= sorted_orig[0]]
    full_fc = M.Forecaster(features, "grid", rounds, None, spike_source="cdf").fit(full_tr, cutoff)
    return oof_origins.sort_values(), oof_table, full_fc


def _pack_oof_extra(oof_table: pd.DataFrame, origins: pd.DatetimeIndex, target_scale: float) -> np.ndarray:
    sub = oof_table.loc[oof_table["origin"].isin(origins)].sort_values(["origin", "step"])
    raw_grid = sub[[f"_lgb_q{j}" for j in range(N_QUANTILES)]].to_numpy(dtype=np.float64)
    pd_dir = np.clip(sub[[f"_lgb_d{j}" for j in range(3)]].to_numpy(dtype=np.float64), 1e-5, 1.0)
    pd_dir = pd_dir / pd_dir.sum(axis=1, keepdims=True)

    cdf_hi = np.array([np.interp(P.SPIKE_MARGIN, g, M.QGRID) for g in raw_grid])
    cdf_lo = np.array([np.interp(-P.SPIKE_MARGIN, g, M.QGRID) for g in raw_grid])
    p_hi = np.where(raw_grid[:, -1] < P.SPIKE_MARGIN, 0.025, 1.0 - cdf_hi)
    p_lo = np.where(raw_grid[:, 0] > -P.SPIKE_MARGIN, 0.025, cdf_lo)
    pd_spk = np.column_stack([np.clip(1.0 - p_hi - p_lo, 1e-5, 1.0), p_hi, p_lo])
    pd_spk = pd_spk / pd_spk.sum(axis=1, keepdims=True)

    feat_2d = np.column_stack([
        raw_grid / target_scale,
        np.log(np.clip(pd_dir, 1e-5, 1.0)),
        np.log(np.clip(pd_spk, 1e-5, 1.0)),
    ]).astype(np.float32)
    return feat_2d.reshape(len(origins), N_STEPS, 25)


@dataclass
class FittedDLSuite:
    arch: str
    models: list[nn.Module]
    scaler: ControlledScaler
    scale: dict[tuple[int, int], tuple[float, float]]
    best_epoch: int
    spike_source: str
    device: torch.device
    lgbm_fc: M.Forecaster | None = None

    def predict_df(self, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Predict on any ``eval_df`` containing complete 40-step origins; preserves input row order."""
        sorted_df = eval_df.assign(_pos=np.arange(len(eval_df))).sort_values(["origin", "step"]).reset_index(drop=True)
        orig_keep = pd.DatetimeIndex(sorted_df["origin"].unique()).sort_values()
        batch = build_masked_trajectories(sorted_df, orig_keep, self.scaler, cutoff=None, require_labels=False)
        if self.arch == "lgbm_tide_corrector":
            assert self.lgbm_fc is not None
            batch.extra_step = extract_lgbm_extra_features(self.lgbm_fc, sorted_df, self.scaler.target_scale)

        q_spread, d_prob, s_prob, p50_std = predict_models_raw(self.models, self.scaler, batch, self.device)
        pred_sorted, price_grid_sorted = assemble_prediction_df(
            sorted_df, q_spread, d_prob, s_prob, self.scale, spike_source=self.spike_source
        )
        order = np.argsort(sorted_df["_pos"].to_numpy())
        pred_final = pred_sorted.iloc[order].copy()
        pred_final.index = eval_df.index
        return M.finalize(pred_final), price_grid_sorted[order], p50_std.reshape(-1)[order]

    def feature_saliency(self, eval_df: pd.DataFrame) -> np.ndarray:
        """Compute per-row ``|Input x Gradient|`` attribution on ``p50`` across ``self.scaler.features``."""
        sorted_df = eval_df.assign(_pos=np.arange(len(eval_df))).sort_values(["origin", "step"]).reset_index(drop=True)
        orig_keep = pd.DatetimeIndex(sorted_df["origin"].unique()).sort_values()
        batch = build_masked_trajectories(sorted_df, orig_keep, self.scaler, cutoff=None, require_labels=False)
        n_orig = len(orig_keep)
        n_feat = len(self.scaler.features)
        state_sl = self.scaler.channel_slices(self.scaler.state_cols)
        step_sl = self.scaler.channel_slices(self.scaler.step_cols)
        feat_idx = {f: i for i, f in enumerate(self.scaler.features)}

        sal = np.zeros((n_orig, N_STEPS, n_feat), dtype=np.float64)
        ex = torch.from_numpy(batch.extra_step).to(self.device) if batch.extra_step is not None else None
        # Baseline at training median per feature (avoids mutating structural df["step"] = 1..40)
        state_base_dict = {
            c: np.full((n_orig,), self.scaler.medians[c], dtype=np.float32)
            for c in self.scaler.state_cols
        }
        step_base_dict = {
            c: np.full((n_orig, N_STEPS), self.scaler.medians[c], dtype=np.float32)
            for c in self.scaler.step_cols
        }
        xs_base_np = self.scaler._encode_block(state_base_dict, self.scaler.state_cols)
        xh_base_np = self.scaler._encode_block(step_base_dict, self.scaler.step_cols)
        xs_base = torch.from_numpy(xs_base_np).to(self.device)
        xh_base = torch.from_numpy(xh_base_np).to(self.device)

        with torch.backends.cudnn.flags(enabled=False):
            for m in self.models:
                m.eval()
                xs = torch.from_numpy(batch.X_state).to(self.device).requires_grad_(True)
                xh = torch.from_numpy(batch.X_step).to(self.device).requires_grad_(True)
                qs, _, _ = m(xs, xh, ex)
                p50 = qs[..., MEDIAN_IDX]  # (N, 40)
                for s in range(N_STEPS):
                    gs, gh = torch.autograd.grad(
                        p50[:, s].sum(), (xs, xh), retain_graph=(s < N_STEPS - 1)
                    )
                    ds_attr = (gs * (xs.detach() - xs_base)).abs().cpu().numpy()       # (N, Ds)
                    dh_attr = (gh[:, s, :] * (xh[:, s, :].detach() - xh_base[:, s, :])).abs().cpu().numpy()
                    for c, sl in state_sl.items():
                        sal[:, s, feat_idx[c]] += ds_attr[:, sl].sum(axis=-1)
                    for c, sl in step_sl.items():
                        sal[:, s, feat_idx[c]] += dh_attr[:, sl].sum(axis=-1)

        sal /= max(len(self.models), 1)
        flat_sal = sal.reshape(-1, n_feat)
        order = np.argsort(sorted_df["_pos"].to_numpy())
        return flat_sal[order]


def fit_dl_architecture(
    arch: str,
    panel_df: pd.DataFrame,
    features: list[str],
    cutoff: pd.Timestamp,
    post_june_only: bool,
    calibrate: bool,
    seeds: tuple[int, ...] = (101, 202, 303),
    max_epochs: int = 38,
    lr: float = 1.5e-3,
    weight_decay: float = 1.5e-3,
    spike_source: str = "mean",
    oof_cache: tuple[pd.DatetimeIndex, pd.DataFrame, M.Forecaster] | None = None,
    device: torch.device | None = None,
    fixed_best_epoch: int | None = None,
) -> FittedDLSuite:
    """Leakage-free two-pass fit (inner temporal early-stop + calibration -> full refit up to cutoff)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_orig = P.FIRST_ORIGIN if post_june_only else pd.Timestamp(panel_df["origin"].min())
    cand_df = panel_df.loc[(panel_df["origin"] >= min_orig) & (panel_df["origin"] < cutoff)]
    all_train_origins = pd.DatetimeIndex(cand_df["origin"].unique()).sort_values()

    lgbm_fc = None
    oof_table = None
    if arch == "lgbm_tide_corrector":
        if oof_cache is None:
            oof_origins, oof_table, lgbm_fc = build_walkforward_lgbm_oof(
                panel_df, features, all_train_origins, cutoff
            )
        else:
            oof_origins, oof_table, lgbm_fc = oof_cache
        usable_origins = oof_origins
        lr = 8e-4
        weight_decay = 3e-3
        max_epochs = 28
    else:
        usable_origins = all_train_origins

    if fixed_best_epoch is not None and not calibrate:
        best_epoch = int(fixed_best_epoch)
        scale = {b: (1.0, 1.0) for b in M.BANDS}
    else:
        span_days = (cutoff - usable_origins[0]).total_seconds() / 86400.0
        hold_days = 14.0 if span_days >= 60.0 else max(3.5, round(span_days * 0.20, 1))
        inner_cut = cutoff - pd.Timedelta(days=hold_days)

        inner_orig = usable_origins[usable_origins < inner_cut]
        hold_orig = usable_origins[usable_origins >= inner_cut]

        inner_tr_rows = M.training_rows(panel_df.loc[panel_df["origin"].isin(inner_orig)], inner_cut)
        inner_scaler = ControlledScaler.fit(inner_tr_rows, features)

        b_inner = build_masked_trajectories(panel_df, inner_orig, inner_scaler, cutoff=inner_cut, require_labels=True)
        b_hold = build_masked_trajectories(panel_df, hold_orig, inner_scaler, cutoff=cutoff, require_labels=True)
        if arch == "lgbm_tide_corrector":
            assert oof_table is not None
            b_inner.extra_step = _pack_oof_extra(oof_table, inner_orig, inner_scaler.target_scale)
            b_hold.extra_step = _pack_oof_extra(oof_table, hold_orig, inner_scaler.target_scale)

        tr_gpu_1 = _to_gpu(b_inner, inner_scaler.target_scale, device)
        va_gpu_1 = None if fixed_best_epoch is not None else _to_gpu(b_hold, inner_scaler.target_scale, device)

        pass1_models = []
        curve_list = []
        for sd in seeds:
            m_sd, vl = _train_single_seed(
                arch, inner_scaler, tr_gpu_1, va_gpu_1, sd, max_epochs, lr, weight_decay, device,
                fixed_epochs=int(fixed_best_epoch) if fixed_best_epoch is not None else None,
            )
            pass1_models.append(m_sd)
            curve_list.append(vl)

        if fixed_best_epoch is not None:
            best_epoch = int(fixed_best_epoch)
        else:
            mean_curve = np.mean(curve_list, axis=0)
            min_ep = min(6, max_epochs)
            best_epoch = int(np.argmin(mean_curve[min_ep - 1 :]) + min_ep)

        if calibrate:
            hold_df_sorted = panel_df.loc[panel_df["origin"].isin(hold_orig)].sort_values(["origin", "step"])
            q_hold, _, _, _ = predict_models_raw(pass1_models, inner_scaler, b_hold, device)
            scale = fit_scale_from_holdout(hold_df_sorted, q_hold, b_hold.mask)
        else:
            scale = {b: (1.0, 1.0) for b in M.BANDS}

    # Pass 2: Refit on 100% of usable training rows up to cutoff for best_epoch
    full_tr_rows = M.training_rows(panel_df.loc[panel_df["origin"].isin(usable_origins)], cutoff)
    full_scaler = ControlledScaler.fit(full_tr_rows, features)
    b_full = build_masked_trajectories(panel_df, usable_origins, full_scaler, cutoff=cutoff, require_labels=True)
    if arch == "lgbm_tide_corrector":
        assert oof_table is not None
        b_full.extra_step = _pack_oof_extra(oof_table, usable_origins, full_scaler.target_scale)

    tr_gpu_2 = _to_gpu(b_full, full_scaler.target_scale, device)
    final_models = [
        _train_single_seed(
            arch, full_scaler, tr_gpu_2, None, sd + 1000, max_epochs, lr, weight_decay, device,
            fixed_epochs=best_epoch,
        )[0]
        for sd in seeds
    ]

    return FittedDLSuite(
        arch=arch,
        models=final_models,
        scaler=full_scaler,
        scale=scale,
        best_epoch=best_epoch,
        spike_source=spike_source,
        device=device,
        lgbm_fc=lgbm_fc,
    )
