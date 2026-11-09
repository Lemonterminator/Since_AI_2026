"""Strict feature-controlled tensor pipeline built directly on submission/panel.py.

Guarantees:
1. Zero extra features or external series: consumes ONLY the exact ``cols`` list
   used by ``submission/make_submission.py`` (18 features for ``post_june``, or
   19 features with ``regime_pre_june`` for ``hist_relabel_flag``).
2. Zero lookahead / leakage:
   - ``training_rows(panel, cutoff)`` determines every usable ``(origin, step)``
     label (``label_usable <= cutoff``, ``origin < cutoff``, ``delivery < TEST_START``).
   - Unusable ``(origin, step)`` cells have ``mask = False`` and their labels are
     erased (``0.0``) before entering any tensor.
   - Median/IQR scalers and empirical quantile priors are fitted strictly on
     masked training cells.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import history as H  # noqa: E402
import panel as P  # noqa: E402
from make_submission import DIRECTIONS, QGRID, SPIKES, training_rows  # noqa: E402

N_STEPS = 40

# Exact decomposition of ``core_both_plan_stable`` (+ optional ``regime_pre_june``)
# into origin-constant state features vs step-varying known horizon features.
ORIGIN_STATE_CANDIDATES = (
    "last_spread",
    "last_direction",
    "up_share_8q",
    "spread_mean_8q",
    "spread_std_8q",
    "spread_change_1q",
    "last_mfrr_up",
    "last_mfrr_down",
    "wind_err_last",
)


def load_or_build_panels(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Load or build the exact raw and relabeled panels through `` LAST_TEST_ORIGIN ``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "panel_raw_full.parquet"
    rel_path = cache_dir / "panel_relabel_full.parquet"
    if raw_path.exists() and rel_path.exists():
        raw = pd.read_parquet(raw_path)
        rel = pd.read_parquet(rel_path)
        for df in (raw, rel):
            for c in ("origin", "delivery", "label_usable"):
                df[c] = P.to_utc(df[c])
        return {"raw": raw, "relabel": rel}

    base_rows = P.load_rows(external=True)
    nuclear = P.load_nuclear()
    raw = H.build_history_panel(False, base_rows, nuclear, end=P.LAST_TEST_ORIGIN)
    rel = H.build_history_panel(True, base_rows, nuclear, end=P.LAST_TEST_ORIGIN)
    raw.to_parquet(raw_path, index=False)
    rel.to_parquet(rel_path, index=False)
    return {"raw": raw, "relabel": rel}


@dataclass
class ControlledScaler:
    """Leakage-free dual-scale preprocessor fitted strictly on masked training cells."""

    features: list[str]
    state_cols: list[str]
    step_cols: list[str]
    medians: dict[str, float]
    scales: dict[str, float]
    has_na: dict[str, bool]
    target_scale: float
    target_q_init: np.ndarray       # shape (19,) in scaled spread units
    dir_log_prior: np.ndarray       # shape (3,)
    spike_log_prior: np.ndarray     # shape (3,)

    @classmethod
    def fit(cls, train_df: pd.DataFrame, features: list[str]) -> "ControlledScaler":
        state_cols = [c for c in features if c in ORIGIN_STATE_CANDIDATES]
        step_cols = [c for c in features if c not in state_cols]
        assert set(state_cols) | set(step_cols) == set(features)

        medians, scales, has_na = {}, {}, {}
        for c in features:
            vals = train_df[c].to_numpy(dtype=np.float64)
            na = np.isnan(vals)
            has_na[c] = bool(na.any())
            med = float(np.nanmedian(vals)) if (~na).any() else 0.0
            q75, q25 = np.nanpercentile(vals, [75, 25]) if (~na).any() else (1.0, 0.0)
            iqr = float(q75 - q25)
            if iqr < 1e-4:
                std = float(np.nanstd(vals)) if (~na).any() else 1.0
                scale = std if std > 1e-4 else 1.0
            else:
                scale = iqr
            medians[c] = med
            scales[c] = scale

        spread = train_df["spread"].to_numpy(dtype=np.float64)
        q75_y, q25_y = np.percentile(spread, [75, 25])
        target_scale = max(float(q75_y - q25_y), 10.0)
        q_emp = np.percentile(spread, QGRID * 100.0) / target_scale

        dir_codes = pd.Categorical(train_df["direction"], DIRECTIONS).codes
        dir_counts = np.bincount(dir_codes[dir_codes >= 0], minlength=3).astype(np.float64) + 1.0
        dir_probs = dir_counts / dir_counts.sum()

        spk_codes = pd.Categorical(train_df["spike"], SPIKES).codes
        spk_counts = np.bincount(spk_codes[spk_codes >= 0], minlength=3).astype(np.float64) + 1.0
        spk_probs = spk_counts / spk_counts.sum()

        return cls(
            features=list(features),
            state_cols=state_cols,
            step_cols=step_cols,
            medians=medians,
            scales=scales,
            has_na=has_na,
            target_scale=target_scale,
            target_q_init=q_emp.astype(np.float32),
            dir_log_prior=np.log(dir_probs).astype(np.float32),
            spike_log_prior=np.log(spk_probs).astype(np.float32),
        )

    def _encode_block(self, df_vals: dict[str, np.ndarray], cols: list[str]) -> np.ndarray:
        """Deterministic bounded linear + heavy-tail asinh + cyclical basis of exact cols."""
        chans = []
        for c in cols:
            raw = df_vals[c].astype(np.float32, copy=False)
            na = np.isnan(raw)
            filled = np.where(na, np.float32(self.medians[c]), raw)
            z = (filled - np.float32(self.medians[c])) / np.float32(self.scales[c])
            # Linear bounded view + smooth invertible heavy-tail view
            chans.append(np.clip(z, -6.0, 6.0) / 3.0)
            chans.append(np.arcsinh(z) / 3.0)
            if c in ("wind_err_last", "spread_change_1q", "spot_ramp_1h", "net_load_ramp_1h",
                     "delivery_hour", "delivery_weekday") or self.has_na[c]:
                chans.append(na.astype(np.float32))
            if c == "delivery_hour":
                rad = 2.0 * np.pi * filled / 24.0
                chans.append(np.sin(rad).astype(np.float32))
                chans.append(np.cos(rad).astype(np.float32))
            elif c == "delivery_weekday":
                rad = 2.0 * np.pi * filled / 7.0
                chans.append(np.sin(rad).astype(np.float32))
                chans.append(np.cos(rad).astype(np.float32))
            elif c == "step":
                rad = 2.0 * np.pi * (filled - 1.0) / 40.0
                chans.append(np.sin(rad).astype(np.float32))
                chans.append(np.cos(rad).astype(np.float32))
        return np.stack(chans, axis=-1).astype(np.float32, copy=False)

    def channel_slices(self, cols: list[str]) -> dict[str, slice]:
        """Return exact channel slice for each raw feature name in ``_encode_block``."""
        out: dict[str, slice] = {}
        pos = 0
        for c in cols:
            width = 2
            if c in ("wind_err_last", "spread_change_1q", "spot_ramp_1h", "net_load_ramp_1h",
                     "delivery_hour", "delivery_weekday") or self.has_na[c]:
                width += 1
            if c in ("delivery_hour", "delivery_weekday", "step"):
                width += 2
            out[c] = slice(pos, pos + width)
            pos += width
        return out

    def transform_trajectory(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert sorted ``(origin, step=1..40)`` panel slice to ``(N, Ds)`` and ``(N, 40, Dh)``.

        Requires every origin in ``df`` to have all 40 steps ordered ``1..40``.
        Returns ``(origins, X_state [N, Ds], X_step [N, 40, Dh])``.
        """
        n_rows = len(df)
        assert n_rows % N_STEPS == 0, f"Expected multiple of 40 rows, got {n_rows}"
        n_orig = n_rows // N_STEPS
        steps = df["step"].to_numpy().reshape(n_orig, N_STEPS)
        assert np.all(steps == np.arange(1, N_STEPS + 1)[None, :]), "Steps must be ordered 1..40 per origin"

        origins = df["origin"].to_numpy().reshape(n_orig, N_STEPS)[:, 0]
        state_dict = {
            c: df[c].to_numpy().reshape(n_orig, N_STEPS)[:, 0]
            for c in self.state_cols
        }
        step_dict = {
            c: df[c].to_numpy().reshape(n_orig, N_STEPS)
            for c in self.step_cols
        }
        X_state = self._encode_block(state_dict, self.state_cols)
        X_step = self._encode_block(step_dict, self.step_cols)
        return origins, X_state, X_step


@dataclass
class TrajectoryBatch:
    origins: pd.DatetimeIndex
    X_state: np.ndarray          # (N, Ds)
    X_step: np.ndarray           # (N, 40, Dh)
    spot: np.ndarray             # (N, 40)
    y_spread: np.ndarray         # (N, 40), 0.0 where masked
    y_dir: np.ndarray            # (N, 40) int64, -1 where masked
    y_spike: np.ndarray          # (N, 40) int64, -1 where masked
    mask: np.ndarray             # (N, 40) bool: True iff label published <= cutoff
    extra_step: np.ndarray | None = None  # optional OOF/LGBM trajectory features for Arch 5


def build_masked_trajectories(
    panel_df: pd.DataFrame,
    origins_keep: pd.DatetimeIndex,
    scaler: ControlledScaler,
    cutoff: pd.Timestamp | None = None,
    require_labels: bool = True,
) -> TrajectoryBatch:
    """Build complete 40-step trajectories per origin with strict per-step publication masking."""
    sub = panel_df.loc[panel_df["origin"].isin(origins_keep)].sort_values(["origin", "step"]).reset_index(drop=True)
    origins_arr, X_state, X_step = scaler.transform_trajectory(sub)
    n_orig = len(origins_arr)
    spot = sub["spot_level"].to_numpy(dtype=np.float32).reshape(n_orig, N_STEPS)

    if not require_labels:
        zeros_f = np.zeros((n_orig, N_STEPS), dtype=np.float32)
        neg_i = np.full((n_orig, N_STEPS), -1, dtype=np.int64)
        mask = np.zeros((n_orig, N_STEPS), dtype=bool)
        return TrajectoryBatch(pd.DatetimeIndex(origins_arr), X_state, X_step, spot, zeros_f, neg_i, neg_i, mask)

    assert cutoff is not None, "Training trajectories require an explicit publication cutoff"
    ok_flat = (
        sub["label_usable"].notna()
        & (pd.DatetimeIndex(sub["label_usable"]) <= cutoff)
        & (sub["delivery"] < P.TEST_START)
        & (sub["origin"] < cutoff)
        & sub["spread"].notna()
        & sub["direction"].notna()
        & sub["spot_level"].notna()
    ).to_numpy()

    # Double-check zero leakage on every unmasked entry
    if ok_flat.any():
        assert (pd.DatetimeIndex(sub.loc[ok_flat, "label_usable"]) <= cutoff).all()
        assert (sub.loc[ok_flat, "delivery"] < P.TEST_START).all()
        assert (sub.loc[ok_flat, "origin"] < cutoff).all()

    mask = ok_flat.reshape(n_orig, N_STEPS)
    raw_spread = sub["spread"].to_numpy(dtype=np.float32).reshape(n_orig, N_STEPS)
    y_spread = np.where(mask, raw_spread, 0.0).astype(np.float32)

    dir_codes = pd.Categorical(sub["direction"], DIRECTIONS).codes.astype(np.int64).reshape(n_orig, N_STEPS)
    spk_codes = pd.Categorical(sub["spike"], SPIKES).codes.astype(np.int64).reshape(n_orig, N_STEPS)
    y_dir = np.where(mask, dir_codes, -1)
    y_spike = np.where(mask, spk_codes, -1)

    return TrajectoryBatch(pd.DatetimeIndex(origins_arr), X_state, X_step, spot, y_spread, y_dir, y_spike, mask)
