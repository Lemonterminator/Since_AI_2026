"""Torch-free 50/50 blend of two validated test submissions (Blend B).

Moved out of ``dl/make_test_submission.py`` so ``reproduce_submission.py`` can rebuild the
submitted file from a LightGBM CSV and a DL CSV without importing PyTorch. The arithmetic is
unchanged: plain average of p10/p50/p90 and all probabilities, then ``make_submission.finalize``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import make_submission as M

PRED_COLS = ["p10", "p50", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"]


def merge_top_drivers(drivers_a: pd.Series, drivers_b: pd.Series, top: int = 3) -> list[str]:
    """Rank-combine semicolon-separated top-3 feature driver strings from LightGBM and DL."""
    out: list[str] = []
    for sa, sb in zip(drivers_a.astype(str), drivers_b.astype(str)):
        scores: dict[str, float] = {}
        for rank, f in enumerate(sa.split(";")):
            if f:
                scores[f] = scores.get(f, 0.0) + (3.0 - rank)
        for rank, f in enumerate(sb.split(";")):
            if f:
                scores[f] = scores.get(f, 0.0) + (3.0 - rank) * 0.99  # tie-break toward LightGBM SHAP
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
        out.append(";".join(k for k, _ in ranked))
    return out


def blend_submission_csvs_50_50(
    lgbm_csv: Path,
    dl_csv: Path,
    out_csv: Path,
    origins_csv: Path,
) -> pd.DataFrame:
    """Exact 50/50 average of quantiles and probabilities between two validated test CSVs."""
    a = pd.read_csv(lgbm_csv)
    b = pd.read_csv(dl_csv)
    keys = ["forecast_origin_utc", "delivery_start_utc", "step"]
    assert len(a) == len(b), f"Row count mismatch: {len(a)} vs {len(b)}"
    assert (a[keys].reset_index(drop=True) == b[keys].reset_index(drop=True)).all().all(), (
        "Key ordering mismatch between LightGBM and DL test files"
    )

    raw_avg = pd.DataFrame(index=a.index)
    for c in PRED_COLS:
        raw_avg[c] = 0.5 * a[c].to_numpy(dtype=np.float64) + 0.5 * b[c].to_numpy(dtype=np.float64)

    # Verify mathematically zero quantile crossing BEFORE M.finalize()
    n_cross_lo = int((raw_avg["p10"] > raw_avg["p50"] + 1e-12).sum())
    n_cross_hi = int((raw_avg["p50"] > raw_avg["p90"] + 1e-12).sum())
    assert n_cross_lo == 0 and n_cross_hi == 0, f"Unexpected quantile crossing: {n_cross_lo}, {n_cross_hi}"

    fin = M.finalize(raw_avg)
    merged = a[keys].copy()
    for c in PRED_COLS:
        merged[c] = fin[c]
    merged["data_cutoff_utc"] = a["data_cutoff_utc"]
    merged["top_drivers"] = merge_top_drivers(a["top_drivers"], b["top_drivers"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[M.OUT_COLUMNS].to_csv(out_csv, index=False, lineterminator="\n")

    rc = M.run_official("validate_submission.py", "--backtest", str(out_csv), "--origins", str(origins_csv))
    if rc != 0:
        raise RuntimeError(f"validate_submission.py failed for {out_csv}")
    return merged
