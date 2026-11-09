"""Build the official forecast_backtest.csv (or a validation-period equivalent).

    # validation period (train on labels published by 2026-09-06T21Z, score with score.py)
    python research/imbalance-forecast/submission/make_submission.py --mode validation
    # test period, the file to submit (train on labels published by 2026-10-04T21Z)
    python research/imbalance-forecast/submission/make_submission.py --mode test

Model (LightGBM, all targets are relative to the known delivery spot price):
  * direction: 3-class classifier -> p_up, p_down, p_none
  * spikes:    3-class classifier (normal/high/low) -> p_spike_high, p_spike_low
  * price, ``--model direct``: quantile regression of (imbalance - spot) at 0.1/0.5/0.9
  * price, ``--model two_stage``: conditional quantile grids of the spread for up- and
    down-regulated quarters, mixed with the direction probabilities (none -> spread 0);
    P10/P50/P90 are read from the mixture CDF.
  * ``--model blend`` averages the direct and two-stage quantiles.
Interval widths are rescaled per horizon band with factors fitted on an inner
time split of the training window (``--calibrate``).

Leakage rules (also checked by leakage_check.py):
  * features: panel.MarketRows.at only returns rows with first_usable_origin_utc <= origin;
  * training rows: every label row (price, direction, spot) must have
    first_usable_origin_utc <= training cut-off, and delivery < 2026-10-04T21Z
    (official training end); test-period labels are never read for training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402
import history as H  # noqa: E402

DIRECTIONS = ("up", "down", "none")
SPIKES = ("normal", "high", "low")
GRID = np.array([.02, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .98])
BANDS = ((1, 8), (9, 20), (21, 40))
QGRID = np.round(np.arange(1, 20) * .05, 2)  # 0.05 ... 0.95, used by --model grid
OUT_COLUMNS = ["forecast_origin_utc", "delivery_start_utc", "step", "p10", "p50", "p90", "p_up", "p_down",
               "p_none", "p_spike_high", "p_spike_low", "data_cutoff_utc", "top_drivers"]
TS = "%Y-%m-%dT%H:%M:%SZ"

PARAMS = {
    "max_depth": 5, "num_leaves": 24, "min_data_in_leaf": 200, "learning_rate": .04,
    "lambda_l2": 5, "feature_fraction": .8, "bagging_fraction": .8, "bagging_freq": 1,
    "verbosity": -1, "seed": 20261107, "num_threads": 4, "deterministic": True, "force_col_wise": True,
}
# Concurrency: ``jobs`` models trained at once, each with ``threads_per_model`` threads.
# Default: all cores, 1 thread per model (1 thread ~7 s vs 4 threads ~2.9 s for one 500k-row
# quantile model, so 4 single-thread models in parallel are ~1.6x faster than 4 threads each).
# Override with env TT_JOBS / TT_THREADS_PER_MODEL or set_parallel(). Never run two scripts
# at once on the same cores: oversubscribed OpenMP threads made one fit ~25x slower.
PARALLEL = {"jobs": int(os.environ.get("TT_JOBS", os.cpu_count() or 1)),
            "threads_per_model": int(os.environ.get("TT_THREADS_PER_MODEL", 1))}


def set_parallel(jobs: int | None = None, threads_per_model: int | None = None) -> None:
    if jobs:
        PARALLEL["jobs"] = jobs
    if threads_per_model:
        PARALLEL["threads_per_model"] = threads_per_model


# --------------------------------------------------------------------------
# training data
# --------------------------------------------------------------------------

def training_rows(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows whose labels were all published by ``cutoff`` and lie inside the official training window."""
    ok = (
        panel.label_usable.notna()
        & (pd.DatetimeIndex(panel.label_usable) <= cutoff)
        & (panel.delivery < P.TEST_START)
        & (panel.origin < cutoff)
        & panel.spread.notna() & panel.direction.notna() & panel.spot_level.notna()
    )
    train = panel.loc[ok]
    assert (pd.DatetimeIndex(train.label_usable) <= cutoff).all()
    assert (train.delivery < P.TEST_START).all()
    return train


def recency_weight(train: pd.DataFrame, cutoff: pd.Timestamp, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    age = (cutoff - train.origin).dt.total_seconds().to_numpy() / 86400
    return np.power(0.5, age / half_life_days)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class Forecaster:
    def __init__(self, features: list[str], model: str, rounds: int, half_life: float | None, params=PARAMS,
                 prob_features: list[str] | None = None, spike_source: str = "classifier"):
        # ``features`` drive the P50 model; ``prob_features`` drive direction, spikes and P10/P90.
        self.features, self.model, self.rounds, self.half_life = features, model, rounds, half_life
        self.prob_features = prob_features or features
        self.spike_source = spike_source
        self.last_grid = None
        self.params = dict(params)

    def _train(self, X, y, w, objective, rounds=None, **kw):
        """Submit one LightGBM fit to the pool; returns a future (resolved in ``fit``)."""
        ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
        params = dict(self.params, objective=objective, num_threads=PARALLEL["threads_per_model"], **kw)
        return self._pool.submit(lgb.train, params, ds, num_boost_round=rounds or self.rounds)

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> "Forecaster":
        # The 2-23 models are independent: train them concurrently (LightGBM releases the GIL).
        # Few features + many rows -> one model scales poorly beyond ~2 threads, so several
        # models with fewer threads each keep all cores busy. See PARALLEL / --jobs.
        X, Xp = train[self.features], train[self.prob_features]
        w = recency_weight(train, cutoff, self.half_life)
        with ThreadPoolExecutor(max_workers=PARALLEL["jobs"]) as self._pool:
            direction = self._train(Xp, pd.Categorical(train.direction, DIRECTIONS).codes, w,
                                    "multiclass", num_class=3)
            spike = self._train(Xp, pd.Categorical(train.spike, SPIKES).codes, w, "multiclass", num_class=3)
            q = {}
            if self.model in ("direct", "blend"):
                for a in (.1, .5, .9):
                    q[a] = self._train(X if a == .5 else Xp, train.spread, w, "quantile", alpha=a)
            if self.model == "grid":
                for a in QGRID:
                    q[a] = self._train(X if a == .5 else Xp, train.spread, w, "quantile", alpha=a)
            cond = {}
            if self.model in ("two_stage", "blend"):
                for regime in ("up", "down"):
                    sel = (train.direction == regime).to_numpy()
                    wr = None if w is None else w[sel]
                    cond[regime] = [self._train(X.loc[sel], train.spread[sel], wr, "quantile", alpha=a)
                                    for a in GRID]
            self.direction, self.spike = direction.result(), spike.result()
            self.q = {a: f.result() for a, f in q.items()}
            if cond:
                self.cond = {r: [f.result() for f in fs] for r, fs in cond.items()}
        del self._pool
        self.scale = {b: (1.0, 1.0) for b in BANDS}
        return self

    def _mixture(self, X: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
        n = len(X)
        qs = {}
        for regime in ("up", "down"):
            grid = np.column_stack([m.predict(X) for m in self.cond[regime]])
            qs[regime] = np.sort(grid, axis=1)  # enforce monotone quantiles
        lo = np.minimum(qs["down"][:, 0], 0) - 1
        hi = np.maximum(qs["up"][:, -1], 0) + 1
        xs = np.linspace(0, 1, 801)[None, :] * (hi - lo)[:, None] + lo[:, None]
        cdf = np.zeros_like(xs)
        for k, regime in enumerate(("up", "down")):
            q = qs[regime]
            # piecewise-linear CDF through (q_j, GRID_j), flat tails at GRID ends extended to 0/1
            ext_q = np.column_stack([q[:, :1] - (q[:, 1:2] - q[:, :1]).clip(1) * 2, q, q[:, -1:] + (q[:, -1:] - q[:, -2:-1]).clip(1) * 2])
            ext_p = np.concatenate([[0], GRID, [1]])
            for i in range(n):
                cdf[i] += probs[i, k] * np.interp(xs[i], ext_q[i], ext_p)
        cdf += probs[:, 2:3] * (xs >= 0)
        out = np.empty((n, 3))
        for j, a in enumerate((.1, .5, .9)):
            idx = (cdf >= a).argmax(axis=1)
            out[:, j] = xs[np.arange(n), idx]
        return out

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X, Xp = rows[self.features], rows[self.prob_features]
        pd_dir = self.direction.predict(Xp)
        pd_spk = self.spike.predict(Xp)
        parts = []
        grid = None
        if self.model == "grid":
            # monotone rearrangement (Chernozhukov et al.): sorting never increases pinball loss
            grid = np.sort(np.column_stack([self.q[a].predict(X if a == .5 else Xp) for a in QGRID]), axis=1)
            parts.append(grid[:, [1, 9, 17]])  # 0.10, 0.50, 0.90
        if self.model in ("direct", "blend"):
            parts.append(np.column_stack([self.q[a].predict(X if a == .5 else Xp) for a in (.1, .5, .9)]))
        if self.model in ("two_stage", "blend"):
            parts.append(self._mixture(X, pd_dir))
        q = np.mean(parts, axis=0)
        q10, q50, q90 = np.minimum(q[:, 0], q[:, 1]), q[:, 1], np.maximum(q[:, 2], q[:, 1])
        lo_w, hi_w = q50 - q10, q90 - q50
        sl_row, sh_row = np.ones(len(rows)), np.ones(len(rows))
        for (a, b), (sl, sh) in self.scale.items():
            sel = rows.step.between(a, b).to_numpy()
            sl_row[sel], sh_row[sel] = sl, sh
        lo_w, hi_w = lo_w * sl_row, hi_w * sh_row
        spot = rows.spot_level.to_numpy()
        if grid is not None:
            # same width scaling for every quantile: below the median by sl, above by sh
            dev = grid - q50[:, None]
            grid = q50[:, None] + np.where(dev < 0, dev * sl_row[:, None], dev * sh_row[:, None])
            self.last_grid = grid + spot[:, None]
            if self.spike_source in ("cdf", "mean"):
                cdf_hi = np.array([np.interp(P.SPIKE_MARGIN, g, QGRID) for g in grid])
                cdf_lo = np.array([np.interp(-P.SPIKE_MARGIN, g, QGRID) for g in grid])
                # beyond the 5%/95% grid ends the CDF is extrapolated flat (conservative)
                p_hi = np.where(grid[:, -1] < P.SPIKE_MARGIN, 0.025, 1 - cdf_hi)
                p_lo = np.where(grid[:, 0] > -P.SPIKE_MARGIN, 0.025, cdf_lo)
                if self.spike_source == "cdf":
                    pd_spk = np.column_stack([1 - p_hi - p_lo, p_hi, p_lo])
                else:
                    pd_spk = (pd_spk + np.column_stack([1 - p_hi - p_lo, p_hi, p_lo])) / 2
        out = pd.DataFrame({
            "p10": spot + q50 - lo_w, "p50": spot + q50, "p90": spot + q50 + hi_w,
            "p_up": pd_dir[:, 0], "p_down": pd_dir[:, 1], "p_none": pd_dir[:, 2],
            "p_spike_high": pd_spk[:, 1], "p_spike_low": pd_spk[:, 2],
        }, index=rows.index)
        return out

    def drivers(self, rows: pd.DataFrame, top: int = 3) -> list[str]:
        """Top |SHAP| features of the P50 spread model (or the direction model for two_stage)."""
        X = rows[self.features]
        if .5 in self.q:
            contrib = self.q[.5].predict(X, pred_contrib=True)[:, :-1]
        else:
            c = self.direction.predict(rows[self.prob_features], pred_contrib=True)
            contrib = c[:, : len(self.prob_features)]  # class "up" contributions
            names = np.array(self.prob_features)
            order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
            return [";".join(names[o]) for o in order]
        order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
        names = np.array(self.features)
        return [";".join(names[o]) for o in order]


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def fit_interval_scale(fc_factory, train: pd.DataFrame, cutoff: pd.Timestamp, holdout_days: int = 14) -> dict:
    """Fit P10/P90 width multipliers per horizon band on the last ``holdout_days`` of the training window."""
    inner_cut = cutoff - pd.Timedelta(days=holdout_days)
    inner = training_rows(train, inner_cut)
    hold = train.loc[train.origin >= inner_cut]
    fc = fc_factory().fit(inner, inner_cut)
    pred = fc.predict(hold)
    y = hold.imbalance_price.to_numpy()
    scale = {}
    grid = np.linspace(.6, 2.5, 39)
    for a, b in BANDS:
        sel = hold.step.between(a, b).to_numpy()
        p50 = pred.p50.to_numpy()[sel]
        lo = p50 - pred.p10.to_numpy()[sel]
        hi = pred.p90.to_numpy()[sel] - p50
        sl = grid[np.argmin([pinball(y[sel], p50 - s * lo, .1) for s in grid])]
        sh = grid[np.argmin([pinball(y[sel], p50 + s * hi, .9) for s in grid])]
        scale[(a, b)] = (float(sl), float(sh))
    return scale


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def finalize(pred: pd.DataFrame) -> pd.DataFrame:
    """Clip/renormalise so that the validator constraints hold exactly."""
    out = pred.copy()
    out["p50"] = out[["p10", "p50", "p90"]].median(axis=1).where(
        (out.p10 <= out.p50) & (out.p50 <= out.p90), out.p50)
    out["p10"] = np.minimum(out.p10, out.p50)
    out["p90"] = np.maximum(out.p90, out.p50)
    d = out[["p_up", "p_down", "p_none"]].clip(1e-4, 1)
    d = d.div(d.sum(axis=1), axis=0)
    out[["p_up", "p_down", "p_none"]] = d.round(6)
    out["p_none"] = (1 - out.p_up - out.p_down).clip(0, 1).round(6)
    s = out[["p_spike_high", "p_spike_low"]].clip(1e-4, 1)
    tot = s.sum(axis=1)
    s = s.div(np.maximum(tot, 1), axis=0) * np.where(tot > .999, .999, 1)[:, None]
    out[["p_spike_high", "p_spike_low"]] = s.round(6)
    for c in ("p10", "p50", "p90"):
        out[c] = out[c].round(3)
    return out


def write_submission(rows: pd.DataFrame, pred: pd.DataFrame, drivers: list[str], template: Path, path: Path):
    tmpl = pd.read_csv(template, dtype=str, keep_default_na=False)
    key = pd.DataFrame({
        "forecast_origin_utc": rows.origin.dt.strftime(TS).to_numpy(),
        "step": rows.step.astype(str).to_numpy(),
        "delivery_start_utc": rows.delivery.dt.strftime(TS).to_numpy(),
    })
    body = pd.concat([key, pred.reset_index(drop=True)], axis=1)
    body["data_cutoff_utc"] = key.forecast_origin_utc
    body["top_drivers"] = drivers
    merged = tmpl[["forecast_origin_utc", "delivery_start_utc", "step"]].merge(
        body, on=["forecast_origin_utc", "step"], how="left", suffixes=("", "_ours"), validate="one_to_one")
    if merged.p50.isna().any():
        raise ValueError(f"{int(merged.p50.isna().sum())} template rows have no forecast")
    if not (merged.delivery_start_utc == merged.delivery_start_utc_ours).all():
        raise ValueError("delivery_start_utc mismatch with template")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged[OUT_COLUMNS].to_csv(path, index=False, lineterminator="\n")


def ensure_validation_templates(out_dir: Path) -> Path:
    tdir = out_dir / "validation_templates"
    if not (tdir / "sample_submission.csv").exists():
        subprocess.run([sys.executable, str(P.STARTER / "make_templates.py"), "--test-start", "2026-09-07",
                        "--test-end", "2026-10-04", "--output-dir", str(tdir)], check=True, cwd=P.STARTER)
    return tdir


def run_official(script: str, *args: str) -> int:
    return subprocess.run([sys.executable, str(P.STARTER / script), *args]).returncode


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("validation", "test"), default="validation")
    ap.add_argument("--model", choices=("direct", "two_stage", "blend", "grid"), default="grid")
    ap.add_argument("--spike-source", choices=("classifier", "cdf", "mean"), default="cdf",
                    help="grid model only: spike probabilities from the classifier, the quantile CDF or their mean")
    ap.add_argument("--features", choices=sorted(P.VARIANTS), default="core_both_plan_stable",
                    help="feature set of the P50 model")
    ap.add_argument("--prob-features", choices=sorted(P.VARIANTS), default=None,
                    help="feature set of direction, spike and P10/P90 models (default: --features)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--half-life-days", type=float, default=None)
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-external", action="store_true", help="starter package only (disables plan/nuclear extras)")
    ap.add_argument("--history", choices=("none", "raw", "relabel"), default="none",
                    help="add gold_ext history from 2025-04 to the training set; 'relabel' re-prices the "
                         "pre-June imbalance prices with the post-June volume-weighted rule (history.py)")
    ap.add_argument("--regime-flag", action="store_true",
                    help="add feature regime_pre_june (1 for deliveries before 2026-06-01 local, else 0)")
    ap.add_argument("--out-dir", type=Path, default=P.HERE / "runs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rows = P.load_rows(external=not args.no_external)
    nuclear = None if args.no_external else P.load_nuclear()
    first_origin = P.FIRST_ORIGIN
    if args.history != "none":
        rows = H.combined_rows(rows, relabel=args.history == "relabel")
        first_origin = H.HISTORY_START + pd.Timedelta(days=2)
    origins = P.hourly_origins(first_origin, P.LAST_TEST_ORIGIN)
    panel = P.build_panel(P.MarketRows(rows), origins, nuclear)
    panel["regime_pre_june"] = (panel.delivery < H.SWITCH).astype(float)
    print(f"panel {panel.shape} built in {time.time() - t0:.0f}s", flush=True)

    tag = args.tag or f"{args.mode}_{args.model}_{args.features}" + (
        f"__spike_{args.spike_source}" if args.model == "grid" and args.spike_source != "cdf" else "") + (
        f"__prob_{args.prob_features}" if args.prob_features else "") + (
        f"__hist_{args.history}" if args.history != "none" else "") + ("__flag" if args.regime_flag else "")
    out_dir = args.out_dir / tag
    if args.mode == "validation":
        cutoff = P.VALIDATION_START
        tdir = ensure_validation_templates(args.out_dir)
        template, origins_csv = tdir / "sample_submission.csv", tdir / "forecast_origins.csv"
    else:
        cutoff = P.TEST_START
        template, origins_csv = P.STARTER / "sample_submission.csv", P.STARTER / "forecast_origins.csv"
    target_origins = P.to_utc(pd.read_csv(origins_csv).iloc[:, 0])
    predict_rows = panel.loc[panel.origin.isin(target_origins)].reset_index(drop=True)
    if predict_rows.spot_level.isna().any():
        raise ValueError("delivery spot price missing at some origin - cannot anchor forecast")

    train = training_rows(panel, cutoff)
    print(f"train rows {len(train)} (origins {train.origin.nunique()}, last label published "
          f"{pd.DatetimeIndex(train.label_usable).max()}), predict rows {len(predict_rows)}", flush=True)

    features = list(P.VARIANTS[args.features])
    prob_features = list(P.VARIANTS[args.prob_features]) if args.prob_features else None
    if args.regime_flag:
        features.append("regime_pre_june")
        if prob_features is not None:
            prob_features.append("regime_pre_june")
    factory = lambda: Forecaster(features, args.model, args.rounds, args.half_life_days,  # noqa: E731
                                 prob_features=prob_features, spike_source=args.spike_source)
    fc = factory().fit(train, cutoff)
    if args.calibrate:
        fc.scale = fit_interval_scale(factory, train, cutoff)
        print("interval scale per band:", fc.scale, flush=True)
    pred = finalize(fc.predict(predict_rows))
    drivers = fc.drivers(predict_rows)

    out_file = out_dir / "forecast_backtest.csv"
    write_submission(predict_rows, pred, drivers, template, out_file)
    if fc.last_grid is not None:
        grid = pd.DataFrame(fc.last_grid, columns=[f"q{int(round(a * 100)):02d}" for a in QGRID])
        grid.insert(0, "step", predict_rows.step.to_numpy())
        grid.insert(0, "delivery_start_utc", predict_rows.delivery.dt.strftime(TS).to_numpy())
        grid.insert(0, "forecast_origin_utc", predict_rows.origin.dt.strftime(TS).to_numpy())
        grid.to_parquet(out_dir / "quantile_grid.parquet", index=False)
    meta = {
        "mode": args.mode, "model": args.model, "features": features,
        "prob_features": prob_features or features, "rounds": args.rounds,
        "half_life_days": args.half_life_days, "spike_source": args.spike_source, "calibrate": args.calibrate,
        "interval_scale": {f"{a}-{b}": v for (a, b), v in fc.scale.items()},
        "training_cutoff_utc": cutoff.strftime(TS), "train_rows": int(len(train)),
        "max_label_first_usable_utc": pd.DatetimeIndex(train.label_usable).max().strftime(TS),
        "external": not args.no_external, "history": args.history, "regime_flag": args.regime_flag,
        "params": PARAMS,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_file}", flush=True)

    rc = run_official("validate_submission.py", "--backtest", str(out_file), "--origins", str(origins_csv))
    if rc != 0:
        sys.exit(rc)
    if args.mode == "validation":
        baseline = P.STARTER / "baseline_output" / "validation" / "score" / "score_summary.csv"
        extra = ["--baseline-summary", str(baseline)] if baseline.exists() else []
        run_official("score.py", "--submission", str(out_file), "--actuals",
                     str(P.STARTER / "data" / "actuals_validation.csv"), "--spike-margin", "50",
                     "--output-dir", str(out_dir / "score"), *extra)


if __name__ == "__main__":
    main()
