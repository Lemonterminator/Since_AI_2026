"""Uncertainty-quantification report for one or more validation-period forecasts.

    python research/imbalance-forecast/submission/uq_metrics.py RUN_DIR [RUN_DIR ...] [--out DIR]

RUN_DIR holds forecast_backtest.csv and, for --model grid, quantile_grid.parquet.
Reports, next to the official score.py metrics:
  * quantile hit rates: share of actuals below P10 / P50 / P90 (ideal 0.10 / 0.50 / 0.90), by horizon band
  * PIT, 4 bins from P10/P50/P90 (ideal 10/40/40/10 %); with a grid, 20 bins of 5 %
  * CRPS from the 19-quantile grid: 2 x mean pinball over the grid (a quantile approximation;
    it ignores the tails beyond 5 % / 95 %, so it is a lower bound of the true CRPS)
  * ECE (10 equal-width bins) and log loss for direction and spike probabilities,
    plus reliability-diagram tables
Writes uq_summary.csv, uq_pit.csv and uq_reliability.csv in --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402

BANDS = {"all": (1, 40), "step_01_08": (1, 8), "step_09_20": (9, 20), "step_21_40": (21, 40)}
EPS = 1e-6


def load(run: Path, actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    sub = pd.read_csv(run / "forecast_backtest.csv")
    df = sub.merge(actuals, on="delivery_start_utc", how="inner")
    df = df[df.imbalance_price.notna()].reset_index(drop=True)
    grid = None
    if (run / "quantile_grid.parquet").exists():
        g = pd.read_parquet(run / "quantile_grid.parquet")
        g = df[["forecast_origin_utc", "step"]].merge(g, on=["forecast_origin_utc", "step"], how="left")
        grid = g[[c for c in g.columns if c.startswith("q")]]
    return df, grid


def ece(p: np.ndarray, o: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    t = pd.DataFrame({"bin": idx, "p": p, "o": o}).groupby("bin").agg(n=("p", "size"), mean_p=("p", "mean"),
                                                                     freq=("o", "mean")).reset_index()
    return float((t.n * (t.mean_p - t.freq).abs()).sum() / len(p)), t


def logloss(p: np.ndarray, o: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def report(name: str, df: pd.DataFrame, grid: pd.DataFrame | None):
    y = df.imbalance_price.to_numpy()
    summary, pit_rows, rel_rows = {"run": name, "rows": len(df)}, [], []
    for band, (a, b) in BANDS.items():
        m = df.step.between(a, b).to_numpy()
        for q in ("p10", "p50", "p90"):
            summary[f"hit_{q}_{band}"] = float((y[m] < df[q].to_numpy()[m]).mean())
    summary["mae_p50"] = float(np.abs(y - df.p50).mean())
    summary["rmse_p50"] = float(np.sqrt(((y - df.p50) ** 2).mean()))
    summary["coverage_p10_p90"] = float(((y >= df.p10) & (y <= df.p90)).mean())

    edges = np.column_stack([df.p10, df.p50, df.p90])
    four = (y[:, None] >= edges).sum(axis=1)
    for k, ideal in enumerate((.1, .4, .4, .1)):
        pit_rows.append({"run": name, "scheme": "p10_p50_p90", "bin": k, "share": float((four == k).mean()),
                         "ideal": ideal})
    if grid is not None and grid.notna().all().all():
        G = grid.to_numpy()
        qs = np.array([int(c[1:]) / 100 for c in grid.columns])
        d = y[:, None] - G
        summary["crps_grid"] = float(2 * np.mean(np.maximum(qs * d, (qs - 1) * d)))
        k20 = (y[:, None] >= G).sum(axis=1)
        for k in range(len(qs) + 1):
            pit_rows.append({"run": name, "scheme": "grid_20", "bin": k, "share": float((k20 == k).mean()),
                             "ideal": .05})
        summary["pit_grid_max_abs_dev"] = float(max(abs(r["share"] - .05) for r in pit_rows
                                                    if r["scheme"] == "grid_20"))

    spot = df.spot_price.to_numpy()
    targets = {
        "p_up": (df.direction == "up").to_numpy(float), "p_down": (df.direction == "down").to_numpy(float),
        "p_spike_high": (y > spot + P.SPIKE_MARGIN).astype(float),
        "p_spike_low": (y < spot - P.SPIKE_MARGIN).astype(float),
    }
    for col, o in targets.items():
        ok = ~np.isnan(spot) if col.startswith("p_spike") else df.direction.notna().to_numpy()
        e, table = ece(df[col].to_numpy()[ok], o[ok])
        summary[f"ece_{col}"] = e
        summary[f"logloss_{col}"] = logloss(df[col].to_numpy()[ok], o[ok])
        summary[f"brier_{col}"] = float(np.mean((df[col].to_numpy()[ok] - o[ok]) ** 2))
        rel_rows += [{"run": name, "target": col, **r} for r in table.to_dict("records")]
    return summary, pit_rows, rel_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--actuals", type=Path, default=P.STARTER / "data" / "actuals_validation.csv")
    ap.add_argument("--out", type=Path, default=P.HERE / "results")
    args = ap.parse_args()
    actuals = pd.read_csv(args.actuals)[["delivery_start_utc", "imbalance_price", "spot_price", "direction"]]
    S, PIT, REL = [], [], []
    for run in args.runs:
        s, p, r = report(run.name, *load(run, actuals))
        S.append(s); PIT += p; REL += r
    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(S)
    summary.to_csv(args.out / "uq_summary.csv", index=False, float_format="%.6f", lineterminator="\n")
    pd.DataFrame(PIT).to_csv(args.out / "uq_pit.csv", index=False, float_format="%.6f", lineterminator="\n")
    pd.DataFrame(REL).to_csv(args.out / "uq_reliability.csv", index=False, float_format="%.6f", lineterminator="\n")
    cols = ["run", "mae_p50", "rmse_p50", "coverage_p10_p90", "hit_p10_all", "hit_p50_all", "hit_p90_all",
            "crps_grid", "pit_grid_max_abs_dev", "ece_p_up", "logloss_p_up", "ece_p_spike_high",
            "ece_p_spike_low", "logloss_p_spike_high", "logloss_p_spike_low"]
    print(summary.reindex(columns=cols).round(4).T.to_string())


if __name__ == "__main__":
    main()
