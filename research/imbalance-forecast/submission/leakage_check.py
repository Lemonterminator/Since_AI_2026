"""Automated leakage checks for the submission pipeline.

    python research/imbalance-forecast/submission/leakage_check.py [--origins 40]

1. Future-perturbation test (all panel features, incl. plan/nuclear/neighbour):
   for sampled origins t, rows with first_usable_origin_utc > t are (a) removed and
   (b) overwritten with garbage. The features at t must be identical to those built
   from the untouched data. Any feature that changes reads a row before it is usable.
2. Nuclear UMM table: for sampled origins, data/gold_ext/out/nuclear_asof.parquet must
   equal features recomputed from only the UMM versions with first_usable_origin_utc <= t,
   and must not change when later versions are deleted.
3. Training set: every label row (price, direction, spot) of the training set must be
   published by the cut-off, and no training delivery may fall in the test period.
4. Submission file (optional --submission): data_cutoff_utc <= forecast_origin_utc.

Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402
import history as H  # noqa: E402
from make_submission import training_rows  # noqa: E402

LABELS = {"imbalance_price", "spot_price", "spread", "direction", "label_usable", "spike"}


def sample_origins(n: int, seed: int = 7) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed)
    allo = P.hourly_origins(P.FIRST_ORIGIN, P.LAST_TEST_ORIGIN)
    fixed = pd.DatetimeIndex([
        P.FIRST_ORIGIN, P.VALIDATION_START, P.TEST_START, P.LAST_TEST_ORIGIN,
        pd.Timestamp("2026-09-09T17:00Z"),  # OL3 trip, UMM usable from 17:00
        pd.Timestamp("2026-10-25T00:00Z"), pd.Timestamp("2026-10-25T01:00Z"),  # DST change
        pd.Timestamp("2026-10-04T11:00Z"),  # D-1 day-ahead publication edge (13:00/14:00 local)
        pd.Timestamp("2026-10-04T12:00Z"),
    ]).astype(P.UTC)
    rand = pd.DatetimeIndex(rng.choice(allo, size=max(n - len(fixed), 0), replace=False)).astype(P.UTC)
    return fixed.append(rand).unique().sort_values()


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    bad = []
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(np.asarray(x).dtype, np.number):
            # tolerance only absorbs summation-order noise; garbage values are 9.99e6
            same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-9)
        else:
            same = (pd.isna(x) & pd.isna(y)) | (x == y)
        if not np.all(same):
            bad.append(c)
    return bad


def check_perturbation(rows: pd.DataFrame, nuclear, origins: pd.DatetimeIndex) -> list[str]:
    full = P.build_panel(P.MarketRows(rows), origins, nuclear, with_labels=False)
    problems = []
    for t in origins:
        ref = full.loc[full.origin == t].reset_index(drop=True)
        future = rows.first_usable_origin_utc > t
        truncated = rows.loc[~future]
        garbage = rows.copy()
        garbage.loc[future, "value"] = 9.99e6
        for label, variant in (("truncated", truncated), ("garbage", garbage)):
            got = P.build_panel(P.MarketRows(variant), pd.DatetimeIndex([t]), nuclear, with_labels=False)
            bad = frames_equal(ref, got.reset_index(drop=True))
            if bad:
                problems.append(f"{t} {label}: features depend on not-yet-usable rows: {bad}")
    return problems


def check_nuclear(origins: pd.DatetimeIndex) -> list[str]:
    path = P.REPO / "data" / "gold_ext" / "build_nuclear_asof.py"
    spec = spec_from_file_location("nuc", path)
    nuc = module_from_spec(spec)
    spec.loader.exec_module(nuc)
    umm = nuc.load(P.SILVER / "umm" / "umm.parquet")
    table = P.load_nuclear()
    if table is None:
        return ["nuclear_asof.parquet missing"]
    problems = []
    for t in origins:
        deliveries = pd.date_range(t, periods=40, freq="15min")
        known_only = umm.loc[umm.first_usable_origin_utc <= t]
        a = pd.DataFrame(nuc.features_at(known_only, t, deliveries))
        b = pd.DataFrame(nuc.features_at(umm, t, deliveries))
        stored = table.loc[table.forecast_origin_utc == t].sort_values("step").reset_index(drop=True)[a.columns]
        if frames_equal(a, b):
            problems.append(f"{t}: nuclear features change when later UMM versions are removed")
        bad = frames_equal(a, stored)
        if bad:
            problems.append(f"{t}: stored nuclear_asof differs from point-in-time recompute: {bad}")
    return problems


def check_training(panel: pd.DataFrame) -> list[str]:
    problems = []
    for cutoff in (P.VALIDATION_START, P.TEST_START):
        train = training_rows(panel, cutoff)
        if (pd.DatetimeIndex(train.label_usable) > cutoff).any():
            problems.append(f"cut-off {cutoff}: training label published after cut-off")
        if (train.delivery >= P.TEST_START).any():
            problems.append(f"cut-off {cutoff}: training delivery inside the test period")
        if (train.origin >= cutoff).any():
            problems.append(f"cut-off {cutoff}: training origin at/after cut-off")
    return problems


def check_submission(path: Path) -> list[str]:
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    given = sub.data_cutoff_utc != ""
    late = pd.to_datetime(sub.data_cutoff_utc[given], utc=True) > pd.to_datetime(sub.forecast_origin_utc[given], utc=True)
    return [f"{int(late.sum())} rows with data_cutoff_utc after the origin"] if late.any() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origins", type=int, default=30)
    ap.add_argument("--submission", type=Path, action="append", default=[])
    ap.add_argument("--history", choices=("none", "raw", "relabel"), default="none",
                    help="also check the gold_ext history rows (history.py), with extra pre-June origins")
    args = ap.parse_args()

    rows = P.load_rows()
    nuclear = P.load_nuclear()
    origins = sample_origins(args.origins)
    first_origin = P.FIRST_ORIGIN
    if args.history != "none":
        rows = H.combined_rows(rows, relabel=args.history == "relabel")
        first_origin = H.HISTORY_START + pd.Timedelta(days=2)
        rng = np.random.default_rng(11)
        pre = P.hourly_origins(first_origin, H.SWITCH)
        extra = pd.DatetimeIndex([H.SWITCH - P.HOUR, H.SWITCH, H.SWITCH + P.HOUR]).astype(P.UTC)
        extra = extra.append(pd.DatetimeIndex(rng.choice(pre, size=12, replace=False)).astype(P.UTC))
        origins = origins.append(extra).unique().sort_values()
    report = {}
    report["perturbation"] = check_perturbation(rows, nuclear, origins)
    # nuclear_asof is only loaded from the starter window on (panel.load_nuclear)
    report["nuclear_asof"] = check_nuclear(origins[origins >= P.FIRST_ORIGIN])
    panel = P.build_panel(P.MarketRows(rows), P.hourly_origins(first_origin, P.LAST_TEST_ORIGIN), nuclear)
    report["training_set"] = check_training(panel)
    for s in args.submission:
        report[f"submission:{s.name}"] = check_submission(s)

    failed = False
    for name, problems in report.items():
        status = "OK" if not problems else f"FAILED ({len(problems)})"
        print(f"{name:28s} {status}  [{len(origins)} origins]" if name in ("perturbation", "nuclear_asof")
              else f"{name:28s} {status}")
        for p in problems[:10]:
            print("   ", p)
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
