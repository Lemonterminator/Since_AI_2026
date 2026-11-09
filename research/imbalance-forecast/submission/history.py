"""Pre-June 2026 history for the panel, optionally re-priced with the post-June rule.

From 2026-06-01 00:00 Finnish time (2026-05-31T21:00Z) Fingrid prices imbalances
with the volume-weighted average of the mFRR marginal price and the aFRR
volume-weighted price in the dominating direction (news 27.4.2026; official
challenge brief p. 5). Before, the price was the max (up) / min (down) of the two.
``rule_prices`` rebuilds both from Fingrid's component series in gold_ext; they
reproduce 99.8 % (old rule, before) and 99.5 % (new rule, after) of the published
quarters, see history_relabel.py --check-rule.

Re-priced rows keep the *latest* first_usable_origin_utc of the published price
and all its components, so panel.MarketRows point-in-time rules still hold.

The formulas and the switch constants live in the pure ``price_rules`` module,
which the DL experiments import without this file's panel dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import panel as P
from price_rules import COMPONENTS, HISTORY_START, SWITCH, rule_prices  # noqa: F401

GOLD_EXT = P.REPO / "data" / "gold_ext" / "out" / "full_ext.parquet"


# ---- long rows with history -------------------------------------------------

def combined_rows(base_rows: pd.DataFrame, relabel: bool) -> pd.DataFrame:
    """Starter-package rows plus gold_ext history before them (no overlap)."""
    hist = history_rows(set(base_rows.series_id.unique()), relabel)
    rows = pd.concat([hist, base_rows], ignore_index=True)
    if rows.duplicated(["series_id", "timestamp_utc"]).any():
        raise ValueError("history overlaps the starter package rows")
    return rows


def history_rows(names: set[str], relabel: bool) -> pd.DataFrame:
    """Long-format rows before the starter package window, from gold_ext."""
    names = sorted(names | {P.IMBALANCE})
    want = ["timestamp_utc"] + [c for n in names for c in (n, f"{n}__first_usable_origin_utc")]
    want += [c for n in COMPONENTS + [P.SPOT] for c in (n, f"{n}__first_usable_origin_utc") if c not in want]
    avail = set(pq.read_schema(GOLD_EXT).names)
    wide = pd.read_parquet(GOLD_EXT, columns=[c for c in want if c in avail])
    wide = wide.loc[(wide.timestamp_utc >= HISTORY_START - pd.Timedelta(days=2))
                    & (wide.timestamp_utc < P.DATA_START - pd.Timedelta(days=2))].copy()
    if relabel:
        _, new = rule_prices(wide)
        fu = pd.concat([P.to_utc(wide[f"{n}__first_usable_origin_utc"]) for n in [P.IMBALANCE, *COMPONENTS, P.SPOT]],
                       axis=1).max(axis=1)
        wide[P.IMBALANCE] = np.where(wide[P.IMBALANCE].notna(), new, np.nan)
        wide[f"{P.IMBALANCE}__first_usable_origin_utc"] = fu
    parts = []
    for n in names:
        if n not in wide.columns:
            continue
        parts.append(pd.DataFrame({"series_id": n, "timestamp_utc": wide.timestamp_utc, "value": wide[n].to_numpy(),
                                   "first_usable_origin_utc": P.to_utc(wide[f"{n}__first_usable_origin_utc"])}))
    rows = pd.concat(parts, ignore_index=True)
    rows["timestamp_utc"] = P.to_utc(rows.timestamp_utc)
    return rows.loc[rows.first_usable_origin_utc.notna()]


def build_history_panel(relabel: bool, base_rows: pd.DataFrame, nuclear, end: pd.Timestamp = P.TEST_START - P.HOUR) -> pd.DataFrame:
    rows = combined_rows(base_rows, relabel)
    origins = P.hourly_origins(HISTORY_START + pd.Timedelta(days=2), end)
    panel = P.build_panel(P.MarketRows(rows), origins, nuclear)
    panel["regime_pre_june"] = (panel.delivery < SWITCH).astype(float)
    return panel
