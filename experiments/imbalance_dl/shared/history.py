"""Re-price pre-June 2026 FI imbalance history under the current rule.

Fingrid has priced imbalances with the volume-weighted mFRR/aFRR average since
2026-05-31T21:00Z (``SWITCH``). Earlier published prices follow the old
max/min rule. ``reprice_history`` rebuilds those earlier prices from Fingrid's
component series before any derived spread, scaler, history window, label or
spike is built, so the DL inputs and targets share one rule.

The formulas come from the pure submission module, which depends on numpy and
pandas only. Pre-switch rows keep the latest release of the published price,
the spot price and all components. Rows with an invalid direction or a finite
component without a release cannot be replayed, so they keep no price. Missing
published prices, the switch and later rows, and neighbour markets stay
untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PRICE, SPOT, SUFFIX, repository_module

rules = repository_module("research/imbalance-forecast/submission/price_rules.py", "imbalance_price_rules")
SWITCH = rules.SWITCH
HISTORY_START = rules.HISTORY_START
COMPONENTS = rules.COMPONENTS
# Direction-selected inputs of the rule as (up column, down column) pairs.
PAIRS = (("fi_mfrr_up_price", "fi_mfrr_down_price"),
         ("fi_mfrr_up_activated", "fi_mfrr_down_activated"),
         ("fi_afrr_up_price", "fi_afrr_down_price"),
         ("fi_afrr_up_activated", "fi_afrr_down_activated"))


def reprice_history(gold: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the pre-switch FI imbalance prices rebuilt from components."""
    result = gold.copy()
    before = np.asarray(result.index < SWITCH, dtype=bool)
    if not before.any():
        return result
    inputs = (PRICE, SPOT, *COMPONENTS)
    missing = [name for key in inputs for name in (key, key + SUFFIX) if name not in result]
    if missing:
        raise ValueError(f"Cannot re-price history without the rule inputs: {missing}")
    _, rebuilt = rules.rule_prices(result)

    direction = result["fi_mfrr_direction"].to_numpy(dtype=float)
    up, down = direction == 1, direction == -1
    directed = up | down
    usable = np.isin(direction, (1.0, -1.0, 0.0))
    for up_name, down_name in PAIRS:
        values = np.where(up, result[up_name].to_numpy(dtype=float),
                          result[down_name].to_numpy(dtype=float))
        released = np.where(up, result[up_name + SUFFIX].notna().to_numpy(),
                            result[down_name + SUFFIX].notna().to_numpy())
        usable &= ~directed | ~np.isfinite(values) | released
    # Without a dominating direction the rule falls back to the day-ahead spot price.
    spot = result[SPOT].to_numpy(dtype=float)
    usable &= directed | ~np.isfinite(spot) | result[SPOT + SUFFIX].notna().to_numpy()
    # The published price and the direction are also needed to replay the rule.
    usable &= result[PRICE + SUFFIX].notna().to_numpy()
    usable &= result["fi_mfrr_direction" + SUFFIX].notna().to_numpy()

    published = result[PRICE].notna().to_numpy()
    repriced = before & published & usable
    latest = pd.concat([result[key + SUFFIX] for key in inputs], axis=1).max(axis=1)
    result.loc[repriced, PRICE] = pd.Series(rebuilt, index=result.index)[repriced]
    result.loc[repriced, PRICE + SUFFIX] = latest[repriced]
    masked = before & published & ~usable
    result.loc[masked, PRICE] = np.nan
    result.loc[masked, PRICE + SUFFIX] = pd.NaT
    return result
