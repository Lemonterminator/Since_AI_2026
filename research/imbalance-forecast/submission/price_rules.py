"""Pure imbalance-price rule formulas and the regime-switch constants.

From 2026-06-01 00:00 Finnish time (2026-05-31T21:00Z) Fingrid prices
imbalances with the volume-weighted average of the mFRR marginal price and the
aFRR volume-weighted price in the dominating direction (news 27.4.2026;
official challenge brief p. 5). Before, the price was the max (up) / min
(down) of the two. ``rule_prices`` rebuilds both from Fingrid's component
series; they reproduce 99.8 % (old rule, before) and 99.5 % (new rule, after)
of the published quarters, see history_relabel.py --check-rule.

This module only depends on numpy and pandas so that both the submission tree
and the DL experiments can import it without the panel or sys.path changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SWITCH = pd.Timestamp("2026-05-31T21:00Z")  # 2026-06-01 00:00 Europe/Helsinki
HISTORY_START = pd.Timestamp("2025-04-01T00:00Z")
COMPONENTS = ["fi_mfrr_direction", "fi_mfrr_up_price", "fi_mfrr_down_price", "fi_afrr_up_price",
              "fi_afrr_down_price", "fi_mfrr_up_activated", "fi_mfrr_down_activated",
              "fi_afrr_up_activated", "fi_afrr_down_activated"]


def rule_prices(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Old (max/min) and new (volume-weighted) imbalance price per quarter from Fingrid components."""
    up, nodir = d.fi_mfrr_direction == 1, d.fi_mfrr_direction == 0
    mp = np.where(up, d.fi_mfrr_up_price, d.fi_mfrr_down_price)
    ap = np.where(up, d.fi_afrr_up_price, d.fi_afrr_down_price)
    mv = np.abs(np.where(up, d.fi_mfrr_up_activated, d.fi_mfrr_down_activated))
    av = np.abs(np.where(up, d.fi_afrr_up_activated, d.fi_afrr_down_activated))
    no_afrr = np.isnan(ap)  # aFRR price missing = no aFRR activation in that direction
    av = np.where(no_afrr, 0.0, np.nan_to_num(av))
    old = np.where(no_afrr, mp, np.where(up, np.fmax(mp, ap), np.fmin(mp, ap)))
    with np.errstate(invalid="ignore", divide="ignore"):
        new = np.where(av > 0, (mp * mv + np.nan_to_num(ap) * av) / (mv + av), mp)
    spot = d.fi_spot_price.to_numpy()
    return np.where(nodir, spot, old), np.where(nodir, spot, new)
