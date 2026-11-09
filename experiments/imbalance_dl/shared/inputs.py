"""Feature specifications and target-price selection for the extended Gold layer.

This module is deliberately an input adapter.  It does not build batches, fit
scalers, compute labels, or change any model internals.  Call ``prepare_inputs``
*before* fitting a scaler or constructing a ``BatchBuilder`` so that every
downstream path (history, labels, spikes and normalization) sees the same
imbalance-price definition.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .data import DIRECTION, Feature, PRICE, ROOT, SUFFIX, official_features

# The committed extended layer already contains the pre-June conversion.  Never
# recompute it here: consuming the committed column is the audit-friendly path.
NEW_RULE = "fi_imbalance_price_newrule"
PRICE_SERIES = (PRICE, NEW_RULE)

_NONCONTINUOUS_UNITS = {"0/1", "-1/0/1", "0=Mon..6=Sun", "code", "octas"}

# Units are taken from build_nuclear_asof.py's documented output columns.
NUCLEAR_UNITS = {
    "fi_nuc_unavail_planned_mw": "MW",
    "fi_nuc_unavail_unplanned_mw": "MW",
    "fi_nuc_unavail_mw": "MW",
    "fi_nuc_available_mw": "MW",
    "se3_nuc_unavail_planned_mw": "MW",
    "se3_nuc_unavail_unplanned_mw": "MW",
    "se3_nuc_unavail_mw": "MW",
    "ol3_unavail_mw": "MW",
    "fi_unplanned_last_start_age_h": "h",
    "fi_unplanned_active_restore_h": "h",
    "fi_nuc_known_revisions_24h": "count",
}


def prepare_inputs(gold: pd.DataFrame, price_series: str = NEW_RULE) -> pd.DataFrame:
    """Return a deep copy with one selected imbalance-price definition aliased.

    ``price_series`` must be either the committed new-rule column or the
    published-price column.  Its paired ``__first_usable_origin_utc`` column is
    copied to the canonical availability column at the same time.  The selected
    source and its availability are required; there is no fallback to the
    canonical columns and no in-module re-pricing.

    All columns except the canonical price and its availability stay unchanged.
    The input frame is never mutated.
    """
    if price_series not in PRICE_SERIES:
        raise ValueError(
            f"price_series must be one of {PRICE_SERIES}, got {price_series!r}"
        )

    source_availability = price_series + SUFFIX
    missing = [column for column in (price_series, source_availability)
               if column not in gold.columns]
    if missing:
        raise ValueError(
            f"Selected price series is incomplete; missing required columns: {missing}"
        )

    prepared = gold.copy(deep=True)
    # Assigning these two canonical columns is the whole adaptation.  Keep the
    # selected source columns in place so provenance and later ablations remain
    # inspectable.
    prepared[PRICE] = prepared[price_series]
    prepared[PRICE + SUFFIX] = prepared[source_availability]
    return prepared


def extended_features(schema_path: str | Path = ROOT / "data/gold_ext/out/schema.json") -> tuple[Feature, ...]:
    """Describe all scalar inputs in schema order, excluding both price labels."""
    schema = json.loads(Path(schema_path).read_text())
    known = {feature.name for feature in official_features() if feature.role == "known"}
    features = []
    for name, metadata in sorted(schema.items()):
        if name in PRICE_SERIES:
            continue
        sources, unit = metadata["sources"], metadata["unit"]
        day_ahead = (name.startswith("esett_") or sources == ["elering"]) and name.endswith("_spot_price")
        continuous = not (name.startswith("cal_") or name == DIRECTION
                          or name.endswith("_direction") or unit in _NONCONTINUOUS_UNITS)
        features.append(Feature(name, ",".join(sources), unit,
                                "known" if name in known or day_ahead else "observed",
                                metadata["availability_column"], continuous))
    return tuple(features)


def nuclear_features() -> tuple[Feature, ...]:
    """Return the 11 explicit, opt-in nuclear as-of event features.

    These columns live in ``nuclear_asof.parquet`` and are keyed by
    ``(forecast_origin_utc, step)``.  ``BatchBuilder`` resolves historical
    snapshots at each historical origin and future snapshots at the current
    origin, so revisions published after an origin can never leak in.
    """
    return tuple(
        Feature(
            name=name,
            group="nuclear",
            unit=unit,
            role="event",
            availability_rule="origin/step snapshot from nuclear_asof.parquet",
            continuous=True,
            last_known=False,
        )
        for name, unit in NUCLEAR_UNITS.items()
    )
