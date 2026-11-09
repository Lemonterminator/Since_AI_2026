"""Leakage-safe (origin, step) feature panel for the imbalance price forecast.

Every input value goes through ``MarketRows.at``: a value of series ``s`` for
delivery quarter ``d`` is used at origin ``t`` only if that exact source row
has ``first_usable_origin_utc <= t``. "Latest" state features take the largest
delivery timestamp whose row is usable at ``t``. Rolling windows end at that
latest usable timestamp. Labels are kept in separate columns together with
``label_usable`` (the latest ``first_usable_origin_utc`` of the label rows), so
the training set can be cut by publication time, not by delivery time.

Nuclear UMM features come from ``data/gold_ext/out/nuclear_asof.parquet``,
which is already built per origin from the message versions known at that
origin; ``leakage_check.py`` re-derives sampled origins from the raw UMM rows
to prove it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STARTER = REPO / "data" / "silver" / "starter_package_v1"
SILVER = REPO / "data" / "silver"
NUCLEAR_ASOF = REPO / "data" / "gold_ext" / "out" / "nuclear_asof.parquet"
UTC = "datetime64[ns, UTC]"

IMBALANCE = "fi_imbalance_price"
SPOT = "fi_spot_price"
DIRECTION = "fi_mfrr_direction"
HOUR = pd.Timedelta(hours=1)
QUARTER = pd.Timedelta(minutes=15)
STEPS = np.arange(1, 41)
SPIKE_MARGIN = 50.0

DATA_START = pd.Timestamp("2026-05-31T21:00Z")
FIRST_ORIGIN = pd.Timestamp("2026-06-02T00:00Z")
VALIDATION_START = pd.Timestamp("2026-09-06T21:00Z")
TEST_START = pd.Timestamp("2026-10-04T21:00Z")
LAST_TEST_ORIGIN = pd.Timestamp("2026-11-01T21:00Z")

# External long-format series (data/silver/<source>/<source>.parquet). All carry
# their own conservative first_usable_origin_utc (see data/bronze/README.md).
EXTERNAL = {
    "fingrid_extra": ["fi_wind_rt", "fi_nuclear_rt", "fi_hydro_rt"],
    "esett": [
        "esett_se1_spot_price", "esett_se3_spot_price",
        "esett_se1_imbalance_price", "esett_se3_imbalance_price",
        "esett_se1_main_direction", "esett_se3_main_direction",
    ],
    "elering": ["ee_spot_price"],
    # Fingrid system-balance measurements (gold_ext, 2026-11-07); see REPORT_SYSTEM.md
    "fingrid_system": [
        "fi_imbalance_calc_rt", "fi_surplus_rt",
        "fi_afrr_up_activated", "fi_afrr_down_activated",
        "fi_fcrn_up_activated", "fi_fcrn_down_activated",
    ],
}

# ---- feature groups -------------------------------------------------------
CALENDAR = ["step", "delivery_hour", "delivery_weekday"]
CORE = [*CALENDAR, "last_spread"]
RECENT = [
    "last_direction", "up_share_8q", "spread_mean_8q", "spread_std_8q",
    "spread_change_1q", "last_mfrr_up", "last_mfrr_down",
]
DAY_AHEAD = ["spot_level", "spot_ramp_1h", "wind_da", "net_load_da", "net_load_ramp_1h"]
# (3) delivery-period plan minus latest known value
PLAN = [
    "wind_da_minus_latest_rt",       # planned wind at delivery - latest real-time wind
    "wind_err_last",                 # latest real-time wind - its own day-ahead forecast
    "load_da_minus_latest",          # planned consumption at delivery - latest measured
    "load_err_last",                 # latest measured consumption - its day-ahead forecast
    "ee_export_headroom",            # FI->EE DA capacity at delivery - latest FI->EE flow
    "ee_import_headroom",            # EE->FI DA capacity at delivery + latest FI->EE flow
    "net_export_last",               # latest sum of measured cross-border flows
    "se3_load_da_change",            # SE3 load forecast at delivery - at latest quarter
    "fi_minus_se1_spot", "fi_minus_se3_spot", "fi_minus_ee_spot",
]
# (4) nuclear outage / availability
NUCLEAR = [
    "fi_nuc_available_mw", "fi_nuc_unavail_unplanned_mw", "se3_nuc_unavail_mw",
    "fi_unplanned_last_start_age_h", "fi_unplanned_active_restore_h",
    "fi_nuc_known_revisions_24h", "nuclear_rt_last", "nuc_avail_minus_rt",
]
# (5) real-time system balance (Fingrid 397 / 198 / 349 / 350 / 344 / 343), latest usable
# quarter and mean of the last 4 usable quarters; down activations are negative MW
SYSTEM_SERIES = {
    "imb_calc": "fi_imbalance_calc_rt", "surplus": "fi_surplus_rt",
    "afrr_up": "fi_afrr_up_activated", "afrr_down": "fi_afrr_down_activated",
    "fcrn_up": "fi_fcrn_up_activated", "fcrn_down": "fi_fcrn_down_activated",
}
IMB_CALC = ["imb_calc_last", "imb_calc_mean4"]
RESERVES = ["afrr_up_last", "afrr_down_last", "afrr_net_last", "afrr_net_mean4", "fcrn_net_last", "fcrn_net_mean4"]
SURPLUS = ["surplus_last", "surplus_mean4"]
SYSTEM = IMB_CALC + RESERVES + SURPLUS
# subset of PLAN that improved most metrics in the per-feature rolling-fold test (factor_round3.py --single)
PLAN_STABLE = ["load_da_minus_latest", "wind_err_last"]
NEIGHBOUR = ["se1_last_spread", "se3_last_spread", "se3_last_direction"]

GROUPS = {
    "calendar": CALENDAR, "core": CORE, "recent": RECENT, "day_ahead": DAY_AHEAD,
    "plan": PLAN, "nuclear": NUCLEAR, "neighbour": NEIGHBOUR,
    "imb_calc": IMB_CALC, "reserves": RESERVES, "surplus": SURPLUS, "system": SYSTEM,
}
VARIANTS = {
    "core_both": CORE + RECENT + DAY_AHEAD,
    "core_both_plan": CORE + RECENT + DAY_AHEAD + PLAN,
    "core_both_plan_stable": CORE + RECENT + DAY_AHEAD + PLAN_STABLE,
    "core_both_nuclear": CORE + RECENT + DAY_AHEAD + NUCLEAR,
    "core_both_neighbour": CORE + RECENT + DAY_AHEAD + NEIGHBOUR,
    "core_both_plan_nuclear": CORE + RECENT + DAY_AHEAD + PLAN + NUCLEAR,
    # system-signal experiments on top of the submission set (system_signals.py)
    "core_both_plan_stable_imb_calc": CORE + RECENT + DAY_AHEAD + PLAN_STABLE + IMB_CALC,
    "core_both_plan_stable_reserves": CORE + RECENT + DAY_AHEAD + PLAN_STABLE + RESERVES,
    "core_both_plan_stable_imb_reserves": CORE + RECENT + DAY_AHEAD + PLAN_STABLE + IMB_CALC + RESERVES,
    "core_both_plan_stable_system": CORE + RECENT + DAY_AHEAD + PLAN_STABLE + SYSTEM,
}


def to_utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True).astype(UTC)


def load_rows(external: bool = True, since: pd.Timestamp = DATA_START - pd.Timedelta(days=2)) -> pd.DataFrame:
    """Starter package rows (+ selected external silver rows) in long format."""
    data = STARTER / "data"
    parts = [pd.read_parquet(data / "inputs.parquet"), pd.read_csv(data / "spot_nordpool.csv")]
    if external:
        for source, names in EXTERNAL.items():
            path = SILVER / source / f"{source}.parquet"
            if path.exists():
                block = pd.read_parquet(path, filters=[("series_id", "in", names)])
                parts.append(block)
    rows = pd.concat(parts, ignore_index=True)
    for col in ("timestamp_utc", "first_usable_origin_utc"):
        rows[col] = to_utc(rows[col])
    rows = rows.loc[rows.timestamp_utc >= since, ["series_id", "timestamp_utc", "value", "first_usable_origin_utc"]]
    if rows.duplicated(["series_id", "timestamp_utc"]).any():
        raise ValueError("duplicate series_id + timestamp_utc rows")
    return rows.reset_index(drop=True)


class MarketRows:
    """Point-in-time access to long-format rows."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        self.series = {
            name: block.set_index("timestamp_utc").sort_index()
            for name, block in rows.groupby("series_id")
        }
        self._latest: dict[str, pd.DataFrame] = {}

    def has(self, name: str) -> bool:
        return name in self.series

    def at(self, name: str, delivery, origin) -> np.ndarray:
        """Value of ``name`` at ``delivery`` where that row is usable at ``origin``; else NaN."""
        delivery = pd.DatetimeIndex(delivery)
        if name not in self.series:
            return np.full(len(delivery), np.nan)
        rows = self.series[name].reindex(delivery)
        usable = pd.DatetimeIndex(rows.first_usable_origin_utc) <= pd.DatetimeIndex(origin)
        return rows.value.where(usable).to_numpy(dtype=float)

    def latest_stamp(self, name: str, origin: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Largest delivery timestamp with a non-missing row usable at each origin
        (and not after the origin itself, so only realised values count)."""
        if name not in self._latest:
            x = self.series[name]
            x = x.loc[x.value.notna()].reset_index()[["first_usable_origin_utc", "timestamp_utc"]]
            x = x.groupby("first_usable_origin_utc").timestamp_utc.max().cummax().reset_index()
            self._latest[name] = x
        x = self._latest[name]
        merged = pd.merge_asof(
            pd.DataFrame({"origin": pd.DatetimeIndex(origin).astype(UTC)}),
            x.astype({"first_usable_origin_utc": UTC, "timestamp_utc": UTC}),
            left_on="origin", right_on="first_usable_origin_utc",
        )
        stamp = pd.DatetimeIndex(merged.timestamp_utc)
        # realised series are never published ahead of delivery; for safety clip to the origin
        return stamp.where(stamp <= pd.DatetimeIndex(origin), pd.DatetimeIndex(origin) - QUARTER)

    def latest(self, name: str, origin: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
        stamp = self.latest_stamp(name, origin)
        return stamp, self.at(name, stamp, origin)


def hourly_origins(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Full-hour origins in [start, end]."""
    return pd.date_range(start, end, freq="h").astype(UTC)


def _origin_features(m: MarketRows, hourly: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    stamp, last_price = m.latest(IMBALANCE, hourly)
    last_spread = last_price - m.at(SPOT, stamp, hourly)
    dir_stamp, last_direction = m.latest(DIRECTION, hourly)

    n = 8
    offsets = pd.to_timedelta(np.tile(np.arange(n) * 15, len(hourly)), unit="min")
    rep = hourly.repeat(n)
    win = stamp.repeat(n) - offsets
    spreads = pd.DataFrame((m.at(IMBALANCE, win, rep) - m.at(SPOT, win, rep)).reshape(-1, n))
    dirs = m.at(DIRECTION, dir_stamp.repeat(n) - offsets, rep)
    up = pd.DataFrame(np.where(np.isnan(dirs), np.nan, dirs == 1).reshape(-1, n))
    _, mfrr_up = m.latest("fi_mfrr_up_activated", hourly)
    _, mfrr_down = m.latest("fi_mfrr_down_activated", hourly)

    f = {
        "last_price": last_price, "last_spread": last_spread, "last_direction": last_direction,
        "last_price_age_min": (hourly - stamp).total_seconds().to_numpy() / 60,
        "up_share_8q": up.mean(axis=1).to_numpy(),
        "spread_mean_8q": spreads.mean(axis=1).to_numpy(),
        "spread_std_8q": spreads.std(axis=1, ddof=0).to_numpy(),
        "spread_change_1q": (spreads[0] - spreads[1]).to_numpy(),
        "last_mfrr_up": mfrr_up, "last_mfrr_down": mfrr_down,
    }

    # (3) latest-known values for plan-minus-latest features
    if m.has("fi_wind_rt"):
        w_stamp, w_last = m.latest("fi_wind_rt", hourly)
        f["wind_rt_last"] = w_last
        f["wind_err_last"] = w_last - m.at("fi_wind_fc_da", w_stamp, hourly)
    else:
        f["wind_rt_last"] = f["wind_err_last"] = np.full(len(hourly), np.nan)
    c_stamp, c_last = m.latest("fi_consumption", hourly)
    f["load_last"] = c_last
    f["load_err_last"] = c_last - m.at("fi_consumption_fc_da", c_stamp, hourly)
    flows = []
    for name in ("fi_se1_flow", "fi_se3_flow", "fi_no4_flow", "fi_ee_flow"):
        _, v = m.latest(name, hourly)
        flows.append(v)
        if name == "fi_ee_flow":
            f["ee_flow_last"] = v
    f["net_export_last"] = np.sum(flows, axis=0)
    f["latest_quarter"] = stamp  # helper, not a model feature

    # (4) latest real-time nuclear output
    f["nuclear_rt_last"] = m.latest("fi_nuclear_rt", hourly)[1] if m.has("fi_nuclear_rt") else np.full(len(hourly), np.nan)

    # neighbour balancing state (eSett, 60 min after period end)
    for area in ("se1", "se3"):
        name = f"esett_{area}_imbalance_price"
        if m.has(name):
            s, v = m.latest(name, hourly)
            f[f"{area}_last_spread"] = v - m.at(f"esett_{area}_spot_price", s, hourly)
        else:
            f[f"{area}_last_spread"] = np.full(len(hourly), np.nan)
    f["se3_last_direction"] = (
        m.latest("esett_se3_main_direction", hourly)[1] if m.has("esett_se3_main_direction")
        else np.full(len(hourly), np.nan)
    )
    # (5) real-time system balance
    for key, name in SYSTEM_SERIES.items():
        if not m.has(name):
            f[f"{key}_last"] = f[f"{key}_mean4"] = np.full(len(hourly), np.nan)
            continue
        s_stamp = m.latest_stamp(name, hourly)
        win = s_stamp.repeat(4) - pd.to_timedelta(np.tile(np.arange(4) * 15, len(hourly)), unit="min")
        vals = m.at(name, win, hourly.repeat(4)).reshape(-1, 4)
        f[f"{key}_last"] = vals[:, 0]
        n_ok = (~np.isnan(vals)).sum(axis=1)
        f[f"{key}_mean4"] = np.where(n_ok > 0, np.nansum(vals, axis=1) / np.maximum(n_ok, 1), np.nan)
    for kind in ("afrr", "fcrn"):
        f[f"{kind}_net_last"] = f[f"{kind}_up_last"] + f[f"{kind}_down_last"]
        f[f"{kind}_net_mean4"] = f[f"{kind}_up_mean4"] + f[f"{kind}_down_mean4"]
    return f


def build_panel(m: MarketRows, hourly: pd.DatetimeIndex, nuclear: pd.DataFrame | None = None,
                with_labels: bool = True) -> pd.DataFrame:
    """One row per (origin, step)."""
    of = _origin_features(m, hourly)
    k = len(STEPS)
    origin = hourly.repeat(k)
    step = np.tile(STEPS, len(hourly))
    delivery = origin + pd.to_timedelta((step - 1) * 15, unit="min")
    prior = delivery - HOUR
    rep = {name: np.repeat(np.asarray(v), k) for name, v in of.items()}

    spot = m.at(SPOT, delivery, origin)
    wind = m.at("fi_wind_fc_da", delivery, origin)
    solar = m.at("fi_solar_fc_da", delivery, origin)
    load = m.at("fi_consumption_fc_da", delivery, origin)
    net_load = load - wind - solar
    prior_net = (m.at("fi_consumption_fc_da", prior, origin) - m.at("fi_wind_fc_da", prior, origin)
                 - m.at("fi_solar_fc_da", prior, origin))
    latest_q = pd.DatetimeIndex(rep.pop("latest_quarter"))

    p = pd.DataFrame({"origin": origin, "step": step, "delivery": delivery})
    for name, v in rep.items():
        p[name] = v
    p["delivery_hour"] = m.at("cal_hour_local", delivery, origin)
    p["delivery_weekday"] = m.at("cal_weekday_local", delivery, origin)
    p["delivery_holiday"] = m.at("cal_is_holiday_fi", delivery, origin)
    p["spot_level"] = spot
    p["spot_ramp_1h"] = spot - m.at(SPOT, prior, origin)
    p["wind_da"] = wind
    p["net_load_da"] = net_load
    p["net_load_ramp_1h"] = net_load - prior_net

    # (3) plan minus latest known
    p["wind_da_minus_latest_rt"] = wind - p["wind_rt_last"]
    p["load_da_minus_latest"] = load - p["load_last"]
    p["ee_export_headroom"] = m.at("fi_ee_capacity_da", delivery, origin) - p["ee_flow_last"]
    p["ee_import_headroom"] = m.at("ee_fi_capacity_da", delivery, origin) + p["ee_flow_last"]
    p["se3_load_da_change"] = m.at("se3_load_fc_da", delivery, origin) - m.at("se3_load_fc_da", latest_q, origin)
    p["fi_minus_se1_spot"] = spot - m.at("esett_se1_spot_price", delivery, origin)
    p["fi_minus_se3_spot"] = spot - m.at("esett_se3_spot_price", delivery, origin)
    p["fi_minus_ee_spot"] = spot - m.at("ee_spot_price", delivery, origin)

    # (4) nuclear UMM as-of features
    if nuclear is not None:
        cols = [c for c in NUCLEAR if c in nuclear.columns]
        nuc = nuclear.set_index(["forecast_origin_utc", "step"])[cols]
        idx = pd.MultiIndex.from_arrays([origin, step])
        joined = nuc.reindex(idx)
        for c in cols:
            p[c] = joined[c].to_numpy()
    for c in ("fi_nuc_available_mw", "fi_nuc_unavail_unplanned_mw", "se3_nuc_unavail_mw",
              "fi_unplanned_last_start_age_h", "fi_unplanned_active_restore_h", "fi_nuc_known_revisions_24h"):
        if c not in p:
            p[c] = np.nan
    p["nuc_avail_minus_rt"] = p["fi_nuc_available_mw"] - p["nuclear_rt_last"]

    if with_labels:
        price = m.series[IMBALANCE].reindex(delivery)
        dirn = m.series[DIRECTION].reindex(delivery)
        spot_rows = m.series[SPOT].reindex(delivery)
        p["imbalance_price"] = price.value.to_numpy()
        p["spot_price"] = spot_rows.value.to_numpy()
        p["spread"] = p.imbalance_price - p.spot_price
        p["direction"] = dirn.value.map({1: "up", -1: "down", 0: "none"}).to_numpy()
        p["label_usable"] = pd.concat([
            price.first_usable_origin_utc.reset_index(drop=True),
            dirn.first_usable_origin_utc.reset_index(drop=True),
            spot_rows.first_usable_origin_utc.reset_index(drop=True),
        ], axis=1).max(axis=1, skipna=False).to_numpy()
        p["spike"] = np.select(
            [p.imbalance_price > p.spot_price + SPIKE_MARGIN, p.imbalance_price < p.spot_price - SPIKE_MARGIN],
            ["high", "low"], default="normal",
        )
    return p


def load_nuclear(path: Path = NUCLEAR_ASOF) -> pd.DataFrame | None:
    if not path.exists():
        return None
    nuc = pd.read_parquet(path)
    nuc["forecast_origin_utc"] = to_utc(nuc["forecast_origin_utc"])
    return nuc.loc[nuc.forecast_origin_utc >= DATA_START]
