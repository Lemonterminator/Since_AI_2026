"""Experiment-only multi-market inputs. Never write to the source data layers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
from itertools import islice

import numpy as np
import pandas as pd
import torch

from .contracts import ForecastBatch, Targets
from .data import (BatchBuilder, Feature, RobustAsinhScaler, StandardScaler, TargetSeries,
                   QUARTER, SPOT, SUFFIX, concatenate_batches, official_features, utc)
from .history import reprice_history


MARKET_NAMES = ("FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5", "DK1", "DK2")
MARKET_IDS = {name: i for i, name in enumerate(MARKET_NAMES)}
TARGET_SETS = {"fi": ("FI",), "nearby": ("FI", "SE1", "SE2", "SE3"), "nordic": MARKET_NAMES}
# Fingrid system-balance observations in canonical order; all values are MW.
SYSTEM_BALANCE_SERIES = ("fi_imbalance_calc_rt", "fi_surplus_rt", "fi_mfrr_need",
                         "fi_afrr_up_activated", "fi_afrr_down_activated",
                         "fi_fcrn_up_activated", "fi_fcrn_down_activated")
OFFICIAL_START = pd.Timestamp("2026-05-31T21:00:00Z")
TRAIN_END = pd.Timestamp("2026-10-04T21:00:00Z")
INVALID = np.iinfo(np.int64).min  # pandas NaT in the cached int64 release arrays
HOUR_NS = 3_600_000_000_000
# Fingrid physical measurements that only the extended Gold layer carries.
FINGRID_EXTRA_SERIES = ("fi_wind_actual", "fi_wind_rt", "fi_wind_capacity_fc",
                        "fi_hydro_rt", "fi_nuclear_rt")
# Causal physical differentials: name, minuend, subtrahend, unit.
NEURAL_DIFF_SPECS = (("fi_wind_err", "fi_wind_actual", "fi_wind_fc_da", "MW"),
                     ("fi_load_err", "fi_consumption", "fi_consumption_fc_da", "MW"),
                     ("fi_mfrr_up_spread", "fi_mfrr_up_price", "fi_spot_price", "EUR/MWh"),
                     ("fi_mfrr_down_spread", "fi_mfrr_down_price", "fi_spot_price", "EUR/MWh"),
                     ("fi_afrr_up_spread", "fi_afrr_up_price", "fi_spot_price", "EUR/MWh"),
                     ("fi_afrr_down_spread", "fi_afrr_down_price", "fi_spot_price", "EUR/MWh"))
NET_LOAD_FORECASTS = ("fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da")
# Reserve label order: mfrr_up, mfrr_down, afrr_up, afrr_down. Even indices are
# up-regulation components, odd indices are down-regulation components.
RESERVE_PRICE_COLUMNS = ("fi_mfrr_up_price", "fi_mfrr_down_price",
                         "fi_afrr_up_price", "fi_afrr_down_price")
SCALER_TYPES = ("standard", "robust_asinh")
RAMP_QUARTERS = 4  # one hour at the Gold 15-minute resolution
RAMP_SPECS = (("fi_spot_price", "fi_spot_ramp_1h", "EUR/MWh"),
              ("fi_net_load_fc_da", "fi_net_load_ramp_1h", "MW"))


def _take(values, positions):
    """Gather with pandas reindex semantics: positions outside the table become NaN."""
    if positions.size and positions.min() >= 0 and positions.max() < len(values):
        return values[positions]
    result = np.full(positions.shape, np.nan, dtype=values.dtype)
    inside = (positions >= 0) & (positions < len(values))
    result[inside] = values[positions[inside]]
    return result


def _take_release(releases, positions):
    if positions.size and positions.min() >= 0 and positions.max() < len(releases):
        return releases[positions]
    result = np.full(positions.shape, INVALID, dtype=np.int64)
    inside = (positions >= 0) & (positions < len(releases))
    result[inside] = releases[positions[inside]]
    return result


def _ages(origin_ns, releases, mask):
    """Hours since publication, zero where the entry is masked; mirrors pandas total_seconds."""
    result = np.zeros(releases.shape, dtype=np.float64)
    known = releases != INVALID
    result[known] = (origin_ns[known] - releases[known]) / HOUR_NS
    return np.where(mask, np.maximum(result, 0.0), 0.0)


def _read_matrix(data, names, positions, origin_ns):
    """Gather [G, T] windows for every feature and keep per-row publication cutoffs."""
    values, masks, ages = [], [], []
    grid = np.broadcast_to(origin_ns[:, None], positions.shape)
    cutoffs = np.full(origin_ns.shape, INVALID, dtype=np.int64)
    for name in names:
        raw, releases = data["columns"][name]
        value = _take(raw, positions)
        release = _take_release(releases, positions)
        mask = np.isfinite(value) & (release != INVALID) & (release <= grid)
        # One transform definition per market: asinh links live in the scaler.
        values.append(np.where(mask, data["scaler"].transform(name, value), 0.0))
        masks.append(mask)
        ages.append(_ages(grid, release, mask))
        cutoffs = np.maximum(cutoffs, np.where(mask, release, INVALID).max(axis=1))
    return (np.stack(values, -1), np.stack(masks, -1), np.stack(ages, -1), cutoffs)


def _reserve_components(reserve, window, cutoff_ns, spots, scale):
    """Aligned reserve-spread labels and per-component availability masks.

    ``reserve`` holds cached ``(values, releases)`` pairs in ``RESERVE_PRICE_COLUMNS``
    order.  A component is available only when its source release is at or before
    the label cutoff and both the reserve price and the delivery spot are finite.
    """
    components, masks = [], []
    for index, (values, releases) in enumerate(reserve):
        release = _take_release(releases, window)
        published = (release != INVALID) & (release <= cutoff_ns)
        price = np.where(published, _take(values, window), np.nan)
        valid = published & np.isfinite(price) & np.isfinite(spots)
        side = 1.0 if index % 2 == 0 else -1.0
        spread = np.where(valid, np.maximum(0.0, side * (price - spots)) / scale, 0.0)
        components.append(spread)
        masks.append(valid)
    return np.stack(components, -1).astype(np.float32), np.stack(masks, -1)


def target_series(market):
    if market == "FI":
        return TargetSeries()
    prefix = f"esett_{market.lower()}"
    # Neighbor direction definitions have not been validated against Fingrid 369.
    return TargetSeries(prefix + "_imbalance_price", prefix + "_spot_price", None)


def _system_features(system_series):
    """Validate the selected system-balance observations and describe their channels."""
    names = tuple(system_series)
    if len(set(names)) != len(names) or not set(names) <= set(SYSTEM_BALANCE_SERIES):
        raise ValueError("System inputs must be a unique subset of the Fingrid system-balance series")
    return tuple(Feature(name, "fi_system_balance", "MW", "observed", name + SUFFIX) for name in names)


def _neural_features(gold, derived):
    """Physical measurement, causal differential, net-load, and ramp descriptors."""
    features = []
    for name in FINGRID_EXTRA_SERIES:
        if name in gold:
            features.append(Feature(name, "fi_physical", "MW", "observed", name + SUFFIX))

    def release_of(names):
        return gold[[name + SUFFIX for name in names]].max(axis=1, skipna=False)

    for name, minuend, subtrahend, unit in NEURAL_DIFF_SPECS:
        if minuend not in gold or subtrahend not in gold:
            continue
        derived[name] = gold[minuend] - gold[subtrahend]
        # The later publication decides availability; a diff is never earlier.
        derived[name + SUFFIX] = release_of((minuend, subtrahend))
        features.append(Feature(name, "fi_differential", unit, "observed", name + SUFFIX))
    if set(NET_LOAD_FORECASTS) <= set(gold):
        derived["fi_net_load_fc_da"] = (gold[NET_LOAD_FORECASTS[0]] - gold[NET_LOAD_FORECASTS[1]]
                                        - gold[NET_LOAD_FORECASTS[2]])
        derived["fi_net_load_fc_da" + SUFFIX] = release_of(NET_LOAD_FORECASTS)
        features.append(Feature("fi_net_load_fc_da", "fi_differential", "MW", "known",
                                "fi_net_load_fc_da" + SUFFIX))

    def column(name):
        return derived[name] if name in derived else gold[name]

    for name, ramp, unit in RAMP_SPECS:
        if name not in gold and name not in derived:
            continue
        derived[ramp] = column(name).diff(RAMP_QUARTERS)
        # The ramp reads t and t-4, so its release is the later of the two.
        derived[ramp + SUFFIX] = pd.concat(
            (column(name + SUFFIX), column(name + SUFFIX).shift(RAMP_QUARTERS)), axis=1
        ).max(axis=1, skipna=False)
        features.append(Feature(ramp, "fi_differential", unit, "known", ramp + SUFFIX))
    return features


def prepare_inputs(gold, weather_series, system_series=SYSTEM_BALANCE_SERIES, history_relabel=True,
                   include_neural_diffs=False):
    """Derive covariates in memory; all scenarios use the same regional references.

    ``history_relabel`` rebuilds pre-June 2026 FI imbalance prices under the
    current rule before any derived column exists, so spreads, scalers,
    history windows, labels and spikes all share that rule.  ``include_neural_diffs``
    adds the Fingrid physical measurements and their causal differentials.
    """
    gold = reprice_history(gold) if history_relabel else gold
    derived = {}
    features = list(official_features())
    for market in MARKET_NAMES:
        target = target_series(market)
        name = f"market_{market.lower()}_spread"
        derived[name] = gold[target.price] - gold[target.spot]
        derived[name + SUFFIX] = gold[[target.price + SUFFIX, target.spot + SUFFIX]].max(axis=1, skipna=False)
        features.append(Feature(name, "market_history", "EUR/MWh", "observed", name + SUFFIX))
        if target.spot != SPOT:
            features.append(Feature(target.spot, "market_spot", "EUR/MWh", "known", target.spot + SUFFIX))
    for name in weather_series:
        # FMI is a Finnish regional reference, not the sampled market's local weather.
        unit = "C" if name.endswith("_t2m") else "m/s"
        features.append(Feature(name, "fi_weather", unit, "observed", name + SUFFIX))
    features.extend(_system_features(system_series))
    if include_neural_diffs:
        features.extend(_neural_features(gold, derived))
    features.append(Feature("target_spot_price", "target", "EUR/MWh", "known", "target_spot_price" + SUFFIX))
    return pd.concat((gold, pd.DataFrame(derived, index=gold.index)), axis=1), tuple(features)


class MultiMarketBuilder:
    """Sample market-origin pairs lazily, with per-market training transforms."""

    def __init__(self, gold, markets, weather_series, history=720, horizon=40,
                 system_series=SYSTEM_BALANCE_SERIES, history_relabel=True,
                 include_neural_diffs=False, scaler_type="standard"):
        self.markets = tuple(markets)
        if not self.markets or len(set(self.markets)) != len(self.markets) or not set(self.markets) <= MARKET_IDS.keys():
            raise ValueError("Select unique supported target markets")
        self.weather_series = tuple(weather_series)
        if any(not name.startswith("fmi_") or not name.endswith(("_t2m", "_ws_10min")) for name in self.weather_series):
            raise ValueError("Weather inputs must be FMI temperature or wind-speed observations")
        self.system_series = tuple(system_series)
        self.history_relabel = bool(history_relabel)
        self.include_neural_diffs = bool(include_neural_diffs)
        if scaler_type not in SCALER_TYPES:
            raise ValueError(f"scaler_type must be one of {SCALER_TYPES}, got {scaler_type!r}")
        self.scaler_type = scaler_type
        # Auxiliary reserve-spread labels exist for the Finnish target only.
        if self.include_neural_diffs and self.markets != ("FI",):
            raise ValueError("Neural differentials and reserve-spread labels require the FI target set")
        self.gold, self.features = prepare_inputs(gold, self.weather_series, self.system_series,
                                                  self.history_relabel, self.include_neural_diffs)
        self.history, self.horizon = history, horizon
        if min(history, horizon) <= 0:
            raise ValueError("History and horizon must be positive")
        self.builders = {}
        self._arrays = None  # lazily built numpy window cache; see tensors()

    @classmethod
    def fit(cls, gold, start, cutoff, markets, history=720, horizon=40,
            weather_series=None, allow_pre_official=False, system_series=SYSTEM_BALANCE_SERIES,
            history_relabel=True, include_neural_diffs=False, scaler_type="standard"):
        if weather_series is None:
            weather_series = sorted(name for name in gold if name.startswith("fmi_") and name.endswith(("_t2m", "_ws_10min")))
        result = cls(gold, markets, weather_series, history, horizon, system_series, history_relabel,
                     include_neural_diffs, scaler_type)
        result.start, result.cutoff = utc(start), utc(cutoff)
        result.allow_pre_official = allow_pre_official
        scaler_class = StandardScaler if scaler_type == "standard" else RobustAsinhScaler
        for market in result.markets:
            view, target = result._view(market)
            scaler = scaler_class.fit(view, result.features, start, cutoff,
                                      allow_pre_official=allow_pre_official, target=target)
            result.builders[market] = BatchBuilder(view, result.features, scaler, history, horizon, target=target)
        return result

    def _view(self, market):
        target = target_series(market)
        anchors = pd.DataFrame({"target_spot_price": self.gold[target.spot],
                                "target_spot_price" + SUFFIX: self.gold[target.spot + SUFFIX]}, index=self.gold.index)
        view = pd.concat((self.gold, anchors), axis=1)
        return view, replace(target, spot="target_spot_price")

    @property
    def observed_channels(self):
        return sum(f.role == "observed" for f in self.features)

    @property
    def known_channels(self):
        return sum(f.role == "known" for f in self.features)

    def training_samples(self, start=None, end=None, label_cutoff=None):
        """Return metadata only; require released targets and known spot curves.

        Defaults select the training interval. Pass the official validation
        start/end and its label cutoff to enumerate validation origins without
        refitting the scaler.
        """
        start = self.start if start is None else utc(start)
        end = (self.cutoff - pd.Timedelta(hours=12)) if end is None else utc(end)
        label_cutoff = self.cutoff if label_cutoff is None else utc(label_cutoff)
        first = max(start, self.gold.index[0] + self.history * QUARTER)
        origins = pd.date_range(first.ceil("h"), end, freq="h")
        positions = self.gold.index.get_indexer(origins)
        future = positions[:, None] + np.arange(self.horizon)
        records = []
        def ns(values):
            return pd.DatetimeIndex(values).as_unit("ns").asi8
        for market, builder in self.builders.items():
            target, frame = builder.target, builder.gold
            valid = (np.isfinite(frame[target.price]) & np.isfinite(frame[target.spot])
                     & frame[target.price + SUFFIX].notna() & frame[target.spot + SUFFIX].notna())
            release = np.maximum(ns(frame[target.price + SUFFIX]), ns(frame[target.spot + SUFFIX]))
            in_bounds = (positions >= self.history) & (future[:, -1] < len(frame))
            selected = np.flatnonzero(in_bounds)
            delivery = future[selected]
            released = (valid.to_numpy() & (release <= label_cutoff.value))[delivery].all(axis=1)
            known_spot = (ns(frame[target.spot + SUFFIX])[delivery] <= ns(origins[selected])[:, None]).all(axis=1)
            past = positions[selected, None] - np.arange(1, self.history + 1)
            has_history = (valid.to_numpy()[past] & (release[past] <= ns(origins[selected])[:, None])).any(axis=1)
            for i in selected[released & known_spot & has_history]:
                records.append((market, origins[i]))
        return sorted(records, key=lambda item: (item[1], MARKET_IDS[item[0]]))

    def build(self, samples):
        batches = []
        for market, origin in samples:
            builder = self.builders[market]
            origin = utc(origin)
            # Slice before publication replay to avoid scanning the full history per sample.
            window = builder.gold.loc[origin - self.history * QUARTER:origin + (self.horizon - 1) * QUARTER]
            local = BatchBuilder(window, self.features, builder.scaler, self.history, self.horizon, target=builder.target)
            batch = local.build([origin])
            batches.append(replace(batch, series_ids=torch.tensor([MARKET_IDS[market]], dtype=torch.long)))
        return concatenate_batches(batches)

    def labels(self, samples):
        if self.include_neural_diffs:
            # The cached path owns the reserve-spread labels; keep one definition.
            return self._fast_labels(list(samples), self.cutoff, self.allow_pre_official)
        labels = []
        for market, value in samples:
            origin = utc(value)
            builder = self.builders[market]
            window = builder.gold.loc[origin:origin + (self.horizon - 1) * QUARTER]
            local = BatchBuilder(window, self.features, builder.scaler, self.history, self.horizon, target=builder.target)
            labels.append(local.labels([origin], self.cutoff, allow_pre_official=self.allow_pre_official))
        if not labels:
            raise ValueError("At least one sample is required")
        columns = {}
        for field in fields(Targets):
            values = [getattr(label, field.name) for label in labels]
            if any(value is None for value in values):
                if not all(value is None for value in values):
                    raise ValueError(f"Cannot mix present and absent {field.name} labels")
                columns[field.name] = None
            else:
                columns[field.name] = torch.cat(values)
        return Targets(**columns)

    def _column(self, name):
        values = self.gold[name].to_numpy(dtype=np.float32)
        releases = self.gold[name + SUFFIX].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        return name, values, releases

    def _cache(self):
        """Aligned numpy columns; one copy per shared feature, per-market target columns."""
        if self._arrays is not None:
            return self._arrays
        self._index_ns = self.gold.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        shared = {name: (values, releases)
                  for name, values, releases in (self._column(f.name) for f in self.features
                                                 if f.name != "target_spot_price")}
        scaled = {}
        # Raw reserve columns are only needed when reserve-spread labels are trained.
        reserve = ([self._column(name)[1:] for name in RESERVE_PRICE_COLUMNS]
                   if self.include_neural_diffs else None)
        for market in self.markets:
            base = target_series(market)
            builder = self.builders[market]
            columns = dict(shared)
            _, spot_values, spot_releases = self._column(base.spot)
            columns["target_spot_price"] = (spot_values, spot_releases)
            scaled[market] = {
                "columns": columns,
                "price": self._column(base.price)[1:],
                "spot": (spot_values, spot_releases),
                # Preserve strict +/-50 boundaries before the float32 value cache.
                "spikes": np.stack((self.gold[base.price] > self.gold[base.spot] + 50,
                                    self.gold[base.price] < self.gold[base.spot] - 50), axis=-1),
                "direction": self._column(base.direction)[1:] if base.direction else None,
                "observed": [f.name for f in builder.observed],
                "known": [f.name for f in builder.known],
                "observed_editable": np.array([f.continuous for f in builder.observed]),
                "known_editable": np.array([f.continuous and f.name != "target_spot_price"
                                            for f in builder.known]),
                "center": builder.scaler.center, "scale": builder.scaler.scale,
                "scaler": builder.scaler,
                "asinh": getattr(builder.scaler, "asinh", ()),
                "spread_center": float(builder.scaler.center["spread"]),
                "spread_scale": float(builder.scaler.scale["spread"]),
                "reserve": reserve,
                "series_id": MARKET_IDS[market],
            }
        self._arrays = scaled
        return scaled

    def _positions(self, origin):
        origin = utc(origin)
        if origin != origin.floor("h"):
            raise ValueError("Forecast origins must be full hours")
        origin_ns = int(origin.value)
        position = int(np.searchsorted(self._index_ns, origin_ns))
        if position >= len(self._index_ns) or int(self._index_ns[position]) != origin_ns:
            raise ValueError(f"Origin is not on the Gold grid: {origin}")
        return origin, origin_ns, position

    def _fast_batch(self, samples):
        cache = self._cache()
        count = len(samples)
        if not count:
            raise ValueError("At least one sample is required")
        parsed, origin_ns, positions = [], np.empty(count, np.int64), np.empty(count, np.int64)
        for index, (market, origin) in enumerate(samples):
            if market not in cache:
                raise ValueError(f"Unknown market: {market}")
            parsed_origin, sample_ns, position = self._positions(origin)
            parsed.append(parsed_origin)
            origin_ns[index], positions[index] = sample_ns, position
        history = positions[:, None] - np.arange(self.history, 0, -1)[None, :]
        delivery = positions[:, None] + np.arange(self.horizon)[None, :]
        observed = np.zeros((count, self.history, self.observed_channels), dtype=np.float32)
        observed_mask = np.zeros((count, self.history, self.observed_channels), dtype=bool)
        observed_age = np.zeros((count, self.history, self.observed_channels), dtype=np.float32)
        known_history = np.zeros((count, self.history, self.known_channels), dtype=np.float32)
        known_history_mask = np.zeros((count, self.history, self.known_channels), dtype=bool)
        known_history_age = np.zeros((count, self.history, self.known_channels), dtype=np.float32)
        known_future = np.zeros((count, self.horizon, self.known_channels), dtype=np.float32)
        known_future_mask = np.zeros((count, self.horizon, self.known_channels), dtype=bool)
        known_future_age = np.zeros((count, self.horizon, self.known_channels), dtype=np.float32)
        target = np.zeros((count, self.history, 1), dtype=np.float32)
        target_mask = np.zeros((count, self.history, 1), dtype=bool)
        target_age = np.zeros((count, self.history, 1), dtype=np.float32)
        spot = np.full((count, self.horizon), np.nan, dtype=np.float32)
        cutoffs = np.full(count, INVALID, dtype=np.int64)
        series_ids = np.zeros(count, dtype=np.int64)
        spread_center = np.zeros(count, dtype=np.float32)
        spread_scale = np.zeros(count, dtype=np.float32)

        for market in self.markets:
            members = np.array([i for i, (name, _) in enumerate(samples) if name == market], dtype=np.int64)
            if not len(members):
                continue
            data = cache[market]
            history_positions = history[members]
            grid = np.broadcast_to(origin_ns[members, None], history_positions.shape)
            observed[members], observed_mask[members], observed_age[members], cutoff = _read_matrix(
                data, data["observed"], history_positions, origin_ns[members])
            known_history[members], known_history_mask[members], known_history_age[members], cutoff = _read_matrix(
                data, data["known"], history_positions, origin_ns[members])
            known_future[members], known_future_mask[members], known_future_age[members], cutoff_future = _read_matrix(
                data, data["known"], delivery[members], origin_ns[members])
            cutoffs[members] = np.maximum(cutoffs[members], np.maximum(cutoff, cutoff_future))

            price_values, price_releases = data["price"]
            spot_values, spot_releases = data["spot"]
            price_release = _take_release(price_releases, history_positions)
            spot_release = _take_release(spot_releases, history_positions)
            prices = np.where((price_release != INVALID) & (price_release <= grid),
                              _take(price_values, history_positions), np.nan)
            spots = np.where((spot_release != INVALID) & (spot_release <= grid),
                             _take(spot_values, history_positions), np.nan)
            raw_spread = prices - spots
            mask = np.isfinite(raw_spread)
            target[members, :, 0] = np.where(
                mask, (raw_spread - data["spread_center"]) / data["spread_scale"], 0.0)
            target_mask[members, :, 0] = mask
            release = np.maximum(price_release, spot_release)
            target_age[members, :, 0] = _ages(grid, release, mask)
            cutoffs[members] = np.maximum(cutoffs[members], np.where(mask, release, INVALID).max(axis=1))

            delivery_release = _take_release(spot_releases, delivery[members])
            delivery_grid = np.broadcast_to(origin_ns[members, None], delivery[members].shape)
            delivery_spot = np.where((delivery_release != INVALID) & (delivery_release <= delivery_grid),
                                     _take(spot_values, delivery[members]), np.nan)
            if not np.isfinite(delivery_spot).all():
                raise ValueError("The complete delivery spot anchor must already be published")
            spot[members] = delivery_spot
            cutoffs[members] = np.maximum(
                cutoffs[members], np.where(delivery_release != INVALID, delivery_release, INVALID).max(axis=1))
            spread_center[members] = data["spread_center"]
            spread_scale[members] = data["spread_scale"]
            series_ids[members] = data["series_id"]

        if (cutoffs == INVALID).any():
            raise ValueError("No published source timestamps for some samples")
        first = cache[self.markets[0]]
        return ForecastBatch(
            target_history=torch.as_tensor(target), observed_history=torch.as_tensor(observed),
            known_history=torch.as_tensor(known_history), known_future=torch.as_tensor(known_future),
            target_mask=torch.as_tensor(target_mask), observed_mask=torch.as_tensor(observed_mask),
            known_history_mask=torch.as_tensor(known_history_mask),
            known_future_mask=torch.as_tensor(known_future_mask),
            target_age=torch.as_tensor(target_age), observed_age=torch.as_tensor(observed_age),
            known_history_age=torch.as_tensor(known_history_age),
            known_future_age=torch.as_tensor(known_future_age),
            spot=torch.as_tensor(spot),
            spread_center=torch.as_tensor(spread_center)[:, None],
            spread_scale=torch.as_tensor(spread_scale)[:, None],
            origins=tuple(value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in parsed),
            data_cutoffs=tuple(pd.Timestamp(int(value), unit="ns", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                               for value in cutoffs),
            observed_editable=torch.as_tensor(first["observed_editable"]),
            known_editable=torch.as_tensor(first["known_editable"]),
            series_ids=torch.as_tensor(series_ids))

    def _fast_labels(self, samples, published_by, allow_pre_official=False):
        cutoff = utc(published_by)
        cutoff_ns = int(cutoff.value)
        if cutoff > TRAIN_END:
            raise ValueError("Training/calibration must not read October test labels")
        cache = self._cache()
        count = len(samples)
        if not count:
            raise ValueError("At least one sample is required")
        origin_ns = np.empty(count, np.int64)
        positions = np.empty(count, np.int64)
        for index, (market, value) in enumerate(samples):
            if market not in cache:
                raise ValueError(f"Unknown market: {market}")
            origin, sample_ns, position = self._positions(value)
            if not allow_pre_official and origin < OFFICIAL_START:
                raise ValueError("Pre-official history requires explicit eligibility opt-in")
            origin_ns[index], positions[index] = sample_ns, position
        delivery = positions[:, None] + np.arange(self.horizon)[None, :]
        last = delivery[:, -1]
        if int(last.max()) >= len(self._index_ns) or int(self._index_ns[last].max()) >= int(TRAIN_END.value):
            raise ValueError("October test deliveries cannot be training labels")

        spread = np.zeros((count, self.horizon), dtype=np.float32)
        direction = np.zeros((count, self.horizon), dtype=np.int64)
        price_mask = np.zeros((count, self.horizon), dtype=bool)
        direction_mask = np.zeros((count, self.horizon), dtype=bool)
        spikes = np.zeros((count, self.horizon, 2), dtype=bool)
        reserve = self.include_neural_diffs
        components = np.zeros((count, self.horizon, len(RESERVE_PRICE_COLUMNS)), dtype=np.float32)
        reserve_mask = np.zeros((count, self.horizon, len(RESERVE_PRICE_COLUMNS)), dtype=bool)
        for market in self.markets:
            members = np.array([i for i, (name, _) in enumerate(samples) if name == market], dtype=np.int64)
            if not len(members):
                continue
            data = cache[market]
            window = delivery[members]
            price_values, price_releases = data["price"]
            spot_values, spot_releases = data["spot"]
            price_release = _take_release(price_releases, window)
            spot_release = _take_release(spot_releases, window)
            prices = np.where((price_release != INVALID) & (price_release <= cutoff_ns),
                              _take(price_values, window), np.nan)
            spots = np.where((spot_release != INVALID) & (spot_release <= cutoff_ns),
                             _take(spot_values, window), np.nan)
            raw = prices - spots
            spread[members] = np.nan_to_num(
                (raw - data["spread_center"]) / data["spread_scale"], nan=0)
            price_mask[members] = np.isfinite(raw)
            spikes[members] = data["spikes"][window] & np.isfinite(raw)[..., None]
            if reserve:
                components[members], reserve_mask[members] = _reserve_components(
                    data["reserve"], window, cutoff_ns, spots, data["spread_scale"])
            if data["direction"] is None:
                continue
            values, releases = data["direction"]
            release = _take_release(releases, window)
            available = np.where((release != INVALID) & (release <= cutoff_ns),
                                 _take(values, window), np.nan)
            direction[members] = np.where(available == 1, 0,
                                          np.where(available == -1, 1,
                                                   np.where(available == 0, 2, 0)))
            direction_mask[members] = np.isin(available, (1.0, -1.0, 0.0))
        if not reserve:
            return Targets(
                torch.as_tensor(spread), torch.as_tensor(direction),
                torch.as_tensor(price_mask), torch.as_tensor(direction_mask), torch.as_tensor(spikes))
        return Targets(
            torch.as_tensor(spread), torch.as_tensor(direction),
            torch.as_tensor(price_mask), torch.as_tensor(direction_mask), torch.as_tensor(spikes),
            up_spread=torch.as_tensor(components[..., 0]),
            down_spread=torch.as_tensor(components[..., 1]),
            afrr_up_spread=torch.as_tensor(components[..., 2]),
            afrr_down_spread=torch.as_tensor(components[..., 3]),
            reserve_mask=torch.as_tensor(reserve_mask))

    def tensors(self, samples, published_by=None, chunk_size=128, workers=2):
        """Cached batch and labels; equivalent to build()+labels()."""
        samples = list(samples)
        if not samples:
            raise ValueError("At least one sample is required")
        published_by = self.cutoff if published_by is None else published_by
        labels = self._fast_labels(samples, published_by, self.allow_pre_official)
        if chunk_size and len(samples) > chunk_size:
            chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    parts = list(pool.map(self._fast_batch, chunks))
            else:
                parts = [self._fast_batch(chunk) for chunk in chunks]
            batch = concatenate_batches(parts)
        else:
            batch = self._fast_batch(samples)
        return batch, labels

    def iter_batches(self, samples, batch_size=16):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        iterator = iter(samples)
        while chunk := list(islice(iterator, batch_size)):
            yield self.tensors(chunk)

    def sample_training_pairs(self, count, seed, fi_probability=0.5, samples=None):
        """Sample with replacement; this changes optimization weight, not data volume."""
        if count <= 0 or not 0 < fi_probability <= 1 or "FI" not in self.markets:
            raise ValueError("Use a positive count, FI targets, and FI probability in (0, 1]")
        pools = {market: [] for market in self.markets}
        for market, origin in (self.training_samples() if samples is None else samples):
            pools[market].append(origin)
        if any(not pool for pool in pools.values()):
            raise ValueError("Every selected market needs training windows")
        rng = np.random.default_rng(seed)
        probabilities = np.full(len(self.markets), (1 - fi_probability) / max(1, len(self.markets) - 1))
        probabilities[self.markets.index("FI")] = fi_probability if len(self.markets) > 1 else 1
        for _ in range(count):
            market = self.markets[int(rng.choice(len(self.markets), p=probabilities))]
            yield market, pools[market][int(rng.integers(len(pools[market])))]

    def state_dict(self):
        return {"kind": "multi_market", "markets": list(self.markets), "market_ids": MARKET_IDS,
                "weather_series": list(self.weather_series), "system_series": list(self.system_series),
                "history": self.history, "horizon": self.horizon,
                "features": [asdict(feature) for feature in self.features],
                "start": self.start.isoformat(), "cutoff": self.cutoff.isoformat(),
                "allow_pre_official": self.allow_pre_official,
                "history_relabel": self.history_relabel,
                "include_neural_diffs": self.include_neural_diffs,
                "scaler_type": self.scaler_type,
                "scalers": {market: builder.scaler.state_dict() for market, builder in self.builders.items()}}

    @classmethod
    def from_state_dict(cls, gold, state):
        if state["market_ids"] != MARKET_IDS:
            raise ValueError("Checkpoint market IDs differ")
        # Older manifests have no system_series field; replay them without the new inputs.
        # They also predate re-pricing, so a missing flag means the published old-rule history.
        result = cls(gold, state["markets"], state["weather_series"], state["history"], state["horizon"],
                     system_series=state.get("system_series", ()),
                     history_relabel=state.get("history_relabel", False),
                     include_neural_diffs=state.get("include_neural_diffs", False),
                     scaler_type=state.get("scaler_type", "standard"))
        if [asdict(feature) for feature in result.features] != state["features"]:
            raise ValueError("Checkpoint market features differ")
        result.start, result.cutoff = utc(state["start"]), utc(state["cutoff"])
        result.allow_pre_official = state["allow_pre_official"]
        for market in result.markets:
            view, target = result._view(market)
            scaler_class = StandardScaler if result.scaler_type == "standard" else RobustAsinhScaler
            result.builders[market] = BatchBuilder(view, result.features,
                                                   scaler_class(**state["scalers"][market]),
                                                  result.history, result.horizon, target=target)
        return result
