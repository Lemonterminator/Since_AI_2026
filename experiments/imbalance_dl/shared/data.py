"""Gold-to-tensor conversion with per-origin publication replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contracts import ForecastBatch, Targets

ROOT = Path(__file__).resolve().parents[3]
SUFFIX = "__first_usable_origin_utc"
REGIME_START = pd.Timestamp("2026-05-31T21:00:00Z")  # current imbalance pricing rule
PRICE, SPOT, DIRECTION = "fi_imbalance_price", "fi_spot_price", "fi_mfrr_direction"
QUARTER = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class TargetSeries:
    price: str = PRICE
    spot: str = SPOT
    direction: str | None = DIRECTION


def repository_module(relative_path, name):
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Separately supplied challenge tooling required: {relative_path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotations in official modules resolve through sys.modules.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gold(path: str | Path | None = None):
    """Read a locally supplied wide table. No datasets are bundled."""
    path = Path(path).expanduser() if path is not None else ROOT / "data/gold_ext/out/full_ext.parquet"
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix == ".csv":
        table = pd.read_csv(path, float_precision="round_trip")
    else:
        raise ValueError("Expected a local CSV or Parquet file")
    for col in ["timestamp_utc", *(c for c in table if c.endswith(SUFFIX))]:
        table[col] = pd.to_datetime(table[col], utc=True).astype("datetime64[ns, UTC]")
    return table.set_index("timestamp_utc")


def rows_available_at(gold, origin_utc):
    """Return only values whose series-specific publication time has passed."""
    origin = pd.Timestamp(origin_utc)
    if origin.tzinfo is None:
        raise ValueError("Forecast origin must be timezone-aware")
    origin = origin.tz_convert("UTC")
    if origin != origin.floor("h"):
        raise ValueError("Forecast origin must be a full UTC hour")
    columns = [c[:-len(SUFFIX)] for c in gold if c.endswith(SUFFIX)]
    return pd.DataFrame({name: gold[name].where(gold[name + SUFFIX].le(origin))
                         for name in columns}, index=gold.index)


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    unit: str
    role: str  # observed, known, or event (origin/step snapshots).
    availability_rule: str
    continuous: bool = True
    last_known: bool = False

    def __post_init__(self):
        if self.role not in {"observed", "known", "event"}:
            raise ValueError(f"Invalid temporal role: {self.role}")
        if self.last_known and self.role != "observed":
            raise ValueError("Last-known carry is for explicitly selected historical observations only")
        if self.name in {PRICE, "data_split"} or self.name.endswith(SUFFIX):
            raise ValueError("Labels and availability metadata cannot be covariates")


def official_features(schema_path=ROOT / "data/gold/schema.json"):
    schema = json.loads(Path(schema_path).read_text())["series"]
    known = {SPOT, "fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da",
             "fi_ee_capacity_da", "ee_fi_capacity_da"}
    return tuple(Feature(name, "official", metadata["unit"],
                         "known" if name in known or name.startswith("cal_") else "observed",
                         metadata["availability_column"],
                         not name.startswith("cal_") and name != DIRECTION)
                 for name, metadata in schema.items() if name != PRICE)


@dataclass
class _TrainScaler:
    """Center/scale transform fitted on published training observations only."""

    center: dict[str, float]
    scale: dict[str, float]
    fitted_cutoff: str

    @classmethod
    def fit(cls, gold, features, start, cutoff, events=None, allow_pre_official=False,
            target=TargetSeries()):
        """Fit preprocessing only. Do not run a forecasting model or use test rows."""
        start, cutoff = utc(start), utc(cutoff)
        if start >= cutoff:
            raise ValueError("Transform fitting needs a nonempty chronological interval")
        if not allow_pre_official and start < REGIME_START:
            raise ValueError("Pre-official transforms require explicit eligibility opt-in")
        if cutoff > pd.Timestamp("2026-10-04T21:00:00Z"):
            raise ValueError("Transforms must not fit on the October test interval")
        available = rows_available_at(gold, cutoff)
        window = available.loc[(available.index >= start) & (available.index < cutoff)]
        values = {f.name: window[f.name].to_numpy() for f in features if f.role != "event"}
        for feature in features:
            if feature.role == "event":
                if events is None:
                    raise ValueError("Event transforms require origin-safe event snapshots")
                origins = pd.to_datetime(events.forecast_origin_utc, utc=True)
                selected = events.loc[(origins >= start) & (origins < cutoff), feature.name]
                values[feature.name] = selected.to_numpy(float)
        values["spread"] = (window[target.price] - window[target.spot]).to_numpy()
        center, scale = {}, {}
        for name, column in values.items():
            valid = column[np.isfinite(column)]
            if not len(valid):
                raise ValueError(f"No training observations for transform: {name}")
            center[name], scale[name] = cls._statistics(valid)
        return cls(center, scale, cutoff.isoformat())

    @staticmethod
    def _statistics(valid):
        """Compute one center and one positive scale from finite observations."""
        raise NotImplementedError("Each scaler defines its training statistics")

    def transform(self, name, values):
        return (values - self.center[name]) / self.scale[name]

    def state_dict(self):
        return asdict(self)


@dataclass
class RobustScaler(_TrainScaler):
    """Median center and interquartile scale."""

    @staticmethod
    def _statistics(valid):
        q25, median, q75 = np.quantile(valid, [.25, .5, .75])
        return float(median), float(q75 - q25) or 1.0


@dataclass
class StandardScaler(_TrainScaler):
    """Mean center and population standard deviation scale."""

    @staticmethod
    def _statistics(valid):
        return float(np.mean(valid)), float(np.std(valid, ddof=0)) or 1.0


@dataclass
class RobustAsinhScaler(RobustScaler):
    """Median/IQR statistics fitted on the delivery regime, with an asinh link.

    Fitting starts at :data:`REGIME_START` so pre-June 2026 spike rows cannot
    compress delivery-regime variation into a narrow normalized band.  ``asinh``
    holds the continuous covariate names that get the link; the spread target
    stays affine, so the exporter's inverse transform remains exact.
    """

    asinh: tuple[str, ...] = ()

    @classmethod
    def fit(cls, gold, features, start, cutoff, events=None, allow_pre_official=False,
            target=TargetSeries()):
        start = max(utc(start), REGIME_START)
        fitted = super().fit(gold, features, start, cutoff, events, allow_pre_official, target)
        return replace(fitted, asinh=tuple(feature.name for feature in features
                                           if feature.continuous and feature.role != "event"))

    def transform(self, name, values):
        scaled = super().transform(name, values)
        return np.arcsinh(scaled) if name in self.asinh else scaled


def utc(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("A timestamp must include its UTC offset")
    return timestamp.tz_convert("UTC")


class BatchBuilder:
    def __init__(self, gold, features, scaler, history=96, horizon=40, events=None,
                 target=TargetSeries()):
        if not gold.index.is_unique or not gold.index.is_monotonic_increasing:
            raise ValueError("Gold timestamps must be unique and sorted")
        self.gold, self.features, self.scaler = gold, tuple(features), scaler
        self.history, self.horizon = history, horizon
        self.target = target
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        self.observed = [f for f in self.features if f.role == "observed"]
        self.known = [f for f in self.features if f.role != "observed"]
        self.events = None
        if events is not None:
            events = events.copy()
            events["forecast_origin_utc"] = pd.to_datetime(events.forecast_origin_utc, utc=True)
            self.events = events.set_index(["forecast_origin_utc", "step"])
            if not self.events.index.is_unique:
                raise ValueError("Event origin/step keys must be unique")
        if any(f.role == "event" for f in self.features) and self.events is None:
            raise ValueError("Event features require the version-safe nuclear_asof table")
        for feature in self.features:
            if feature.role != "event":
                if feature.name not in gold or feature.name + SUFFIX not in gold:
                    raise ValueError(f"Missing Gold series/availability: {feature.name}")
            if feature.name not in scaler.center or feature.name not in scaler.scale:
                raise ValueError(f"Missing fold-fitted transform: {feature.name}")
        if any(not np.isfinite(x) or x <= 0 for x in scaler.scale.values()):
            raise ValueError("Scaler scales must be finite and positive")

    def _read(self, available, origin, index, feature, historical):
        if feature.role == "event":
            # Historical events use the snapshot at the corresponding historical origin.
            snapshots = index.floor("h") if historical else pd.DatetimeIndex([origin] * len(index))
            steps = ((index - snapshots) / QUARTER).astype(int) + 1
            keys = pd.MultiIndex.from_arrays([snapshots, steps])
            values = self.events[feature.name].reindex(keys).to_numpy(float)
            release = pd.Series(snapshots, index=index)
        else:
            series = available[feature.name]
            if feature.last_known:
                valid = series.dropna()
                source_times = pd.Series(valid.index, index=valid.index).reindex(index, method="ffill")
                values = valid.reindex(index, method="ffill").to_numpy(float)
                release = self.gold[feature.name + SUFFIX].reindex(pd.DatetimeIndex(source_times))
                release.index = index
            else:
                values = series.reindex(index).to_numpy(float)
                release = self.gold[feature.name + SUFFIX].reindex(index)
        mask = np.isfinite(values) & release.notna().to_numpy() & release.le(origin).to_numpy()
        normalized = self.scaler.transform(feature.name, values)
        ages = ((origin - pd.DatetimeIndex(release)).total_seconds() / 3600).to_numpy()
        return np.where(mask, normalized, 0), mask, np.where(mask, np.maximum(ages, 0), 0), release[mask]

    def build(self, origins):
        batches = []
        for value in origins:
            origin = utc(value)
            if origin != origin.floor("h"):
                raise ValueError("Forecast origins must be full hours")
            available = rows_available_at(self.gold, origin)
            history = pd.date_range(origin - self.history * QUARTER, periods=self.history, freq="15min")
            delivery = pd.date_range(origin, periods=self.horizon, freq="15min")
            cutoffs = []

            def group(features, index, historical):
                outputs = [self._read(available, origin, index, f, historical) for f in features]
                for output in outputs:
                    cutoffs.extend(output[3].tolist())
                return tuple(np.stack([o[i] for o in outputs], -1) if outputs
                             else np.empty((len(index), 0), dtype=bool if i == 1 else float)
                             for i in range(3))

            observed, observed_mask, observed_age = group(self.observed, history, True)
            kh, kh_mask, kh_age = group(self.known, history, True)
            kf, kf_mask, kf_age = group(self.known, delivery, False)
            raw_spread = (available[self.target.price] - available[self.target.spot]).reindex(history).to_numpy(float)
            target_mask = np.isfinite(raw_spread)
            target = np.where(target_mask, self.scaler.transform("spread", raw_spread), 0)[:, None]
            target_release = self.gold.reindex(history)[[self.target.price + SUFFIX, self.target.spot + SUFFIX]].max(axis=1)
            target_age = np.where(target_mask, np.maximum(
                ((origin - pd.DatetimeIndex(target_release)).total_seconds() / 3600).to_numpy(), 0), 0)[:, None]
            cutoffs.extend(target_release[target_mask].tolist())
            spot = available[self.target.spot].reindex(delivery).to_numpy(float)
            if not np.isfinite(spot).all():
                raise ValueError("The complete delivery spot anchor must already be published")
            cutoffs.extend(self.gold[self.target.spot + SUFFIX].reindex(delivery).tolist())
            def tensor(array, boolean=False):
                return torch.as_tensor(np.array(array), dtype=torch.bool if boolean else torch.float32).unsqueeze(0)
            batches.append(ForecastBatch(
                target_history=tensor(target), observed_history=tensor(observed), known_history=tensor(kh), known_future=tensor(kf),
                target_mask=tensor(target_mask[:, None], True), observed_mask=tensor(observed_mask, True),
                known_history_mask=tensor(kh_mask, True), known_future_mask=tensor(kf_mask, True),
                target_age=tensor(target_age), observed_age=tensor(observed_age), known_history_age=tensor(kh_age), known_future_age=tensor(kf_age),
                spot=tensor(spot), spread_center=tensor([self.scaler.center["spread"]]), spread_scale=tensor([self.scaler.scale["spread"]]),
                origins=(origin.strftime("%Y-%m-%dT%H:%M:%SZ"),),
                data_cutoffs=(max(cutoffs).strftime("%Y-%m-%dT%H:%M:%SZ"),),
                observed_editable=torch.tensor([f.continuous for f in self.observed]),
                known_editable=torch.tensor([f.continuous and f.name != self.target.spot for f in self.known])))
        return concatenate_batches(batches)

    def labels(self, origins, published_by, allow_pre_official=False):
        """Read training labels only when published by the explicit stage cutoff."""
        cutoff = utc(published_by)
        if cutoff > pd.Timestamp("2026-10-04T21:00:00Z"):
            raise ValueError("Training/calibration must not read October test labels")
        available = rows_available_at(self.gold, cutoff)
        spreads, directions, price_masks, direction_masks, spikes = [], [], [], [], []
        for value in origins:
            origin = utc(value)
            delivery = pd.date_range(origin, periods=self.horizon, freq="15min")
            if delivery[-1] >= pd.Timestamp("2026-10-04T21:00:00Z"):
                raise ValueError("October test deliveries cannot be training labels")
            if not allow_pre_official and origin < pd.Timestamp("2026-05-31T21:00:00Z"):
                raise ValueError("Pre-official history requires explicit eligibility opt-in")
            prices = available[self.target.price].reindex(delivery).to_numpy(float)
            spots = available[self.target.spot].reindex(delivery).to_numpy(float)
            raw = prices - spots
            direction = (available[self.target.direction].reindex(delivery).map({1: 0, -1: 1, 0: 2}).to_numpy(float)
                         if self.target.direction else np.full(self.horizon, np.nan))
            spreads.append(np.nan_to_num(self.scaler.transform("spread", raw), nan=0))
            directions.append(np.nan_to_num(direction, nan=0))
            price_masks.append(np.isfinite(raw))
            direction_masks.append(np.isfinite(direction))
            spikes.append(np.stack((prices > spots + 50, prices < spots - 50), axis=-1))
        return Targets(torch.tensor(np.array(spreads), dtype=torch.float32), torch.tensor(np.array(directions), dtype=torch.long),
                       torch.tensor(np.array(price_masks)), torch.tensor(np.array(direction_masks)),
                       torch.tensor(np.array(spikes), dtype=torch.bool))


def concatenate_batches(batches):
    from dataclasses import fields
    if not batches:
        raise ValueError("At least one origin is required")
    result = {}
    for field in fields(ForecastBatch):
        values = [getattr(batch, field.name) for batch in batches]
        if field.name.endswith("_editable"):
            result[field.name] = values[0]
        elif values[0] is None:
            if any(value is not None for value in values):
                raise ValueError(f"Cannot mix present and absent {field.name}")
            result[field.name] = None
        elif isinstance(values[0], torch.Tensor):
            result[field.name] = torch.cat(values)
        else:
            result[field.name] = sum(values, ())
    return ForecastBatch(**result)
