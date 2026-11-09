"""GPU-side construction of the cached multi-market batches.

The CPU path in ``MultiMarketBuilder`` replays publication timestamps per
window; at large batches that costs seconds while the forward/backward costs
milliseconds.  This builder puts the aligned float32 value/release tables on
the GPU once and gathers windows there, so the trainer spends its time on the
model instead of NumPy.
"""
import numpy as np
import pandas as pd
import torch

from .contracts import ForecastBatch, Targets
from .data import utc
from .markets import (HOUR_NS, INVALID, MARKET_IDS, OFFICIAL_START, QUARTER,
                     RESERVE_PRICE_COLUMNS, TRAIN_END)

HOUR_NS_INT = int(HOUR_NS)


class GpuTensorBuilder:
    """Build ``ForecastBatch`` and ``Targets`` tensors on one CUDA device."""

    def __init__(self, builder, device):
        self.builder = builder
        self.device = torch.device(device)
        self.cache = builder._cache()
        self.index = builder._index_ns
        self.history = builder.history
        self.horizon = builder.horizon
        self.observed_channels = builder.observed_channels
        self.known_channels = builder.known_channels
        self.cutoff = builder.cutoff
        self.allow_pre_official = builder.allow_pre_official
        self.tables = {market: self._table(data) for market, data in self.cache.items()}

    def _matrix(self, data, names):
        values = np.stack([data["columns"][name][0] for name in names], axis=0)
        releases = np.stack([data["columns"][name][1] for name in names], axis=0)
        asinh = torch.as_tensor([name in data["asinh"] for name in names],
                                dtype=torch.bool, device=self.device)
        return (torch.as_tensor(values, dtype=torch.float32, device=self.device),
                torch.as_tensor(releases, dtype=torch.int64, device=self.device), asinh)

    def _stats(self, data, names):
        center = torch.as_tensor([data["center"][name] for name in names],
                                 dtype=torch.float32, device=self.device)
        scale = torch.as_tensor([data["scale"][name] for name in names],
                                dtype=torch.float32, device=self.device)
        return center[:, None, None], scale[:, None, None]

    def _column(self, pair):
        values, releases = pair
        return (torch.as_tensor(values, dtype=torch.float32, device=self.device),
                torch.as_tensor(releases, dtype=torch.int64, device=self.device))

    def _table(self, data):
        observed_values, observed_releases, observed_asinh = self._matrix(data, data["observed"])
        known_values, known_releases, known_asinh = self._matrix(data, data["known"])
        return {
            "observed": (observed_values, observed_releases, *self._stats(data, data["observed"]),
                         observed_asinh),
            "known": (known_values, known_releases, *self._stats(data, data["known"]), known_asinh),
            "price": self._column(data["price"]),
            "spot": self._column(data["spot"]),
            "spikes": torch.as_tensor(data["spikes"], dtype=torch.bool, device=self.device),
            "direction": self._column(data["direction"]) if data["direction"] else None,
            "reserve": (None if data["reserve"] is None else
                        tuple(self._column(pair) for pair in data["reserve"])),
            "spread_center": float(data["spread_center"]),
            "spread_scale": float(data["spread_scale"]),
            "observed_editable": torch.as_tensor(data["observed_editable"], device=self.device),
            "known_editable": torch.as_tensor(data["known_editable"], device=self.device),
            "series_id": int(data["series_id"]),
        }

    def _prepare(self, samples):
        parsed = []
        for _, origin in samples:
            value = utc(origin) if not isinstance(origin, pd.Timestamp) else origin.tz_convert("UTC")
            if value != value.floor("h"):
                raise ValueError("Forecast origins must be full hours")
            parsed.append(value)
        origin_ns = np.fromiter((value.value for value in parsed), dtype=np.int64, count=len(parsed))
        positions = np.searchsorted(self.index, origin_ns)
        if positions.max(initial=-1) >= len(self.index) or np.any(self.index[positions] != origin_ns):
            raise ValueError("Origin is not on the Gold grid")
        return (parsed,
                torch.as_tensor(origin_ns, device=self.device),
                torch.as_tensor(positions, device=self.device),
                np.array([MARKET_IDS[market] for market, _ in samples], dtype=np.int64))

    @staticmethod
    def _read(table, positions, origin_ns):
        values, releases, center, scale, asinh = table
        grid = origin_ns[None, :, None]
        value = values[:, positions]
        release = releases[:, positions]
        mask = torch.isfinite(value) & (release != INVALID) & (release <= grid)
        normalized = (value - center) / scale
        normalized = torch.where(asinh[:, None, None], torch.asinh(normalized), normalized)
        out = torch.where(mask, normalized, torch.zeros((), device=value.device))
        stamp = torch.where(mask, release, grid)
        ages = torch.clamp((grid - stamp).float() / HOUR_NS_INT, min=0.0)
        ages = torch.where(mask, ages, torch.zeros((), device=value.device))
        return out.permute(1, 2, 0), mask.permute(1, 2, 0), ages.permute(1, 2, 0)

    def _batch(self, parsed, origin_ns, positions, market_ids):
        count = len(parsed)
        history = positions[:, None] - torch.arange(self.history, 0, -1, device=self.device)[None, :]
        delivery = positions[:, None] + torch.arange(self.horizon, device=self.device)[None, :]
        observed = torch.zeros((count, self.history, self.observed_channels), device=self.device)
        observed_mask = torch.zeros_like(observed, dtype=torch.bool)
        observed_age = torch.zeros_like(observed)
        known_history = torch.zeros((count, self.history, self.known_channels), device=self.device)
        known_history_mask = torch.zeros_like(known_history, dtype=torch.bool)
        known_history_age = torch.zeros_like(known_history)
        known_future = torch.zeros((count, self.horizon, self.known_channels), device=self.device)
        known_future_mask = torch.zeros_like(known_future, dtype=torch.bool)
        known_future_age = torch.zeros_like(known_future)
        target = torch.zeros((count, self.history, 1), device=self.device)
        target_mask = torch.zeros((count, self.history, 1), dtype=torch.bool, device=self.device)
        target_age = torch.zeros_like(target)
        spot = torch.full((count, self.horizon), torch.nan, device=self.device)
        spread_center = torch.zeros((count, 1), device=self.device)
        spread_scale = torch.zeros((count, 1), device=self.device)
        series_ids = torch.as_tensor(market_ids, dtype=torch.long, device=self.device)

        for market, table in self.tables.items():
            members_np = np.flatnonzero(market_ids == table["series_id"])
            if not len(members_np):
                continue
            members = torch.as_tensor(members_np, device=self.device)
            grid = origin_ns[members][:, None]
            observed[members], observed_mask[members], observed_age[members] = self._read(
                table["observed"], history[members], origin_ns[members])
            known_history[members], known_history_mask[members], known_history_age[members] = self._read(
                table["known"], history[members], origin_ns[members])
            known_future[members], known_future_mask[members], known_future_age[members] = self._read(
                table["known"], delivery[members], origin_ns[members])

            price_values, price_releases = table["price"]
            spot_values, spot_releases = table["spot"]
            price = price_values[history[members]]
            price_release = price_releases[history[members]]
            spot_history = spot_values[history[members]]
            spot_release = spot_releases[history[members]]
            mask = (torch.isfinite(price) & (price_release != INVALID) & (price_release <= grid)
                    & torch.isfinite(spot_history) & (spot_release != INVALID) & (spot_release <= grid))
            raw = torch.where(mask, price - spot_history, torch.zeros((), device=self.device))
            target[members] = torch.where(
                mask, (raw - table["spread_center"]) / table["spread_scale"],
                torch.zeros((), device=self.device))[..., None]
            target_mask[members] = mask[..., None]
            release = torch.maximum(price_release, spot_release)
            stamp = torch.where(mask, release, grid)
            target_age[members] = torch.clamp((grid - stamp).float() / HOUR_NS_INT, min=0.0)[..., None]

            delivery_release = spot_releases[delivery[members]]
            delivery_spot = torch.where((delivery_release != INVALID) & (delivery_release <= grid),
                                        spot_values[delivery[members]], torch.nan)
            if not torch.isfinite(delivery_spot).all():
                raise ValueError("The complete delivery spot anchor must already be published")
            spot[members] = delivery_spot
            spread_center[members] = table["spread_center"]
            spread_scale[members] = table["spread_scale"]

        collect = self.tables[self.builder.markets[0]]
        origins = tuple(value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in parsed)
        return ForecastBatch(
            target_history=target, observed_history=observed,
            known_history=known_history, known_future=known_future,
            target_mask=target_mask, observed_mask=observed_mask,
            known_history_mask=known_history_mask, known_future_mask=known_future_mask,
            target_age=target_age, observed_age=observed_age,
            known_history_age=known_history_age, known_future_age=known_future_age,
            spot=spot, spread_center=spread_center, spread_scale=spread_scale,
            origins=origins, data_cutoffs=origins,  # training-only upper bound
            observed_editable=collect["observed_editable"],
            known_editable=collect["known_editable"], series_ids=series_ids)

    def _labels(self, origin_ns, positions, market_ids, published_by):
        cutoff = utc(published_by)
        cutoff_ns = int(cutoff.value)
        if cutoff > TRAIN_END:
            raise ValueError("Training/calibration must not read October test labels")
        if not self.allow_pre_official and (origin_ns < int(OFFICIAL_START.value)).any():
            raise ValueError("Pre-official history requires explicit eligibility opt-in")
        count = len(origin_ns)
        delivery = positions[:, None] + torch.arange(self.horizon, device=self.device)[None, :]
        spread = torch.zeros((count, self.horizon), device=self.device)
        direction = torch.zeros((count, self.horizon), dtype=torch.long, device=self.device)
        price_mask = torch.zeros((count, self.horizon), dtype=torch.bool, device=self.device)
        direction_mask = torch.zeros_like(price_mask)
        spikes = torch.zeros((count, self.horizon, 2), dtype=torch.bool, device=self.device)
        reserve_width = len(RESERVE_PRICE_COLUMNS)  # mfrr_up, mfrr_down, afrr_up, afrr_down
        components = torch.zeros((count, self.horizon, reserve_width), device=self.device)
        reserve_mask = torch.zeros_like(components, dtype=torch.bool)
        cutoff_tensor = torch.as_tensor(cutoff_ns, device=self.device)
        for market, table in self.tables.items():
            members_np = np.flatnonzero(market_ids == table["series_id"])
            if not len(members_np):
                continue
            members = torch.as_tensor(members_np, device=self.device)
            window = delivery[members]
            price_values, price_releases = table["price"]
            spot_values, spot_releases = table["spot"]
            price_release = price_releases[window]
            spot_release = spot_releases[window]
            prices = torch.where((price_release != INVALID) & (price_release <= cutoff_tensor),
                                 price_values[window], torch.nan)
            spots = torch.where((spot_release != INVALID) & (spot_release <= cutoff_tensor),
                                spot_values[window], torch.nan)
            raw = prices - spots
            spread[members] = torch.nan_to_num(
                (raw - table["spread_center"]) / table["spread_scale"], nan=0.0)
            price_mask[members] = torch.isfinite(raw)
            spikes[members] = table["spikes"][window] & torch.isfinite(raw)[..., None]
            if table["reserve"] is not None:
                for index, (values, releases) in enumerate(table["reserve"]):
                    release = releases[window]
                    published = torch.where((release != INVALID) & (release <= cutoff_tensor),
                                            values[window], torch.nan)
                    valid = torch.isfinite(published) & torch.isfinite(spots)
                    side = 1.0 if index % 2 == 0 else -1.0
                    components[members, :, index] = torch.where(
                        valid, torch.clamp(side * (published - spots), min=0.0)
                        / table["spread_scale"], torch.zeros((), device=self.device))
                    reserve_mask[members, :, index] = valid
            if table["direction"] is None:
                continue
            values, releases = table["direction"]
            release = releases[window]
            available = torch.where((release != INVALID) & (release <= cutoff_tensor),
                                    values[window], torch.nan)
            direction[members] = torch.where(
                available == 1, 0, torch.where(available == -1, 1,
                                               torch.where(available == 0, 2, 0)))
            direction_mask[members] = ((available == 1) | (available == -1) | (available == 0))
        if not self.builder.include_neural_diffs:
            return Targets(spread, direction, price_mask, direction_mask, spikes)
        return Targets(spread, direction, price_mask, direction_mask, spikes,
                       up_spread=components[..., 0], down_spread=components[..., 1],
                       afrr_up_spread=components[..., 2], afrr_down_spread=components[..., 3],
                       reserve_mask=reserve_mask)

    def tensors(self, samples, published_by=None):
        samples = list(samples)
        if not samples:
            raise ValueError("At least one sample is required")
        parsed, origin_ns, positions, market_ids = self._prepare(samples)
        published_by = self.cutoff if published_by is None else published_by
        return (self._batch(parsed, origin_ns, positions, market_ids),
                self._labels(origin_ns, positions, market_ids, published_by))

    def check(self, samples, published_by=None):
        """Compare against the CPU reference path; raises on any mismatch."""
        from dataclasses import fields

        reference_batch, reference_targets = self.builder.tensors(samples, published_by)
        batch, targets = self.tensors(samples, published_by)
        for reference, built in ((reference_batch, batch), (reference_targets, targets)):
            for field in fields(type(reference)):
                left, right = getattr(reference, field.name), getattr(built, field.name)
                if left is None:
                    assert right is None, field.name
                elif isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right.to(left.device))
                elif field.name != "data_cutoffs":
                    assert left == right, field.name
        return True
