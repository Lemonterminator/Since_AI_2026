"""Convert normalized forecasts to the official thirteen-column contract."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from .contracts import Forecast, ForecastBatch
from .data import repository_module

COLUMNS = ["forecast_origin_utc", "delivery_start_utc", "step", "p10", "p50", "p90",
           "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low", "data_cutoff_utc", "top_drivers"]
# Holdout bands for asymmetric P10/P90 calibration, as (first, last) steps.
INTERVAL_BANDS = ((1, 4), (5, 12), (13, 40))


def apply_interval_scales(quantiles: torch.Tensor, scales) -> torch.Tensor:
    """Widen the P10/P90 gaps step-wise; ``scales`` is one ``(lo, hi)`` per band."""
    if not scales:
        return quantiles
    median = quantiles[..., 1]
    lower = (median - quantiles[..., 0]).clamp_min(0.0)
    upper = (quantiles[..., 2] - median).clamp_min(0.0)
    step = torch.arange(quantiles.shape[1], device=quantiles.device)
    for (first, last), (lo_scale, hi_scale) in zip(INTERVAL_BANDS, scales):
        band = (step >= first - 1) & (step <= last - 1)
        lower = torch.where(band, lower * float(lo_scale), lower)
        upper = torch.where(band, upper * float(hi_scale), upper)
    return torch.stack((median - lower, median, median + upper), dim=-1)


def forecast_frame(forecast: Forecast, batch: ForecastBatch):
    if batch.series_ids is not None and (batch.series_ids != 0).any():
        raise ValueError("Official export accepts FI series ID 0 only")
    b, horizon, q = forecast.quantiles.shape
    if q != 3 or len(batch.origins) != b or forecast.direction.shape != (b, horizon, 3) or forecast.spikes.shape != (b, horizon, 3):
        raise ValueError("Forecast shapes and origin count disagree")
    if batch.spot.shape != (b, horizon) or (batch.spread_scale <= 0).any():
        raise ValueError("Invalid spot anchor or inverse scale")
    quantiles = (forecast.quantiles * batch.spread_scale[..., None] + batch.spread_center[..., None]
                 + batch.spot[..., None]).detach().cpu().double().numpy()
    direction = forecast.direction.detach().cpu().double().numpy()
    spikes = forecast.spikes.detach().cpu().double().numpy()
    if not all(np.isfinite(a).all() for a in (quantiles, direction, spikes)):
        raise ValueError("Forecast values must be finite")
    if (np.diff(quantiles, axis=-1) < 0).any():
        raise ValueError("Quantiles must be ordered")
    for array in (direction, spikes):
        if ((array < 0) | (array > 1)).any() or not np.allclose(array.sum(-1), 1, atol=1e-6):
            raise ValueError("Probabilities must be in [0,1] and sum to one")
    origins = pd.to_datetime(list(batch.origins), utc=True)
    if origins.has_duplicates or not origins.equals(origins.floor("h")):
        raise ValueError("Origins must be unique full UTC hours")
    rows = origins.repeat(horizon)
    step = np.tile(np.arange(1, horizon + 1), b)
    delivery = rows + pd.to_timedelta((step - 1) * 15, unit="min")
    cutoffs = batch.data_cutoffs or ("",) * b
    if len(cutoffs) != b:
        raise ValueError("Cutoff count must match origins")
    for origin, cutoff in zip(origins, cutoffs):
        if cutoff and pd.Timestamp(cutoff) > origin:
            raise ValueError("A source cutoff cannot be later than its origin")
    frame = pd.DataFrame({"forecast_origin_utc": rows.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "delivery_start_utc": delivery.strftime("%Y-%m-%dT%H:%M:%SZ"), "step": step,
                          **{name: quantiles[..., i].ravel() for i, name in enumerate(("p10", "p50", "p90"))},
                          **{name: direction[..., i].ravel() for i, name in enumerate(("p_up", "p_down", "p_none"))},
                          "p_spike_high": spikes[..., 1].ravel(), "p_spike_low": spikes[..., 2].ravel(),
                          "data_cutoff_utc": np.repeat(cutoffs, horizon), "top_drivers": ""})
    return frame[COLUMNS]


def write_submission(frame, path, origins_path=None):
    """Serialize and optionally validate beside the destination before replacing it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=path.suffix or ".tmp", delete=False,
                                         encoding="utf-8", newline="") as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, float_format="%.9g")
        if origins_path is not None:
            counts = validate_file(temporary, origins_path)
        temporary.replace(path)
        return counts
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_submission(forecast, batch, path, origins_path=None):
    frame = forecast_frame(forecast, batch)
    write_submission(frame, path, origins_path)
    return frame


def validate_file(path, origins_path):
    validator = repository_module("data/silver/starter_package_v1/validate_submission.py", "imbalance_validator")
    report, counts = validator.validate(Path(path), Path(origins_path), 40, 15, .001)
    if report.n_errors:
        raise ValueError(f"Official submission validation failed: {report.n_errors} errors")
    return counts


def samples_to_forecast(samples: torch.Tensor, direction: torch.Tensor, batch: ForecastBatch):
    """Keep empirical tails and strict raw-spread spike definitions consistent."""
    raw = samples * batch.spread_scale[:, None, :] + batch.spread_center[:, None, :]
    high, low = (raw > 50).to(samples.dtype).mean(1), (raw < -50).to(samples.dtype).mean(1)
    normal = ((raw >= -50) & (raw <= 50)).to(samples.dtype).mean(1)
    spikes = torch.stack((normal, high, low), -1)
    quantiles = torch.quantile(samples, samples.new_tensor([.1, .5, .9]), dim=1).permute(1, 2, 0)
    return Forecast(quantiles, direction, spikes, samples)
