"""Official metrics, purged splits, and deterministic checkpoint records."""
from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path

import pandas as pd
import torch

from .data import PRICE, QUARTER, repository_module, utc

# Keep the existing development folds without importing its LightGBM dependency.
FOLDS = (("late_june", "2026-06-19T21:00Z", "2026-06-30T21:00Z"),
         ("july", "2026-06-30T21:00Z", "2026-07-31T21:00Z"),
         ("august_to_sep06", "2026-07-31T21:00Z", "2026-09-06T21:00Z"),
         ("official_validation", "2026-09-06T21:00Z", "2026-10-04T21:00Z"))


def fold_origins(origins, start, end, label_release, horizon=40):
    origins = pd.DatetimeIndex(origins)
    start, end = utc(start), utc(end)
    release = pd.DatetimeIndex(label_release)
    if len(release) != len(origins):
        raise ValueError("Each origin needs its last label publication timestamp")
    train = (origins <= start - pd.Timedelta(hours=12)) & (release <= start)
    train &= origins >= pd.Timestamp("2026-05-31T21:00Z")
    validation = (origins >= start) & (origins + (horizon - 1) * QUARTER < end)
    return origins[train], origins[validation]


def adapter_blocks(origins, label_release, horizon=40):
    """70/20/10 chronological stages with a twelve-hour boundary purge."""
    origins, releases = pd.DatetimeIndex(origins), pd.DatetimeIndex(label_release)
    if not origins.is_monotonic_increasing or origins.has_duplicates or len(origins) != len(releases):
        raise ValueError("Adapter origins must be unique, sorted, and paired with label releases")
    if len(origins) < 3:
        raise ValueError("Three nonempty chronological stages are required")
    first, second = origins[int(.7 * len(origins))], origins[int(.9 * len(origins))]
    if origins[-1] + (horizon - 1) * QUARTER >= pd.Timestamp("2026-10-04T21:00Z"):
        raise ValueError("Adapter stages cannot contain October test targets")
    gap = pd.Timedelta(hours=12)
    base = origins[(origins <= first - gap) & (releases <= first)]
    adapter = origins[(origins >= first) & (origins <= second - gap) & (releases <= second)]
    calibration = origins[origins >= second]
    if not all(len(stage) for stage in (base, adapter, calibration)):
        raise ValueError("Insufficient origins after twelve-hour purges")
    return base, adapter, calibration


def score_frames(prediction, actuals):
    scorer = repository_module("data/silver/starter_package_v1/score.py", "imbalance_scorer")
    merged = prediction.copy()
    truth = actuals.copy()
    for frame in (merged, truth):
        frame["delivery_start_utc"] = pd.to_datetime(frame.delivery_start_utc, utc=True)
    merged = merged.merge(truth, on="delivery_start_utc", how="left", validate="many_to_one")
    metrics = scorer.compute_metrics(merged, 50)
    horizons = {name: scorer.compute_metrics(merged.loc[merged.step.between(lo, hi)], 50)
                for name, lo, hi in (("1-8", 1, 8), ("9-19", 9, 19), ("20-40", 20, 40))}
    by_step = {step: scorer.compute_metrics(group, 50) for step, group in merged.groupby("step")}
    extreme = merged.loc[(merged.imbalance_price - merged.spot_price).abs() > 50]
    return {"overall": metrics, "horizon_groups": horizons, "by_step": by_step,
            "extreme": scorer.compute_metrics(extreme, 50)}


def save_checkpoint(path, model, config, features, scaler, metadata, *, price_series=PRICE):
    from .inputs import PRICE_SERIES
    if price_series not in PRICE_SERIES:
        raise ValueError(f"Unsupported checkpoint price series: {price_series}")
    required = {"upstream_commits", "code_sha", "data_hashes", "splits", "losses", "seed", "device", "runtime", "metrics"}
    if not required.issubset(metadata):
        raise ValueError(f"Missing experiment metadata: {sorted(required - metadata.keys())}")
    payload = {"model": model.state_dict(), "config": asdict(config) if is_dataclass(config) else config,
               "features": [asdict(f) if is_dataclass(f) else f for f in features],
               "scaler": scaler.state_dict(), "metadata": metadata, "price_series": price_series}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, expected_features=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if expected_features is not None:
        expected = [asdict(f) if is_dataclass(f) else f for f in expected_features]
        if checkpoint["features"] != expected:
            raise ValueError("Checkpoint feature specification/order differs")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
