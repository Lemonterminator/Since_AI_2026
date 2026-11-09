"""Frozen, chronological calibration and BOA task experiments.

Input quantiles are spreads, not prices. BOA learns only in the first half of
the calibration window with delayed complete-horizon feedback. Its state is
then frozen before the second half supplies ODIR and CQR labels. CQR changes
grid widths; classifier spike probabilities are independent after that change.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import PointwiseBOA
from .conformal import AsymmetricCQR
from .dirichlet import DirichletCalibrator

LEVELS = np.arange(1, 20, dtype=np.float64) / 20
HORIZON = 40
MEDIAN = 9


def _aware(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be present and timezone-aware")
    return stamp.tz_convert("UTC")


def _origins(rows):
    if len(rows) == 0 or len(rows) % HORIZON:
        raise ValueError("rows must contain complete 40-step origins")
    if not {"origin", "step"}.issubset(rows.columns):
        raise ValueError("rows require origin and step")
    origin = pd.DatetimeIndex([_aware(x, "origin") for x in rows["origin"]])
    times = origin.to_numpy().reshape(-1, HORIZON)
    if not np.all(times == times[:, :1]):
        raise ValueError("each consecutive 40 rows must share an origin")
    unique = pd.DatetimeIndex(times[:, 0])
    if not unique.is_monotonic_increasing or not unique.is_unique:
        raise ValueError("origins must be strictly increasing")
    if not np.all(rows["step"].to_numpy().reshape(-1, HORIZON) == np.arange(1, 41)):
        raise ValueError("steps must be ordered 1..40 per origin")
    return unique


def _predictions(preds, n, expected=None):
    names = tuple(preds) if expected is None else tuple(expected)
    if set(preds) != set(names) or len(names) < 2:
        raise ValueError("predictions must supply exactly the fitted >=2 expert names")
    clean = {}
    for name in names:
        item = {}
        for key, size in (("q", 19), ("d", 3), ("s", 3)):
            a = np.asarray(preds[name][key], dtype=np.float64)
            if a.shape != (n, HORIZON, size) or not np.isfinite(a).all():
                raise ValueError(f"{name}.{key} must be finite shape {(n, HORIZON, size)}")
            if key == "q":
                if np.any(np.diff(a, axis=-1) < 0):
                    raise ValueError(f"{name} quantiles cross")
                item[key] = a.copy()
            else:
                totals = a.sum(-1, keepdims=True)
                if np.any(a < 0) or np.any(totals <= 0) or not np.allclose(totals, 1, atol=1e-6):
                    raise ValueError(f"{name}.{key} probabilities must sum to one")
                item[key] = a / totals
        clean[name] = item
    return clean


def _average(preds):
    return {k: np.mean([p[k] for p in preds.values()], axis=0) for k in ("q", "d", "s")}


def _boa_prediction(boa, preds, origins):
    mean = _average(preds)
    cubes = np.stack([x["q"] for x in preds.values()], axis=1)
    mean["q"] = np.stack([boa.predict(x, issued_at=t.to_pydatetime(), rearrange=True).quantiles
                          for x, t in zip(cubes, origins)])
    return mean


def _width_calibration(grid, cqr):
    """Expand corrected endpoints toward the fixed median if needed.

    All quantiles on each side scale together. This is a task adapter to keep
    the official P10/P50/P90 contract, with no finite-sample coverage claim
    after the median projection. Degenerate input sides use level distances.
    """
    q = np.asarray(grid, dtype=np.float64)
    med = q[..., MEDIAN]
    lo = np.minimum(q[..., 1] - cqr.lower_correction, med)
    hi = np.maximum(q[..., 17] + cqr.upper_correction, med)
    left_width, right_width = med - q[..., 1], q[..., 17] - med
    left_dev, right_dev = med[..., None] - q[..., :MEDIAN], q[..., MEDIAN + 1:] - med[..., None]
    left_ratio = np.divide(left_dev, left_width[..., None], out=np.zeros_like(left_dev),
                           where=left_width[..., None] > 1e-12)
    right_ratio = np.divide(right_dev, right_width[..., None], out=np.zeros_like(right_dev),
                            where=right_width[..., None] > 1e-12)
    left_ratio = np.where((left_width > 1e-12)[..., None], left_ratio,
                          (.5 - LEVELS[:MEDIAN]) / .4)
    right_ratio = np.where((right_width > 1e-12)[..., None], right_ratio,
                           (LEVELS[MEDIAN + 1:] - .5) / .4)
    out = q.copy()
    out[..., :MEDIAN] = med[..., None] - left_ratio * (med - lo)[..., None]
    out[..., MEDIAN + 1:] = med[..., None] + right_ratio * (hi - med)[..., None]
    return out


@dataclass
class FrozenPostprocessors:
    names: tuple[str, ...]
    boa: PointwiseBOA
    odir: dict[str, dict[str, DirichletCalibrator]]
    cqr: dict[str, AsymmetricCQR]
    metadata: dict

    def predict(self, preds, rows):
        origins = _origins(rows)
        if origins.min() < _aware(self.metadata["cutoff"], "cutoff"):
            raise ValueError("frozen postprocessing predictions must begin at or after calibration cutoff")
        clean = _predictions(preds, len(origins), self.names)
        sources = {**clean, "boa": _boa_prediction(self.boa, clean, origins)}
        result = {"equal_ensemble": _average(clean), "boa": sources["boa"]}
        for name, value in sources.items():
            calibrated = {k: v.copy() for k, v in value.items()}
            for key in ("d", "s"):
                calibrated[key] = self.odir[name][key].predict(value[key].reshape(-1, 3)).reshape(value[key].shape)
            result[name + "_odir"] = calibrated
            intervals = {k: v.copy() for k, v in value.items()}
            intervals["q"] = _width_calibration(value["q"], self.cqr[name])
            result[name + "_cqr"] = intervals
        return result

    def save(self, path):
        state = {"schema": "probability-postprocess-v1", "names": list(self.names),
                 "metadata": self.metadata, "boa": self.boa.state_dict(),
                 "odir": {n: {k: v.to_dict() for k, v in group.items()} for n, group in self.odir.items()},
                 "cqr": {n: v.to_dict() for n, v in self.cqr.items()}}
        Path(path).write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("schema") != "probability-postprocess-v1":
            raise ValueError("unknown postprocessor schema")
        obj = cls(tuple(state["names"]), PointwiseBOA.from_state_dict(state["boa"]),
                  {n: {k: DirichletCalibrator.from_dict(v) for k, v in g.items()} for n, g in state["odir"].items()},
                  {n: AsymmetricCQR.from_dict(v) for n, v in state["cqr"].items()}, state["metadata"])
        if not obj.boa.frozen or any(not v.frozen for g in obj.odir.values() for v in g.values()) or any(not v.frozen for v in obj.cqr.values()):
            raise ValueError("loaded postprocessors must all be frozen")
        return obj


def fit_postprocessors(preds, rows, cutoff):
    """Fit fixed experimental variants using two separated calibration halves.

    Callers must provide genuine chronological OOS expert forecasts. Timestamp
    checks cannot establish that the expert training did not see these labels.
    Rows must be complete, sorted, published by cutoff, and precede cutoff.
    """
    cutoff = _aware(cutoff, "cutoff")
    origins = _origins(rows)
    n = len(origins)
    if n < 20:
        raise ValueError("at least 20 calibration origins required for two halves")
    clean = _predictions(preds, n)
    if not {"label_usable", "spread", "direction", "spike"}.issubset(rows.columns):
        raise ValueError("calibration rows require labels and publication times")
    usable_flat = pd.DatetimeIndex([_aware(x, "label_usable") for x in rows["label_usable"]])
    issued_flat = origins.repeat(HORIZON)
    if np.any(issued_flat >= usable_flat) or np.any(usable_flat > cutoff) or origins.max() >= cutoff:
        raise ValueError("require origin < label_usable <= calibration cutoff")
    y = rows["spread"].to_numpy(dtype=np.float64).reshape(n, HORIZON)
    if not np.isfinite(y).all():
        raise ValueError("all calibration spread labels must be finite")
    labels = {}
    for key, column, classes in (("d", "direction", ("up", "down", "none")),
                                 ("s", "spike", ("normal", "high", "low"))):
        codes = pd.Categorical(rows[column], categories=classes).codes.astype(np.int64)
        if np.any(codes < 0):
            raise ValueError(f"unknown or missing {column} class")
        labels[key] = codes.reshape(n, HORIZON)
    midpoint = n // 2
    split = origins[midpoint]
    usable = [max(usable_flat[i * HORIZON:(i + 1) * HORIZON]) for i in range(n)]
    boa = PointwiseBOA(LEVELS, len(clean), HORIZON, training_cutoff=split.to_pydatetime())
    cubes = np.stack([x["q"] for x in clean.values()], axis=1)
    pending = []
    updates = 0

    def flush(asof):
        nonlocal updates
        while pending and pending[0][0] <= asof:
            publication, i, ticket = heapq.heappop(pending)
            boa.update(ticket, y[i], label_usable=publication.to_pydatetime(), asof=asof.to_pydatetime())
            updates += 1

    for i in range(midpoint):
        flush(origins[i])
        ticket = boa.predict(cubes[i], issued_at=origins[i].to_pydatetime())
        heapq.heappush(pending, (usable[i], i, ticket))
    flush(split)
    if updates == 0:
        raise ValueError("no complete first-half horizon labels published before BOA freeze")
    boa.freeze()
    second = {name: {key: a[midpoint:] for key, a in value.items()} for name, value in clean.items()}
    sources = {**second, "boa": _boa_prediction(boa, second, origins[midpoint:])}
    times = {"issued_at": origins[midpoint:].repeat(HORIZON).to_pydatetime(),
             "label_usable": usable_flat[midpoint * HORIZON:].to_pydatetime(),
             "cutoff": cutoff.to_pydatetime()}
    odir, cqr = {}, {}
    for name, value in sources.items():
        odir[name] = {key: DirichletCalibrator("odir").fit(
            value[key].reshape(-1, 3), labels[key][midpoint:].reshape(-1), **times)
            for key in ("d", "s")}
        cqr[name] = AsymmetricCQR(alpha=.2).fit(
            value["q"][..., 1], value["q"][..., 17], y[midpoint:],
            issued_at=origins[midpoint:].to_pydatetime(),
            label_usable=[x.to_pydatetime() for x in usable[midpoint:]], cutoff=cutoff.to_pydatetime())
    metadata = {"cutoff": cutoff.isoformat(), "calibration_start": origins[0].isoformat(),
                "boa_freeze": split.isoformat(), "boa_updates": updates,
                "boa_pending_discarded_at_freeze": len(pending), "origins_first_half": midpoint,
                "origins_second_half": n - midpoint,
                "cqr_policy": "side-width rescaling; endpoints expanded toward fixed median; independent spike classifier",
                "odir_objective": "mean multiclass NLL plus fixed default ODIR penalty; no hyperparameter search"}
    return FrozenPostprocessors(tuple(clean), boa, odir, cqr, metadata)
