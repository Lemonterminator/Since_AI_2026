"""Asymmetric, horizon-wise conformal interval calibration.

Static CQR uses finite-sample ``higher`` order statistics. Optional online
variants are explicit research adaptations for delayed feedback; predictions
never update their own calibration state. Freeze the fitted object for a
competition replay.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_VARIANTS = {"static", "quantile_tracking", "integral_adaptation"}


def _iso_z(value: np.datetime64) -> str:
    return f"{np.datetime_as_string(value, unit='us')}Z"


def _times(values: Any, name: str, *, scalar: bool = False) -> np.ndarray:
    arr = np.asarray([values] if scalar else values, dtype=object)
    if (not scalar and arr.ndim != 1) or (scalar and arr.size != 1):
        raise ValueError(f"{name} must be {'a scalar' if scalar else 'one-dimensional'}")
    out = []
    for value in arr.reshape(-1):
        if value is None or isinstance(value, np.datetime64):
            raise ValueError(f"{name} timestamps must be present and timezone-aware")
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{name} must contain ISO timestamps with a UTC offset") from exc
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} timestamps must be timezone-aware")
        out.append(np.datetime64(value.astimezone(timezone.utc).replace(tzinfo=None), "ns"))
    return np.asarray(out, dtype="datetime64[ns]")


def _matrix(value: Any, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(value, dtype=np.float64)
    was_vector = arr.ndim == 1
    if was_vector:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, horizon), with both dimensions > 0")
    if np.isinf(arr).any():
        raise ValueError(f"{name} may contain NaN for missing values but not infinity")
    return arr, was_vector


def _higher_order_statistic(values: np.ndarray, level: float) -> float:
    """Finite-sample conformal order statistic; never silently caps rank at n."""
    n = len(values)
    rank = int(np.ceil((n + 1) * level))  # one-based rank
    if rank > n:
        raise ValueError(
            f"finite-sample rank {rank} exceeds n={n} at level={level:g}; "
            "collect more calibration rows or choose a lower nominal coverage"
        )
    if rank < 1:
        raise ValueError("finite-sample order-statistic rank is below 1")
    return float(np.partition(values, rank - 1)[rank - 1])


class AsymmetricCQR:
    """Calibrate lower/upper intervals independently for every horizon.

    Missing numeric observations (NaN in any of lower, upper, target) are
    dropped per horizon. Missing timestamps are rejected because their label
    availability cannot be proven. ``variant='integral_adaptation'`` is an
    explicit simple online adaptation, not the paper's native tan-integrator
    or complete ensemble/retraining protocol.
    """

    def __init__(self, *, alpha: float = 0.2, variant: str = "static",
                 eta: float = 0.01, integral_gain: float = 0.001,
                 training_cutoff: Any | None = None):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and in (0, 1)")
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not np.isfinite(integral_gain) or integral_gain < 0:
            raise ValueError("integral_gain must be finite and non-negative")
        self.alpha = float(alpha)
        self.variant = variant
        self.eta = float(eta)
        self.integral_gain = float(integral_gain)
        self.training_cutoff = (None if training_cutoff is None
                                else _iso_z(_times(training_cutoff, "training_cutoff", scalar=True)[0]))
        self.lower_correction: np.ndarray | None = None
        self.upper_correction: np.ndarray | None = None
        self.lower_integral: np.ndarray | None = None
        self.upper_integral: np.ndarray | None = None
        self.lower_count: np.ndarray | None = None
        self.upper_count: np.ndarray | None = None
        self.horizon: int | None = None
        self.fit_cutoff: str | None = None
        self.last_asof: str | None = None
        self.frozen = False

    def fit(self, lower: Any, upper: Any, target: Any, *, issued_at: Any,
            label_usable: Any, cutoff: Any, frozen: bool = True) -> "AsymmetricCQR":
        if self.frozen:
            raise RuntimeError("calibrator is frozen; construct an unfrozen instance to refit")
        lo, lo_vector = _matrix(lower, "lower")
        hi, hi_vector = _matrix(upper, "upper")
        y, y_vector = _matrix(target, "target")
        if lo.shape != hi.shape or lo.shape != y.shape:
            raise ValueError("lower, upper, and target must have identical shapes")
        if np.any((lo > hi) & np.isfinite(lo) & np.isfinite(hi)):
            raise ValueError("lower quantiles must not exceed upper quantiles")
        n, h = y.shape
        issued = _times(issued_at, "issued_at")
        usable = _times(label_usable, "label_usable")
        cutoff_value = _times(cutoff, "cutoff", scalar=True)[0]
        if len(issued) != n or len(usable) != n:
            raise ValueError("issued_at and label_usable must match the sample count")
        if np.any(issued >= usable):
            raise ValueError("calibration rows must be out-of-sample: issued_at < label_usable")
        if np.any(issued > cutoff_value) or np.any(usable > cutoff_value):
            raise ValueError("calibration rows must be issued and label-usable by cutoff")
        if np.any(usable < issued):
            raise ValueError("label_usable cannot precede issued_at")

        tau = 1.0 - self.alpha / 2.0
        lower_adj = np.empty(h, dtype=np.float64)
        upper_adj = np.empty(h, dtype=np.float64)
        for step in range(h):
            valid = np.isfinite(lo[:, step]) & np.isfinite(hi[:, step]) & np.isfinite(y[:, step])
            if not valid.any():
                raise ValueError(f"no complete calibration rows for horizon index {step}")
            # Positive upper residual expands the upper endpoint; positive
            # lower residual expands the lower endpoint downwards.
            upper_scores = y[valid, step] - hi[valid, step]
            lower_scores = lo[valid, step] - y[valid, step]
            lower_adj[step] = _higher_order_statistic(lower_scores, tau)
            upper_adj[step] = _higher_order_statistic(upper_scores, tau)

        self.lower_correction = lower_adj
        self.upper_correction = upper_adj
        self.lower_integral = np.zeros(h, dtype=np.float64)
        self.upper_integral = np.zeros(h, dtype=np.float64)
        self.lower_count = np.zeros(h, dtype=np.int64)
        self.upper_count = np.zeros(h, dtype=np.int64)
        self.horizon = h
        self.fit_cutoff = _iso_z(cutoff_value)
        self.last_asof = self.fit_cutoff
        self.frozen = bool(frozen)
        return self

    def predict(self, lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.lower_correction is None or self.upper_correction is None or self.horizon is None:
            raise RuntimeError("calibrator is not fitted")
        lo, lo_vector = _matrix(lower, "lower")
        hi, hi_vector = _matrix(upper, "upper")
        if lo.shape != hi.shape or lo.shape[1] != self.horizon:
            raise ValueError("prediction shapes must match and use the fitted horizon count")
        if np.any((lo > hi) & np.isfinite(lo) & np.isfinite(hi)):
            raise ValueError("lower quantiles must not exceed upper quantiles")
        out_lo = lo - self.lower_correction[None, :]
        out_hi = hi + self.upper_correction[None, :]
        if np.any((out_lo > out_hi) & np.isfinite(out_lo) & np.isfinite(out_hi)):
            raise ValueError("calibration corrections cross the prediction interval")
        if lo_vector:
            return out_lo[:, 0], out_hi[:, 0]
        return out_lo, out_hi

    def update(self, lower: Any, upper: Any, target: Any, *, issued_at: Any,
               label_usable: Any, asof: Any) -> "AsymmetricCQR":
        """Apply one delayed-feedback online tracking update.

        ``issued_at < label_usable <= asof`` is required. Updates are
        impossible after freeze and require an explicit ``training_cutoff``;
        labels or as-of times beyond that cutoff are rejected.
        """
        if self.frozen:
            raise RuntimeError("calibrator is frozen; online update is disabled")
        if self.variant == "static":
            raise RuntimeError("static CQR has no update rule; select an online variant")
        if self.lower_correction is None or self.upper_correction is None or self.horizon is None:
            raise RuntimeError("calibrator is not fitted")
        lo, lo_vec = _matrix(lower, "lower")
        hi, hi_vec = _matrix(upper, "upper")
        y, y_vec = _matrix(target, "target")
        if lo.shape != hi.shape or lo.shape != y.shape or lo.shape[0] != 1 or lo.shape[1] != self.horizon:
            raise ValueError("update inputs must be one row with the fitted horizon count")
        if np.any((lo > hi) & np.isfinite(lo) & np.isfinite(hi)):
            raise ValueError("lower quantiles must not exceed upper quantiles")
        issued = _times(issued_at, "issued_at", scalar=True)[0]
        usable = _times(label_usable, "label_usable", scalar=True)[0]
        current = _times(asof, "asof", scalar=True)[0]
        fit_cutoff = _times(self.fit_cutoff, "fit_cutoff", scalar=True)[0]
        previous = _times(self.last_asof, "last_asof", scalar=True)[0]
        if self.training_cutoff is None:
            raise RuntimeError("online update requires an explicit training_cutoff")
        training_cutoff = _times(self.training_cutoff, "training_cutoff", scalar=True)[0]
        if issued >= usable or usable > current:
            raise ValueError("update requires issued_at < label_usable <= asof")
        if (current < fit_cutoff or current <= previous or current > training_cutoff
                or usable > training_cutoff):
            raise ValueError("asof and labels must advance after fit cutoff and stay within training_cutoff")

        tau = 1.0 - self.alpha / 2.0
        for step in range(self.horizon):
            if not (np.isfinite(lo[0, step]) and np.isfinite(hi[0, step]) and np.isfinite(y[0, step])):
                continue  # explicit missing policy: skip this horizon only
            u_score = y[0, step] - hi[0, step]
            l_score = lo[0, step] - y[0, step]
            self._track_one(self.upper_correction, self.upper_integral,
                            self.upper_count, step, u_score, tau)
            self._track_one(self.lower_correction, self.lower_integral,
                            self.lower_count, step, l_score, tau)
        self.last_asof = _iso_z(current)
        return self

    def _track_one(self, q: np.ndarray, integral: np.ndarray,
                   count: np.ndarray, step: int, score: float, tau: float) -> None:
        # Quantile-loss subgradient. The integral variant is deliberately
        # named as a simple adaptation, not the paper's tan-saturated PI rule.
        error = tau - float(score <= q[step])
        count[step] += 1
        integral[step] += error
        delta = self.eta * error
        if self.variant == "integral_adaptation":
            delta += self.integral_gain * integral[step] / np.sqrt(count[step])
        q[step] += delta

    def freeze(self) -> "AsymmetricCQR":
        if self.lower_correction is None:
            raise RuntimeError("cannot freeze an unfitted calibrator")
        self.frozen = True
        return self

    def to_dict(self) -> dict[str, Any]:
        if self.lower_correction is None or self.upper_correction is None:
            raise RuntimeError("cannot serialize an unfitted calibrator")
        return {"format": "asymmetric-cqr-v1", "alpha": self.alpha,
                "variant": self.variant, "eta": self.eta, "integral_gain": self.integral_gain,
                "training_cutoff": self.training_cutoff,
                "lower_correction": self.lower_correction.tolist(),
                "upper_correction": self.upper_correction.tolist(),
                "lower_integral": self.lower_integral.tolist(),
                "upper_integral": self.upper_integral.tolist(),
                "lower_count": self.lower_count.tolist(),
                "upper_count": self.upper_count.tolist(), "horizon": self.horizon,
                "fit_cutoff": self.fit_cutoff, "last_asof": self.last_asof,
                "frozen": self.frozen}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "AsymmetricCQR":
        if not isinstance(state, dict) or state.get("format") != "asymmetric-cqr-v1":
            raise ValueError("unsupported or malformed CQR state")
        obj = cls(alpha=state["alpha"], variant=state["variant"],
                  eta=state["eta"], integral_gain=state["integral_gain"],
                  training_cutoff=state.get("training_cutoff"))
        h = int(state["horizon"])
        arrays = {}
        for key in ("lower_correction", "upper_correction", "lower_integral", "upper_integral"):
            arr = np.asarray(state[key], dtype=np.float64)
            if arr.shape != (h,) or not np.isfinite(arr).all():
                raise ValueError(f"invalid {key} in state")
            arrays[key] = arr
        counts = {}
        for key in ("lower_count", "upper_count"):
            arr = np.asarray(state[key], dtype=np.int64)
            if arr.shape != (h,) or np.any(arr < 0):
                raise ValueError(f"invalid {key} in state")
            counts[key] = arr
        if not isinstance(state.get("frozen"), bool):
            raise ValueError("frozen must be a boolean")
        _times(state["fit_cutoff"], "fit_cutoff", scalar=True)
        _times(state["last_asof"], "last_asof", scalar=True)
        obj.lower_correction = arrays["lower_correction"]
        obj.upper_correction = arrays["upper_correction"]
        obj.lower_integral = arrays["lower_integral"]
        obj.upper_integral = arrays["upper_integral"]
        obj.lower_count, obj.upper_count = counts["lower_count"], counts["upper_count"]
        obj.horizon = h
        obj.fit_cutoff, obj.last_asof = state["fit_cutoff"], state["last_asof"]
        obj.frozen = state["frozen"]
        return obj

    @classmethod
    def from_json(cls, payload: str) -> "AsymmetricCQR":
        try:
            state = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid CQR JSON") from exc
        return cls.from_dict(state)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AsymmetricCQR":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
