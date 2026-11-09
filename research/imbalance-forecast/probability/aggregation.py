"""Pointwise CRPS-learning BOA core; no automatic tuning or spline basis reduction.

Independent implementation of Berrisch/Ziel (arXiv:2303.10019, Eq. 12).
Sign convention follows the author's profoc reference, inspected at commit
``b5073ca70aea60ce209676faa99e04da725ffe04`` (src/conline.cpp:325,397-406):
regret = linearized forecaster loss - linearized expert loss, and weights use
``softmax(log(eta) + eta * R)``. The minus sign printed in paper Eq. 12f
is inconsistent with this regret convention. No profoc source is bundled.

Each call issues one origin's ``(experts, horizons, quantiles)`` cube. Delayed
feedback must reference the actual issued forecast ticket, never a prediction
recomputed under newer weights. Predictions do not mutate learner state.
Pointwise combination can cross quantiles. Rearrangement is opt-in and intended
for frozen output; updates reject rearranged tickets so their loss is never
silently confused with the native pointwise gradient loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


def _utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware datetime or ISO strings")
    return value.astimezone(timezone.utc)


def _finite(value, name, shape=None):
    a = np.asarray(value, dtype=np.float64)
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite values")
    return a


@dataclass(frozen=True)
class BOAForecast:
    """Immutable issued feedback ticket. Keep this object until labels publish."""

    quantiles: np.ndarray
    raw_quantiles: np.ndarray
    experts: np.ndarray
    issued_at: datetime
    model_id: str
    forecast_id: str
    rearranged: bool


def smooth_weights(weights, *, horizon_strength=0.0, quantile_strength=0.0):
    """Separate grid-ridge smoothing adaptation, not native profoc B-splines.

    Solves first-difference ridge smoothing on each grid axis, projects negative
    values to zero and renormalizes over experts. Accepts ``(K,H,Q)`` weights.
    This utility does not change any BOA learner. Strengths must be nonnegative.
    """
    w = _finite(weights, "weights").copy()
    if w.ndim != 3 or min(w.shape) < 1 or np.any(w < 0):
        raise ValueError("weights must be a nonnegative nonempty (K,H,Q) cube")
    for axis, strength in ((1, horizon_strength), (2, quantile_strength)):
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("smoothing strengths must be finite and nonnegative")
        n = w.shape[axis]
        if strength and n > 1:
            diff = np.diff(np.eye(n), axis=0)
            operator = np.eye(n) + strength * (diff.T @ diff)
            moved = np.moveaxis(w, axis, 0)
            moved = np.linalg.solve(operator, moved.reshape(n, -1)).reshape(moved.shape)
            w = np.moveaxis(moved, 0, axis)
    w = np.maximum(w, 0)
    denom = w.sum(axis=0, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("weights must have positive mass at every grid location")
    return w / denom


class PointwiseBOA:
    """Uniform-prior pointwise Bernstein online aggregation.

    Parameters are quantile levels, number of experts, number of horizons and an
    explicit training cutoff. ``predict`` returns a BOAForecast ticket;
    ``ticket.quantiles`` is the forecast array. ``update(ticket, truth, ...)``
    consumes one *complete* horizon's delayed feedback exactly once. Partial or
    missing labels must be queued by the caller until this contract is met.
    ``freeze()`` permanently disables updates for competition inference.

    State is JSON-safe (``state_dict``, ``from_state_dict``, ``save``, ``load``).
    No claim of complete profoc reproduction: basis reduction, forgetting,
    hyperparameter search and penalized spline smoothing are not included.
    """

    SCHEMA = 1

    def __init__(self, quantile_levels, n_experts, n_horizons, *, training_cutoff,
                 initial_bound=1.0):
        q = _finite(quantile_levels, "quantile_levels")
        if q.ndim != 1 or not len(q) or np.any(np.diff(q) <= 0) or np.any((q <= 0) | (q >= 1)):
            raise ValueError("quantile_levels must be strictly increasing inside (0,1)")
        if isinstance(n_experts, bool) or int(n_experts) != n_experts or n_experts < 2:
            raise ValueError("n_experts must be an integer >= 2")
        if isinstance(n_horizons, bool) or int(n_horizons) != n_horizons or n_horizons < 1:
            raise ValueError("n_horizons must be a positive integer")
        if not np.isfinite(initial_bound) or initial_bound <= 0:
            raise ValueError("initial_bound must be finite and positive")
        self.quantile_levels = q.copy()
        self.shape = (int(n_experts), int(n_horizons), len(q))
        self.training_cutoff = _utc(training_cutoff)
        self.initial_bound = float(initial_bound)
        self.frozen = False
        self._model_id = uuid4().hex
        self._weights = np.full(self.shape, 1.0 / self.shape[0])
        self._R = np.zeros(self.shape)
        self._V = np.zeros(self.shape)
        # Explicit scale initialization avoids infinite zero-variance learning
        # rates rewarding a zero-regret expert solely through log(eta).
        self._E = np.full(self.shape, self.initial_bound)
        self._last_asof = None
        self._used_tickets = set()

    @property
    def weights(self):
        return self._weights.copy()

    def predict(self, experts, *, issued_at, rearrange=False) -> BOAForecast:
        x = _finite(experts, "experts", self.shape).copy()
        if np.any(np.diff(x, axis=-1) < 0):
            raise ValueError("each expert must supply ordered quantiles")
        issued = _utc(issued_at)
        if self._last_asof is not None and issued < self._last_asof:
            raise ValueError("cannot issue a forecast earlier than the learner's latest feedback")
        raw = np.sum(self._weights * x, axis=0)
        if not np.isfinite(raw).all():
            raise FloatingPointError("aggregation overflow")
        output = np.sort(raw, axis=-1) if rearrange else raw.copy()
        for a in (x, raw, output):
            a.flags.writeable = False
        return BOAForecast(output, raw, x, issued, self._model_id, uuid4().hex, bool(rearrange))

    def update(self, ticket: BOAForecast, truth, *, label_usable, asof):
        """Consume published labels; require issued < usable <= asof <= cutoff.

        ``label_usable`` must be the latest publication timestamp among all labels
        in ``truth``. Caller is responsible for authentic publication metadata.
        Regret uses the ticket's issued raw forecast, not today's weights.
        """
        if self.frozen:
            raise RuntimeError("frozen aggregation cannot update")
        if not isinstance(ticket, BOAForecast) or ticket.model_id != self._model_id:
            raise ValueError("feedback ticket belongs to another learner")
        if ticket.forecast_id in self._used_tickets:
            raise ValueError("feedback ticket already consumed")
        if ticket.rearranged:
            raise ValueError("native pointwise BOA cannot update from rearranged output")
        usable, at = _utc(label_usable), _utc(asof)
        if not (ticket.issued_at < usable <= at <= self.training_cutoff):
            raise ValueError("require issued_at < label_usable <= asof <= training_cutoff")
        if self._last_asof is not None and at < self._last_asof:
            raise ValueError("feedback asof cannot move backwards")
        y = _finite(truth, "truth", (self.shape[1],))
        x = _finite(ticket.experts, "ticket.experts", self.shape)
        pred = _finite(ticket.raw_quantiles, "ticket.raw_quantiles", self.shape[1:])
        # profoc loss.cpp uses the valid quantile-loss subgradient at equality
        # (pred >= truth) - tau. Positive r means the expert was preferable.
        gradient = (pred >= y[:, None]).astype(float) - self.quantile_levels
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r = gradient[None, :, :] * (pred[None, :, :] - x)
            v = self._V + r * r
            e = np.maximum(self._E, np.abs(r))
            # A zero-gradient location leaves its prior unchanged. Floors/caps
            # only define the zero-regret limit and prevent floating overflow.
            floor = np.exp(-350.0)
            eta = np.minimum(np.sqrt(np.log(self.shape[0]) / np.maximum(v, floor)),
                             1.0 / (2.0 * np.maximum(e, floor)))
            eta = np.minimum(eta, np.exp(350.0))
            R = self._R + 0.5 * r * (1.0 - eta * r) + e * (eta * r > 0.5)
            logits = np.log(eta) + eta * R
            logits -= logits.max(axis=0, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=0, keepdims=True)
        self._V, self._E, self._R, self._weights = v, e, R, weights
        self._last_asof = at
        self._used_tickets.add(ticket.forecast_id)
        return self

    def freeze(self):
        self.frozen = True
        return self

    def state_dict(self):
        return {"schema": self.SCHEMA, "quantile_levels": self.quantile_levels.tolist(),
                "n_experts": self.shape[0], "n_horizons": self.shape[1],
                "training_cutoff": self.training_cutoff.isoformat(), "frozen": self.frozen,
                "initial_bound": self.initial_bound,
                "model_id": self._model_id, "weights": self._weights.tolist(),
                "R": self._R.tolist(), "V": self._V.tolist(), "E": self._E.tolist(),
                "last_asof": self._last_asof.isoformat() if self._last_asof else None,
                "used_tickets": sorted(self._used_tickets)}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported BOA state schema")
        model = cls(state["quantile_levels"], state["n_experts"], state["n_horizons"],
                    training_cutoff=state["training_cutoff"], initial_bound=state["initial_bound"])
        for key, target in (("weights", "_weights"), ("R", "_R"), ("V", "_V"), ("E", "_E")):
            setattr(model, target, _finite(state[key], key, model.shape).copy())
        if (model._weights < 0).any() or not np.allclose(model._weights.sum(axis=0), 1):
            raise ValueError("invalid BOA weights")
        if (model._V < 0).any() or (model._E < 0).any():
            raise ValueError("BOA variance and bounds must be nonnegative")
        model._last_asof = _utc(state["last_asof"]) if state["last_asof"] else None
        if model._last_asof and model._last_asof > model.training_cutoff:
            raise ValueError("stored feedback exceeds training cutoff")
        if not isinstance(state["frozen"], bool) or not isinstance(state["model_id"], str):
            raise ValueError("invalid BOA state metadata")
        if not isinstance(state["used_tickets"], list) or not all(isinstance(t, str) for t in state["used_tickets"]):
            raise ValueError("invalid feedback ticket history")
        model.frozen = state["frozen"]
        model._model_id = state["model_id"]
        model._used_tickets = set(state["used_tickets"])
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), allow_nan=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


CRPSBOA = PointwiseBOA
