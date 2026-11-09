"""Small, serializable probability calibrators for multiclass forecasts.

The ODIR objective follows Kull et al. (2019). ``l2`` is a separately named
identity-shrinkage adaptation; it is not the paper's ODIR penalty.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


_METHODS = {"identity", "temperature", "l2", "odir"}


def _cutoff_ns(values: Any, name: str) -> np.ndarray:
    """Convert timezone-aware timestamps to UTC nanoseconds; reject naive data."""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = []
    for value in array:
        if value is None or isinstance(value, np.datetime64):
            raise ValueError(f"{name} timestamps must be present and timezone-aware")
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{name} must contain ISO timestamps with a UTC offset") from exc
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} timestamps must be timezone-aware")
        utc_naive = value.astimezone(timezone.utc).replace(tzinfo=None)
        result.append(np.datetime64(utc_naive, "ns"))
    return np.asarray(result, dtype="datetime64[ns]")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _iso_z(value: np.datetime64) -> str:
    return f"{np.datetime_as_string(value, unit='us')}Z"


class DirichletCalibrator:
    """Fit an identity, temperature, L2, or ODIR multiclass probability map.

    ``fit`` requires issuance time, label availability time, and a calibration
    cutoff. Rows with either timestamp after the cutoff are rejected. By
    default, a fitted calibrator is frozen; use ``frozen=False`` only for an
    explicitly managed development-time refit workflow.
    """

    def __init__(self, method: str = "odir", *, clip: float = 1e-15,
                 l2: float = 1.0, odir_lambda: float = 1.0,
                 odir_mu: float = 1.0, max_iter: int = 500):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if not np.isfinite(clip) or clip <= 0 or clip >= 1:
            raise ValueError("clip must be finite and in (0, 1)")
        for name, value in (("l2", l2), ("odir_lambda", odir_lambda),
                            ("odir_mu", odir_mu)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.method = method
        self.clip = float(clip)
        self.l2 = float(l2)
        self.odir_lambda = float(odir_lambda)
        self.odir_mu = float(odir_mu)
        self.max_iter = max_iter
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.temperature: float | None = None
        self.n_classes: int | None = None
        self.frozen = False
        self.fit_cutoff: str | None = None

    @staticmethod
    def _probabilities(probabilities: Any, *, clip: float) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes>=2), n_samples>0")
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("probabilities must be finite and non-negative")
        totals = p.sum(axis=1)
        if np.any(totals <= 0) or not np.allclose(totals, 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError("each probability row must sum to 1")
        p = p / totals[:, None]
        p = np.maximum(p, clip)
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, probabilities: Any, labels: Any, *, issued_at: Any,
            label_usable: Any, cutoff: Any, frozen: bool = True) -> "DirichletCalibrator":
        if self.frozen:
            raise RuntimeError("calibrator is frozen; construct an unfrozen instance to refit")
        p = self._probabilities(probabilities, clip=self.clip)
        y = np.asarray(labels)
        if y.ndim != 1 or y.shape[0] != p.shape[0] or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("labels must be a 1-D integer class-index array matching probabilities")
        k = p.shape[1]
        if np.any(y < 0) or np.any(y >= k):
            raise ValueError("labels must be class indices in [0, n_classes)")
        issued = _cutoff_ns(issued_at, "issued_at")
        usable = _cutoff_ns(label_usable, "label_usable")
        cutoff_value = _cutoff_ns([cutoff], "cutoff")[0]
        if len(issued) != len(y) or len(usable) != len(y):
            raise ValueError("issued_at and label_usable must match labels")
        if np.any(issued >= usable):
            raise ValueError("calibration rows must be out-of-sample: issued_at < label_usable")
        if np.any(issued > cutoff_value) or np.any(usable > cutoff_value):
            raise ValueError("calibration rows must be issued and label-usable by cutoff")

        logp = np.log(p)
        one_hot = np.eye(k, dtype=np.float64)[y]
        if self.method == "identity":
            self.W, self.b = np.eye(k), np.zeros(k)
            self.temperature = None
        elif self.method == "temperature":
            def objective(log_temperature: float) -> float:
                logits = logp / np.exp(log_temperature)
                return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))
            result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded",
                                     options={"maxiter": self.max_iter, "xatol": 1e-10})
            if not result.success or not np.isfinite(result.fun):
                raise RuntimeError(f"temperature optimization failed: {result.message}")
            self.temperature = float(np.exp(result.x))
            self.W, self.b = np.eye(k), np.zeros(k)
        else:
            offdiag = ~np.eye(k, dtype=bool)
            denom_offdiag = float(k * (k - 1))
            denom_bias = float(k)

            def loss_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
                W = flat[:k * k].reshape(k, k)
                b = flat[k * k:]
                logits = logp @ W.T + b
                probs = _softmax(logits)
                loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
                dlogits = (probs - one_hot) / len(y)
                grad_W = dlogits.T @ logp
                grad_b = dlogits.sum(axis=0)
                if self.method == "odir":
                    loss += self.odir_lambda * np.sum(W[offdiag] ** 2) / denom_offdiag
                    loss += self.odir_mu * np.sum(b ** 2) / denom_bias
                    grad_W[offdiag] += 2 * self.odir_lambda * W[offdiag] / denom_offdiag
                    grad_b += 2 * self.odir_mu * b / denom_bias
                else:  # named identity-shrinkage L2 adaptation
                    delta = W - np.eye(k)
                    loss += self.l2 * (np.sum(delta ** 2) + np.sum(b ** 2)) / (2 * k * k)
                    grad_W += self.l2 * delta / (k * k)
                    grad_b += self.l2 * b / (k * k)
                return float(loss), np.concatenate((grad_W.ravel(), grad_b))

            initial = np.concatenate((np.eye(k).ravel(), np.zeros(k)))
            result = minimize(loss_grad, initial, jac=True, method="L-BFGS-B",
                              options={"maxiter": self.max_iter, "ftol": 1e-12})
            if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f"Dirichlet optimization failed: {result.message}")
            self.W = result.x[:k * k].reshape(k, k)
            self.b = result.x[k * k:]
            self.temperature = None
        self.n_classes = k
        self.fit_cutoff = _iso_z(cutoff_value)
        self.frozen = bool(frozen)
        return self

    def predict(self, probabilities: Any) -> np.ndarray:
        if self.W is None or self.b is None or self.n_classes is None:
            raise RuntimeError("calibrator is not fitted")
        p = self._probabilities(probabilities, clip=self.clip)
        if p.shape[1] != self.n_classes:
            raise ValueError("probability class count differs from fitted calibrator")
        logp = np.log(p)
        if self.method == "temperature":
            return _softmax(logp / float(self.temperature))
        return _softmax(logp @ self.W.T + self.b)

    def freeze(self) -> "DirichletCalibrator":
        if self.W is None:
            raise RuntimeError("cannot freeze an unfitted calibrator")
        self.frozen = True
        return self

    def to_dict(self) -> dict[str, Any]:
        if self.W is None or self.b is None:
            raise RuntimeError("cannot serialize an unfitted calibrator")
        return {"format": "dirichlet-calibrator-v1", "method": self.method,
                "clip": self.clip, "l2": self.l2, "odir_lambda": self.odir_lambda,
                "odir_mu": self.odir_mu, "max_iter": self.max_iter,
                "W": self.W.tolist(), "b": self.b.tolist(),
                "temperature": self.temperature, "n_classes": self.n_classes,
                "frozen": self.frozen, "fit_cutoff": self.fit_cutoff}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "DirichletCalibrator":
        if not isinstance(state, dict) or state.get("format") != "dirichlet-calibrator-v1":
            raise ValueError("unsupported or malformed calibrator state")
        obj = cls(state["method"], clip=state["clip"], l2=state["l2"],
                  odir_lambda=state["odir_lambda"], odir_mu=state["odir_mu"],
                  max_iter=state["max_iter"])
        W = np.asarray(state["W"], dtype=np.float64)
        b = np.asarray(state["b"], dtype=np.float64)
        k = int(state["n_classes"])
        if W.shape != (k, k) or b.shape != (k,) or not np.isfinite(W).all() or not np.isfinite(b).all():
            raise ValueError("invalid parameter dimensions or values in state")
        temperature = state.get("temperature")
        if obj.method == "temperature":
            if temperature is None or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature state must be positive and finite")
        elif temperature is not None:
            raise ValueError("temperature must be null except for temperature method")
        if not isinstance(state.get("frozen"), bool):
            raise ValueError("frozen must be a boolean")
        if state.get("fit_cutoff") is not None:
            _cutoff_ns([state["fit_cutoff"]], "fit_cutoff")
        obj.W, obj.b = W, b
        obj.temperature = None if temperature is None else float(temperature)
        obj.n_classes = k
        obj.frozen = state["frozen"]
        obj.fit_cutoff = state.get("fit_cutoff")
        return obj

    @classmethod
    def from_json(cls, payload: str) -> "DirichletCalibrator":
        try:
            state = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid calibrator JSON") from exc
        return cls.from_dict(state)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DirichletCalibrator":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
