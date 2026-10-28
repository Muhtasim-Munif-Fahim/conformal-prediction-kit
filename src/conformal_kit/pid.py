"""Conformal PID control for streaming scores (Angelopoulos, Candes & Tibshirani 2023).

ACI (:mod:`conformal_kit.aci`) adapts the *level* ``alpha_t`` and reads a
quantile of a score window. Conformal PID control ("Conformal PID Control
for Time Series Prediction", NeurIPS 2023) instead treats the score
threshold ``q_t`` itself as the controlled quantity and steers it with
three terms, borrowed from feedback control:

* **P -- quantile tracking**: online gradient descent on the pinball loss,
  ``qtrack_{t+1} = qtrack_t + lr * (err_t - alpha)``, where ``err_t`` is 1
  when the score exceeded the threshold. A miss widens, a hit shrinks.
* **I -- error integration**: ``r_t(sum_{i<=t} (err_i - alpha))`` with the
  saturation function ``r_t(x) = K_I * tan(x * log(t) / (t * C_sat))``
  (``+-inf`` once the argument reaches ``+-pi/2``). It reacts to a *run*
  of misses much faster than the P term and, because it blows up when the
  coverage error is too large, gives a hard long-run coverage guarantee.
* **D -- scorecasting**: an optional forecast ``s_hat_{t+1}`` of the next
  score from the past ones (any callable). Seasonality or trend in the
  residuals is then anticipated instead of chased.

The threshold issued at time ``t + 1`` is
``q_{t+1} = qtrack_{t+1} + r_t(...) + s_hat_{t+1}``.

For any score sequence bounded by ``B`` the quantile tracker alone
satisfies ``|mean(err_1..T) - alpha| <= (B + lr) / (lr * T)`` -- no
exchangeability, no stationarity. The guarantee holds for *any* stream,
including adversarial ones.

:func:`conformal_pid` runs the controller on an arbitrary score stream
(regression residuals, classification scores, anything), and
:class:`ConformalPIDRegressor` wraps it into the ``fit`` /
``predict_update`` API used by :class:`~conformal_kit.aci.AdaptiveConformalRegressor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .regression import conformal_quantile

__all__ = [
    "PIDResult",
    "ConformalPIDRegressor",
    "conformal_pid",
    "default_c_sat",
    "ewma_scorecaster",
    "pid_saturation",
    "quantile_tracker_update",
]


def _checked_alpha(alpha):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return alpha


def _checked_positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")
    return value


def quantile_tracker_update(q, err, alpha, lr):
    """One P step: ``q + lr * (err - alpha)`` (pinball-loss gradient step)."""
    if err not in (0, 1, 0.0, 1.0, True, False):
        raise ValueError("err must be 0 or 1")
    return float(q) + float(lr) * (float(err) - _checked_alpha(alpha))


def pid_saturation(x, t, k_i=1.0, c_sat=None):
    """Integrator ``K_I * tan(x * log(t) / (t * C_sat))``; ``+-inf`` past ``+-pi/2``.

    ``t`` is the number of outcomes observed so far; ``t <= 1`` gives 0.
    """
    k_i = _checked_positive(k_i, "k_i", allow_zero=True)
    if c_sat is None:
        raise ValueError("c_sat is required")
    c_sat = _checked_positive(c_sat, "c_sat")
    if t <= 1 or k_i == 0.0 or x == 0:
        return 0.0
    arg = float(x) * math.log(t) / (t * c_sat)
    if abs(arg) >= math.pi / 2:
        return math.copysign(math.inf, arg)
    return k_i * math.tan(arg)


def default_c_sat(alpha, horizon):
    """Conservative default ``C_sat = max(2 / pi * (ceil(log T) - 1 / log T), 1)``.

    With this constant the integrator stays small until the cumulative
    coverage error is large relative to ``T / log(T)``. Pass a smaller
    ``c_sat`` (or a larger ``k_i``) for a more aggressive I term.
    """
    _checked_alpha(alpha)
    horizon = max(int(horizon), 3)
    log_t = math.log(horizon)
    return max(2.0 / math.pi * (math.ceil(log_t) - 1.0 / log_t), 1.0)


def ewma_scorecaster(halflife=10.0):
    """Return a scorecaster forecasting the next score as an EWMA of past scores."""
    halflife = _checked_positive(halflife, "halflife")
    decay = 0.5 ** (1.0 / halflife)

    def forecast(past):
        past = np.asarray(past, dtype=float)
        if past.size == 0:
            return 0.0
        w = decay ** np.arange(past.size - 1, -1, -1, dtype=float)
        return float(np.dot(w, past) / w.sum())

    return forecast


@dataclass(frozen=True)
class PIDResult:
    """Thresholds issued *before* each score and whether each score was covered."""

    thresholds: np.ndarray
    covered: np.ndarray
    alpha: float

    @property
    def miscoverage(self) -> float:
        return float(1.0 - self.covered.mean()) if self.covered.size else float("nan")

    def rolling_miscoverage(self, window=50) -> np.ndarray:
        """Miscoverage over a trailing window (shorter at the start)."""
        if int(window) < 1:
            raise ValueError("window must be >= 1")
        miss = 1.0 - self.covered.astype(float)
        c = np.concatenate([[0.0], np.cumsum(miss)])
        idx = np.arange(1, miss.size + 1)
        lo = np.maximum(idx - int(window), 0)
        return (c[idx] - c[lo]) / (idx - lo)


class _Controller:
    def __init__(self, alpha, lr, k_i, c_sat, scorecaster, q0):
        self.alpha = _checked_alpha(alpha)
        self.lr = _checked_positive(lr, "lr")
        self.k_i = _checked_positive(k_i, "k_i", allow_zero=True)
        self.c_sat = None if c_sat is None else _checked_positive(c_sat, "c_sat")
        if scorecaster is not None and not callable(scorecaster):
            raise TypeError("scorecaster must be callable or None")
        self.scorecaster = scorecaster
        self.qtrack = float(q0)
        self.err_sum = 0.0
        self.t = 0
        self.scores: list = []

    def threshold(self):
        q = self.qtrack
        if self.k_i > 0 and self.c_sat is not None:
            q += pid_saturation(self.err_sum, self.t, self.k_i, self.c_sat)
        if self.scorecaster is not None:
            q += float(self.scorecaster(np.asarray(self.scores, dtype=float)))
        return q

    def observe(self, score):
        q = self.threshold()
        covered = bool(score <= q)
        err = 0.0 if covered else 1.0
        self.qtrack = quantile_tracker_update(self.qtrack, err, self.alpha, self.lr)
        self.err_sum += err - self.alpha
        self.t += 1
        self.scores.append(float(score))
        return q, covered


def conformal_pid(scores, alpha=0.1, lr=0.05, k_i=0.0, c_sat=None, scorecaster=None, q0=None):
    """Run conformal PID control over a stream of nonconformity scores.

    Parameters
    ----------
    scores : array-like
        Scores in arrival order (e.g. absolute residuals). Larger = worse.
    alpha : float
        Target long-run miscoverage.
    lr : float
        Quantile-tracker step size (same units as the scores). Rule of
        thumb: ~0.1 x the score scale.
    k_i : float
        Integrator gain; 0 disables the I term.
    c_sat : float, optional
        Saturation constant; defaults to :func:`default_c_sat` for the
        stream length when ``k_i > 0``.
    scorecaster : callable, optional
        ``f(past_scores) -> forecast`` for the D term, e.g.
        :func:`ewma_scorecaster`.
    q0 : float, optional
        Initial tracker value (default 0). Seed it with a calibration
        quantile to avoid a warm-up period of misses.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores must not be empty")
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")
    if k_i and c_sat is None:
        c_sat = default_c_sat(alpha, s.size)
    ctrl = _Controller(alpha, lr, k_i, c_sat, scorecaster, 0.0 if q0 is None else q0)
    qs = np.empty(s.size)
    cov = np.empty(s.size, dtype=bool)
    for i, score in enumerate(s):
        qs[i], cov[i] = ctrl.observe(score)
    return PIDResult(thresholds=qs, covered=cov, alpha=ctrl.alpha)


class ConformalPIDRegressor:
    """Streaming residual intervals ``y_pred +- q_t`` steered by conformal PID control.

    ``fit`` seeds the tracker with the split-conformal quantile of held-out
    absolute residuals (optional; without it the tracker starts at 0 and
    learns the scale online). :meth:`predict_update` issues one interval per
    point *before* seeing its label, then updates. A negative threshold is
    reported as the degenerate interval ``[y_pred, y_pred]`` but still counts
    as a miss, matching the controller's ``score <= q_t`` rule.
    """

    def __init__(self, alpha=0.1, lr=0.05, k_i=0.0, c_sat=None, scorecaster=None):
        self.alpha = _checked_alpha(alpha)
        self.lr = _checked_positive(lr, "lr")
        self.k_i = _checked_positive(k_i, "k_i", allow_zero=True)
        self.c_sat = c_sat
        self.scorecaster = scorecaster
        self._ctrl = None
        self.covered_: list = []

    def _controller(self, q0=0.0, horizon=1000):
        c_sat = self.c_sat
        if self.k_i and c_sat is None:
            c_sat = default_c_sat(self.alpha, horizon)
        return _Controller(self.alpha, self.lr, self.k_i, c_sat, self.scorecaster, q0)

    def fit(self, y_true, y_pred):
        """Seed the tracker at the split-conformal quantile of ``|y_true - y_pred|``."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        residuals = np.abs(y_true - y_pred)
        q0 = 0.0 if self.scorecaster is not None else float(conformal_quantile(residuals, self.alpha))
        self._ctrl = self._controller(q0, horizon=max(residuals.size, 1000))
        if self.scorecaster is not None:
            self._ctrl.scores = residuals.tolist()
        self.covered_ = []
        return self

    @property
    def threshold_(self):
        """Threshold that would be issued for the next point."""
        if self._ctrl is None:
            self._ctrl = self._controller()
        return self._ctrl.threshold()

    def predict_update(self, y_pred, y_true):
        """Issue ``(lower, upper)`` for each point, then observe its label and update."""
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if not (np.all(np.isfinite(y_pred)) and np.all(np.isfinite(y_true))):
            raise ValueError("y_true and y_pred must be finite")
        if self._ctrl is None:
            self._ctrl = self._controller(horizon=max(y_pred.size, 1000))
        lower = np.empty(y_pred.size)
        upper = np.empty(y_pred.size)
        for i in range(y_pred.size):
            q, covered = self._ctrl.observe(abs(y_true[i] - y_pred[i]))
            half = max(q, 0.0)
            lower[i], upper[i] = y_pred[i] - half, y_pred[i] + half
            self.covered_.append(covered)
        return lower, upper

    @property
    def empirical_miscoverage(self):
        """Fraction of updated points that were missed, or ``None`` before any update."""
        if not self.covered_:
            return None
        return 1.0 - float(np.mean(self.covered_))
