"""Localized split conformal prediction (kernel-weighted calibration).

Split conformal uses one quantile of the calibration scores for every test
point, so the interval has the same width everywhere -- too wide where the
model is accurate and too narrow where it is not. Localized conformal
prediction (Guan 2023, *Biometrika*) instead weights calibration score
``s_i`` by a localizer ``H(x_i, x) = exp(-||x_i - x||^2 / (2 h^2))`` centred
at the test point ``x`` (features standardized with calibration statistics)
and takes the weighted quantile with the test point's own weight
``H(x, x) = 1`` placed at ``+inf``, exactly as in weighted conformal
prediction. Intervals therefore adapt to how hard the neighbourhood of ``x``
is, without a separate difficulty model.

Plain localized quantiles at level ``1 - alpha`` need not reach marginal
``1 - alpha`` coverage, because the localizer changes with the test point.
Following the calibrated-LCP idea of Guan (2023), the level is tuned on the
calibration set: for every calibration point the localized quantile is
recomputed *leave-one-out* (point ``i`` plays the test point), the smallest
level ``q`` at which point ``i`` is covered is

    r_i = (localizer mass of scores strictly below s_i) / (W_{-i} + 1),

and the working level is the ``ceil((1 - alpha)(n + 1))``-th smallest
``r_i``, the usual finite-sample conformal rank. This calibration is
leave-one-out rather than the exact augmented construction of the paper, so
the marginal guarantee is approximate (it holds empirically to within
sampling error); with ``calibrate=False`` the raw level ``1 - alpha`` is
used. As ``h -> inf`` all weights become equal and the method reduces to
ordinary split conformal.
"""

from __future__ import annotations

import math

import numpy as np

from .weighted import weighted_conformal_quantile

__all__ = ["LocalizedConformalRegressor", "localizer_weights"]


def _as_matrix(X, name):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must all be finite")
    return values


def localizer_weights(X_calibration, X_test, bandwidth):
    """Gaussian localizer ``exp(-||x_i - x||^2 / (2 h^2))`` (calibration x test)."""
    A = _as_matrix(X_calibration, "X_calibration")
    B = _as_matrix(X_test, "X_test")
    if A.shape[1] != B.shape[1]:
        raise ValueError("X_calibration and X_test must have the same number of features")
    h = float(bandwidth)
    if not h > 0 or not math.isfinite(h):
        raise ValueError("bandwidth must be a positive finite number")
    sq = (
        np.sum(A * A, axis=1)[:, None]
        + np.sum(B * B, axis=1)[None, :]
        - 2.0 * A @ B.T
    )
    return np.exp(-np.maximum(sq, 0.0) / (2.0 * h * h))


class LocalizedConformalRegressor:
    """Kernel-localized split-conformal intervals around any point predictor.

    Parameters
    ----------
    alpha : float
        Target miscoverage.
    bandwidth : float, optional
        Localizer bandwidth on standardized features. Defaults to
        ``n^(-1 / (d + 4))`` (a Scott-type rule). Smaller means more local
        (more adaptive, but fewer effective calibration points).
    calibrate : bool
        Tune the working level by leave-one-out calibration (see module
        docstring); otherwise use ``1 - alpha`` directly.
    """

    def __init__(self, alpha=0.1, bandwidth=None, calibrate=True):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if bandwidth is not None and not float(bandwidth) > 0:
            raise ValueError("bandwidth must be positive")
        self.alpha = float(alpha)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.calibrate = bool(calibrate)
        self.scores_ = None

    def fit(self, X, y_true, y_pred):
        """Store calibration features and absolute-residual scores."""
        X = _as_matrix(X, "X")
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        n = X.shape[0]
        if observed.size != n or predicted.size != n:
            raise ValueError("X, y_true and y_pred must have the same number of rows")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        self.X_ = (X - self.mean_) / self.scale_
        self.scores_ = np.abs(observed - predicted)
        self.n_calibration_ = n
        self.bandwidth_ = (
            self.bandwidth if self.bandwidth is not None else float(n ** (-1.0 / (X.shape[1] + 4)))
        )
        if self.calibrate:
            self.level_ = self._calibrated_level()
        else:
            self.level_ = 1.0 - self.alpha
        return self

    def _calibrated_level(self):
        K = localizer_weights(self.X_, self.X_, self.bandwidth_)
        np.fill_diagonal(K, 0.0)  # leave point i out of its own calibration set
        s = self.scores_
        below = (s[None, :] < s[:, None]).astype(float)  # row i: scores strictly below s_i
        mass_below = np.sum(K * below, axis=1)
        r = mass_below / (K.sum(axis=1) + 1.0)
        n = s.size
        k = int(math.ceil((1.0 - self.alpha) * (n + 1)))
        self.loo_levels_ = r
        if k > n:
            return 1.0
        q = float(np.sort(r)[k - 1])
        # covered iff level > r_i: nudge just above the k-th smallest r_i
        return min(1.0, q + 1e-9)

    def _standardize(self, X):
        Z = _as_matrix(X, "X")
        if Z.shape[1] != self.X_.shape[1]:
            raise ValueError("X must have the same number of features as the calibration data")
        return (Z - self.mean_) / self.scale_

    def quantiles(self, X):
        """Localized score quantile for each test row."""
        if self.scores_ is None:
            raise RuntimeError("fit must be called before predicting")
        Z = self._standardize(X)
        if self.level_ >= 1.0:
            return np.full(Z.shape[0], np.inf)
        out = np.empty(Z.shape[0])
        for start in range(0, Z.shape[0], 512):
            K = localizer_weights(self.X_, Z[start:start + 512], self.bandwidth_)
            for j in range(K.shape[1]):
                out[start + j] = weighted_conformal_quantile(self.scores_, K[:, j], 1.0, 1.0 - self.level_)
        return out

    def effective_sample_sizes(self, X):
        """Kish effective number of calibration points behind each test row."""
        if self.scores_ is None:
            raise RuntimeError("fit must be called before predicting")
        Z = self._standardize(X)
        K = localizer_weights(self.X_, Z, self.bandwidth_)
        return K.sum(axis=0) ** 2 / np.maximum(np.sum(K * K, axis=0), 1e-300)

    def predict_interval(self, X, y_pred):
        """Return ``(lower, upper)`` arrays; intervals may be infinite far from the data."""
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predictions must all be finite")
        q = self.quantiles(X)
        if q.size != predicted.size:
            raise ValueError("X and y_pred must have the same number of rows")
        return predicted - q, predicted + q
