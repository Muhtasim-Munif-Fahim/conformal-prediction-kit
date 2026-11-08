"""Distributional conformal prediction (DCP) from a grid of predicted quantiles.

CQR conformalizes two quantiles. Many models (quantile forests, gradient
boosting with several quantile losses, distributional regression,
quantile networks) already predict a whole *grid* of conditional quantiles.
Distributional conformal prediction (Chernozhukov, Wüthrich & Zhu, PNAS 2021)
uses all of it: the grid defines an estimated conditional CDF ``F(y | x)`` and
the conformity score is the probability integral transform of the label,

* ``"dcp"``: ``|F(y | x) - 1/2|`` – symmetric in probability; the interval is
  ``[Q(1/2 - q), Q(1/2 + q)]``;
* ``"dcp-qr"``: ``max(tau_lo - F, F - tau_hi)`` – the CDF-scale analogue of
  CQR around the levels ``(tau_lo, tau_hi) = (alpha/2, 1 - alpha/2)``.

Because the score lives on the probability scale it is free of the label's
units, and intervals inherit the conditional *shape* (skew, heteroskedastic
spread) of the quantile model, while finite-sample marginal coverage
``>= 1 - alpha`` is restored by the usual split-conformal quantile.

The CDF between grid points is linear interpolation of ``(Q_k, tau_k)``; it is
0 below the smallest predicted quantile and 1 above the largest. Crossing
quantiles are repaired by sorting each row (monotone rearrangement,
Chernozhukov, Fernández-Val & Galichon 2010). Intervals are the exact
inversion of the score set ``{y : score(y) <= q}``; when a required level falls
outside the predicted grid the corresponding bound is the outermost predicted
quantile, or infinite if the level is at or beyond 0 / 1.
"""

from __future__ import annotations

import numpy as np

from .regression import conformal_quantile

__all__ = [
    "DistributionalConformalRegressor",
    "quantile_grid_cdf",
    "quantile_grid_inverse",
    "rearrange_quantiles",
]

_SCORES = ("dcp", "dcp-qr")


def _checked_levels(levels):
    taus = np.asarray(levels, dtype=float).ravel()
    if taus.size < 2:
        raise ValueError("at least two quantile levels are required")
    if not np.all((taus > 0.0) & (taus < 1.0)):
        raise ValueError("quantile levels must lie strictly between 0 and 1")
    if np.any(np.diff(taus) <= 0):
        raise ValueError("quantile levels must be strictly increasing")
    return taus


def rearrange_quantiles(quantiles):
    """Sort each row so predicted quantiles are non-decreasing (fixes crossing)."""
    Q = np.asarray(quantiles, dtype=float)
    if Q.ndim != 2:
        raise ValueError("quantile predictions must be a 2-D array (n_samples, n_levels)")
    if not np.all(np.isfinite(Q)):
        raise ValueError("quantile predictions must all be finite")
    return np.sort(Q, axis=1)


def _checked_grid(quantiles, levels):
    taus = _checked_levels(levels)
    Q = rearrange_quantiles(quantiles)
    if Q.shape[1] != taus.size:
        raise ValueError("number of quantile columns must match the number of levels")
    if Q.shape[0] == 0:
        raise ValueError("quantile predictions must not be empty")
    return Q, taus


def quantile_grid_cdf(y, quantiles, levels):
    """Estimated ``F(y_i | x_i)`` from each row's quantile grid.

    Linear between grid points, 0 below the first predicted quantile and 1
    above the last. Ties in a row (flat quantiles) take the largest level.
    """
    Q, taus = _checked_grid(quantiles, levels)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != Q.shape[0]:
        raise ValueError("y and the quantile predictions must have the same length")
    out = np.empty(y.size)
    for i in range(y.size):
        q = Q[i]
        if y[i] < q[0]:
            out[i] = 0.0
        elif y[i] >= q[-1]:
            out[i] = 1.0
        else:
            # right-continuous: last grid point <= y
            k = int(np.searchsorted(q, y[i], side="right")) - 1
            lo, hi = q[k], q[k + 1]
            frac = 0.0 if hi == lo else (y[i] - lo) / (hi - lo)
            out[i] = taus[k] + frac * (taus[k + 1] - taus[k])
    return out


def quantile_grid_inverse(level, quantiles, levels, side):
    """Inverse of :func:`quantile_grid_cdf` for one probability level per row.

    ``side="lower"`` returns ``inf{y : F(y) >= level}`` and ``side="upper"``
    returns ``sup{y : F(y) <= level}``; levels outside the grid map to the
    outermost quantile or to ``-inf`` / ``+inf`` at 0 / 1.
    """
    Q, taus = _checked_grid(quantiles, levels)
    lv = np.broadcast_to(np.asarray(level, dtype=float), (Q.shape[0],))
    out = np.empty(Q.shape[0])
    for i in range(Q.shape[0]):
        t = lv[i]
        if side == "lower":
            if t <= 0.0:
                out[i] = -np.inf
            elif t <= taus[0]:
                out[i] = Q[i, 0]
            elif t > taus[-1]:
                out[i] = Q[i, -1]
            else:
                out[i] = np.interp(t, taus, Q[i])
        elif side == "upper":
            if t >= 1.0:
                out[i] = np.inf
            elif t >= taus[-1]:
                out[i] = Q[i, -1]
            elif t < taus[0]:
                out[i] = Q[i, 0]
            else:
                out[i] = np.interp(t, taus, Q[i])
        else:
            raise ValueError("side must be 'lower' or 'upper'")
    return out


class DistributionalConformalRegressor:
    """Split DCP intervals from a grid of predicted conditional quantiles.

    Parameters
    ----------
    levels : sequence of float
        Strictly increasing quantile levels in ``(0, 1)`` matching the columns
        of the quantile predictions (e.g. ``np.linspace(0.05, 0.95, 19)``).
    alpha : float
        Miscoverage level of the conformal step.
    score : {"dcp", "dcp-qr"}
        ``|F - 1/2|`` or the CDF-scale CQR score around
        ``(alpha / 2, 1 - alpha / 2)``.

    Notes
    -----
    The grid should reach well into the tails (``levels[0] < alpha / 2`` and
    ``levels[-1] > 1 - alpha / 2``). Every label outside the predicted grid
    gets the same maximal score, so when more than ``alpha`` of the
    calibration labels fall outside, the calibrated band reaches 0 / 1 and
    the intervals become infinite (valid but useless). ``fit`` records that
    share as ``outside_grid_fraction_``.
    """

    def __init__(self, levels, alpha=0.1, score="dcp"):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if score not in _SCORES:
            raise ValueError(f"score must be one of {_SCORES}")
        self.levels = _checked_levels(levels)
        self.alpha = float(alpha)
        self.score = score
        self.quantile_ = None
        self.n_calibration_ = 0

    def scores(self, y_true, quantiles):
        """Conformity scores of labels under the predicted conditional CDFs."""
        F = quantile_grid_cdf(y_true, quantiles, self.levels)
        if self.score == "dcp":
            return np.abs(F - 0.5)
        lo, hi = self.alpha / 2.0, 1.0 - self.alpha / 2.0
        return np.maximum(lo - F, F - hi)

    def fit(self, y_true, quantiles):
        """Calibrate on held-out labels and their predicted quantile grids."""
        y = np.asarray(y_true, dtype=float).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("calibration values must all be finite")
        s = self.scores(y, quantiles)
        self.quantile_ = conformal_quantile(s, self.alpha)
        self.n_calibration_ = int(s.size)
        self.calibration_scores_ = np.sort(s)
        F = quantile_grid_cdf(y, quantiles, self.levels)
        # Labels outside the grid all share the maximal score; if they make up
        # more than alpha of the calibration set the interval becomes infinite.
        self.outside_grid_fraction_ = float(np.mean((F <= 0.0) | (F >= 1.0)))
        return self

    def _probability_band(self):
        q = self.quantile_
        if self.score == "dcp":
            return 0.5 - q, 0.5 + q
        return self.alpha / 2.0 - q, 1.0 - self.alpha / 2.0 + q

    def predict_interval(self, quantiles):
        """Return ``(lower, upper)``: the exact inversion of ``score <= q``."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        lo_level, hi_level = self._probability_band()
        lower = quantile_grid_inverse(lo_level, quantiles, self.levels, "lower")
        upper = quantile_grid_inverse(hi_level, quantiles, self.levels, "upper")
        return lower, upper

    def contains(self, y, quantiles):
        """Boolean membership ``score(y) <= q`` (equivalent to the interval)."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before contains")
        return self.scores(y, quantiles) <= self.quantile_ + 1e-12

    def p_values(self, y, quantiles):
        """Split-conformal p-values ``(1 + #{cal scores >= s}) / (n + 1)`` for candidate labels."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before p_values")
        s = self.scores(y, quantiles)
        cal = self.calibration_scores_
        ge = cal.size - np.searchsorted(cal, s - 1e-12, side="left")
        return (1.0 + ge) / (cal.size + 1.0)
