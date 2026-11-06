"""Asymmetric split-conformal prediction intervals for regression.

Standard split conformal scores absolute residuals, so every interval is
symmetric around the point prediction. When residuals are skewed — a demand
forecast that under-predicts more often than it over-predicts, a latency
estimate whose tail is one-sided — that symmetry wastes width on the easy
side and under-covers on the hard side while still averaging to ``1 - alpha``.

:class:`AsymmetricSplitConformalRegressor` calibrates the lower and upper
tails of the *signed* residual separately. With equal split
``alpha_lo = alpha_hi = alpha / 2`` the finite-sample guarantee is still
``P(Y in [L, U]) >= 1 - alpha`` under exchangeability (Lei, G'Sell,
Rinaldo, Tibshirani & Wasserman 2018, §2.3; Romano, Patterson & Candès
2019, signed-residual variant). Unequal tails are allowed when one side
matters more; coverage then holds at ``1 - (alpha_lo + alpha_hi)``.
"""

from __future__ import annotations

import numpy as np

from .regression import conformal_quantile

__all__ = ["AsymmetricSplitConformalRegressor", "asymmetric_interval"]


def asymmetric_interval(y_pred, lower_q, upper_q, difficulty=None):
    """Build ``[pred + lower_q * s, pred + upper_q * s]`` for each prediction.

    ``lower_q`` is typically negative (the calibrated residual quantile of the
    lower tail) and ``upper_q`` positive. ``difficulty`` scales both sides the
    same way normalized split conformal does.
    """
    pred = np.asarray(y_pred, dtype=float).ravel()
    if difficulty is None:
        scale = np.ones(pred.size)
    else:
        scale = np.asarray(difficulty, dtype=float).ravel()
        if scale.size != pred.size:
            raise ValueError("difficulty must have the same length as the predictions")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("difficulty must be finite and strictly positive")
    lower = pred + float(lower_q) * scale
    upper = pred + float(upper_q) * scale
    return lower, upper


class AsymmetricSplitConformalRegressor:
    """Split conformal with separate lower and upper residual quantiles.

    ``fit`` takes signed residuals ``y - pred`` on a held-out calibration set
    and stores two finite-sample quantiles: the ``alpha_lo`` quantile of the
    residuals (lower tail) and the ``1 - alpha_hi`` quantile (upper tail).
    ``predict_interval`` then returns ``[pred + q_lo, pred + q_hi]``, optionally
    scaled by a per-point difficulty estimate.

    Parameters
    ----------
    alpha:
        Total miscoverage budget when ``alpha_lo`` / ``alpha_hi`` are left at
        ``None``. Split evenly across the two tails.
    alpha_lo, alpha_hi:
        Per-tail budgets. Must sum to a value in ``(0, 1)``. Override ``alpha``
        when set.
    normalize:
        Divide residuals by ``difficulty`` during calibration and re-scale the
        quantiles at prediction time, the same way
        :class:`~conformal_kit.regression.SplitConformalRegressor` does.
    """

    def __init__(self, alpha=0.1, alpha_lo=None, alpha_hi=None, normalize=False):
        if alpha_lo is None and alpha_hi is None:
            if not 0.0 < alpha < 1.0:
                raise ValueError("alpha must be strictly between 0 and 1")
            alpha_lo = alpha_hi = float(alpha) / 2.0
        elif alpha_lo is None or alpha_hi is None:
            raise ValueError("alpha_lo and alpha_hi must both be set or both left None")
        else:
            alpha_lo = float(alpha_lo)
            alpha_hi = float(alpha_hi)
            if not (0.0 < alpha_lo < 1.0 and 0.0 < alpha_hi < 1.0):
                raise ValueError("alpha_lo and alpha_hi must be strictly between 0 and 1")
            if alpha_lo + alpha_hi >= 1.0:
                raise ValueError("alpha_lo + alpha_hi must be strictly less than 1")
        self.alpha = float(alpha_lo + alpha_hi)
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.normalize = bool(normalize)
        self.lower_quantile_ = None
        self.upper_quantile_ = None
        self.n_calibration_ = 0

    def fit(self, y_true, y_pred, difficulty=None):
        """Calibrate lower and upper residual quantiles on held-out predictions."""
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        if observed.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")
        residuals = observed - predicted
        if self.normalize:
            if difficulty is None:
                raise ValueError("difficulty is required when normalize=True")
            scale = np.asarray(difficulty, dtype=float).ravel()
            if scale.size != residuals.size:
                raise ValueError("difficulty must have the same length as the calibration set")
            if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
                raise ValueError("difficulty must be finite and strictly positive")
            residuals = residuals / scale
        # Lower tail: conformal_quantile on -residuals at alpha_lo, then negate.
        # Upper tail: conformal_quantile on residuals at alpha_hi.
        self.lower_quantile_ = -conformal_quantile(-residuals, self.alpha_lo)
        self.upper_quantile_ = conformal_quantile(residuals, self.alpha_hi)
        self.n_calibration_ = int(residuals.size)
        return self

    def _check_fitted(self):
        if self.lower_quantile_ is None or self.upper_quantile_ is None:
            raise RuntimeError("call fit() before predict_interval()")

    def predict_interval(self, y_pred, difficulty=None):
        """Return ``(lower, upper)`` arrays for each prediction."""
        self._check_fitted()
        if self.normalize and difficulty is None:
            raise ValueError("difficulty is required when normalize=True")
        if not self.normalize and difficulty is not None:
            raise ValueError("difficulty is only used when normalize=True")
        return asymmetric_interval(
            y_pred, self.lower_quantile_, self.upper_quantile_, difficulty=difficulty
        )

    def interval_widths(self, y_pred, difficulty=None):
        """Return the width of each asymmetric interval."""
        lower, upper = self.predict_interval(y_pred, difficulty=difficulty)
        return upper - lower
