"""Conformalized quantile regression (CQR) prediction intervals.

Split conformal around a point prediction gives every test point the same
width. That is valid but blunt when the noise level changes with ``x``:
the interval is too wide in quiet regions and too narrow in noisy ones.

CQR (Romano, Patterson, Candes 2019) conformalizes a *pair* of quantile
predictions instead. A quantile-regression model already proposes a lower
and an upper bound that can vary with ``x``; CQR measures how far the
calibration labels fall outside those bounds and expands (or, if the
quantile model already over-covers, shrinks) every new interval by the
same finite-sample quantile of that score. Marginal coverage becomes
``1 - alpha``; the width still follows the quantile model.

sklearn is not required. Train any quantile regressor yourself and pass
the lower and upper predictions — the same wrap-predictions shape as
:class:`~conformal_kit.regression.SplitConformalRegressor`.
"""

from __future__ import annotations

import numpy as np

from .regression import conformal_quantile

__all__ = ["ConformalizedQuantileRegressor", "cqr_scores"]


def cqr_scores(y_true, y_lower, y_upper):
    """Return CQR conformity scores ``max(lower - y, y - upper)``.

    A positive score is how far the label fell outside the predicted
    quantile interval; a negative score is how far inside it landed.
    The finite-sample quantile of these scores is the amount CQR adds
    to each side of a new prediction. When the quantile model is already
    conservative the quantile can be negative, and the intervals shrink.

    If the two bounds are identical the score reduces to the absolute
    residual, and CQR reduces to split conformal around a point prediction.
    """
    observed, lower, upper = _checked_observations(y_true, y_lower, y_upper)
    return np.maximum(lower - observed, observed - upper)


def _checked_bounds(y_lower, y_upper):
    lower = np.asarray(y_lower, dtype=float).ravel()
    upper = np.asarray(y_upper, dtype=float).ravel()
    if lower.size != upper.size:
        raise ValueError("y_lower and y_upper must have the same length")
    if lower.size == 0:
        raise ValueError("quantile predictions must not be empty")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("quantile predictions must all be finite")
    if np.any(upper < lower):
        raise ValueError("y_upper must be greater than or equal to y_lower")
    return lower, upper


def _checked_observations(y_true, y_lower, y_upper):
    observed = np.asarray(y_true, dtype=float).ravel()
    lower, upper = _checked_bounds(y_lower, y_upper)
    if observed.size != lower.size:
        raise ValueError("y_true and the quantile predictions must have the same length")
    if not np.all(np.isfinite(observed)):
        raise ValueError("calibration values must all be finite")
    return observed, lower, upper


class ConformalizedQuantileRegressor:
    """Calibrate quantile-regression bounds to a finite-sample coverage level.

    Wraps an already-fitted quantile model: it never sees features and never
    trains. ``fit`` takes the true values and the model's lower and upper
    quantile predictions on a calibration set the model did not train on;
    ``predict_interval`` applies the calibrated expansion to new quantile
    predictions.

    Train the quantile model at ``alpha / 2`` and ``1 - alpha / 2`` (or any
    other pair of levels whose interval you then want to conformalize).
    The ``alpha`` here is the coverage level of the *conformal* step, not
    the quantile levels themselves.

    The score on a calibration point is how far the label sits outside
    ``[lower, upper]``. Expanding both sides by the finite-sample
    ``1 - alpha`` quantile of those scores is what restores the coverage
    guarantee, even when the quantile model is misspecified.

    When you have a point predictor rather than a quantile model, see
    :class:`~conformal_kit.regression.SplitConformalRegressor`. When the
    noise level is known as a per-point difficulty estimate, the
    normalized form of that class is the cheaper adaptive alternative.
    """

    def __init__(self, alpha=0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self.quantile_ = None
        self.n_calibration_ = 0

    def fit(self, y_true, y_lower, y_upper):
        """Calibrate the interval expansion on held-out quantile predictions."""
        scores = cqr_scores(y_true, y_lower, y_upper)
        self.quantile_ = conformal_quantile(scores, self.alpha)
        self.n_calibration_ = int(scores.size)
        return self

    def predict_interval(self, y_lower, y_upper):
        """Return ``(lower, upper)`` arrays for new quantile predictions."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        lower, upper = _checked_bounds(y_lower, y_upper)
        return lower - self.quantile_, upper + self.quantile_
