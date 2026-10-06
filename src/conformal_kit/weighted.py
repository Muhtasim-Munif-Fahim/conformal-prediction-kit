"""Weighted split conformal prediction under covariate shift.

Split conformal assumes calibration and test points are exchangeable. Under
**covariate shift** that fails in a specific, fixable way: the distribution of
the features ``X`` changes between calibration and deployment, but the
conditional distribution of ``Y`` given ``X`` does not. Tibshirani, Barber,
Candes and Ramdas (2019) showed that coverage is restored by reweighting the
calibration scores with the likelihood ratio

    w(x) = dP_test(x) / dP_calibration(x)

and taking a *weighted* quantile. Each calibration score ``s_i`` gets mass
``w(x_i)``, the test point gets mass ``w(x_test)`` placed at ``+inf``, and the
interval half-width is the ``1 - alpha`` quantile of that weighted
distribution. With a known (or well-estimated) ratio the interval covers at
least ``1 - alpha`` on the shifted test distribution.

Two consequences are worth knowing:

* The quantile depends on the test point's own weight, so every test point
  can get its own width -- wider where the test distribution has moved into
  regions the calibration set barely covers.
* When a test point's weight is large relative to the total calibration
  weight, the ``+inf`` atom alone can exceed ``alpha`` and the honest answer
  is an infinite interval. That is a feature: it flags extrapolation instead
  of silently under-covering.

The ratio is usually estimated with a domain classifier trained to tell
calibration rows from test rows; see :func:`likelihood_ratio_from_probabilities`.
With all weights equal this reduces to ordinary split conformal.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WeightedConformalRegressor",
    "effective_sample_size",
    "likelihood_ratio_from_probabilities",
    "weighted_conformal_quantile",
]


def _check_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(alpha)


def _checked_weights(weights, expected=None, name="weights"):
    values = np.asarray(weights, dtype=float).ravel()
    if expected is not None and values.size != expected:
        raise ValueError(f"{name} must have the same length as the {_LENGTH_HINT[name]}")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must all be finite")
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    return values


_LENGTH_HINT = {
    "weights": "scores",
    "calibration_weights": "calibration set",
    "test_weights": "predictions",
}


def effective_sample_size(weights):
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of a weight vector.

    Equals ``n`` for equal weights and shrinks toward 1 as a few points
    dominate. A small value relative to the calibration size means the
    shift is severe and weighted intervals will be wide (or infinite).
    """
    values = _checked_weights(weights)
    total = float(values.sum())
    if total == 0.0:
        raise ValueError("weights must not all be zero")
    return total * total / float(np.sum(values * values))


def likelihood_ratio_from_probabilities(prob_test, n_calibration, n_test, clip=None):
    """Turn domain-classifier probabilities into covariate-shift weights.

    Train any probabilistic classifier to predict whether a row comes from
    the test pool (label 1) or the calibration pool (label 0), on the
    pooled features. For a row with predicted ``p = P(test | x)``, Bayes'
    rule gives the likelihood ratio

        w(x) = p / (1 - p) * n_calibration / n_test

    where the last factor undoes the class imbalance of the classifier's
    training set. ``clip`` optionally caps ``p`` at ``1 - clip`` and floors
    it at ``clip`` so a saturated classifier cannot produce infinite
    weights; ``clip=None`` rejects ``p`` of exactly 0 or 1 instead.
    """
    p = np.asarray(prob_test, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("prob_test must not be empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("prob_test must all be finite")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("prob_test must lie in [0, 1]")
    for label, n in (("n_calibration", n_calibration), ("n_test", n_test)):
        if isinstance(n, bool) or not float(n) > 0:
            raise ValueError(f"{label} must be positive")
    if clip is not None:
        if not 0.0 < clip < 0.5:
            raise ValueError("clip must be strictly between 0 and 0.5")
        p = np.clip(p, clip, 1.0 - clip)
    elif np.any((p <= 0) | (p >= 1)):
        raise ValueError("prob_test of exactly 0 or 1 gives an infinite or zero ratio; pass clip")
    return p / (1.0 - p) * (float(n_calibration) / float(n_test))


def weighted_conformal_quantile(scores, calibration_weights, test_weights, alpha):
    """Weighted ``1 - alpha`` conformal quantile for one or more test points.

    For each test weight ``w_t`` the calibration scores ``s_i`` receive
    normalized mass ``w_i / (W + w_t)`` (``W`` the total calibration weight)
    and ``+inf`` receives ``w_t / (W + w_t)``. The result is the smallest
    score whose cumulative mass reaches ``1 - alpha``, or ``inf`` when only
    the ``+inf`` atom gets there.

    Returns a float when ``test_weights`` is a scalar, else an array with
    one quantile per test point. With all weights equal and enough
    calibration points this equals
    :func:`~conformal_kit.regression.conformal_quantile`.
    """
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must all be finite")
    alpha = _check_alpha(alpha)
    cal_w = _checked_weights(calibration_weights, values.size, "calibration_weights")
    scalar = np.ndim(test_weights) == 0
    test_w = _checked_weights(np.atleast_1d(test_weights), name="test_weights")

    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    cumulative = np.cumsum(cal_w[order])
    total = float(cumulative[-1])
    if total == 0.0 and np.any(test_w == 0.0):
        raise ValueError("calibration and test weights must not all be zero")

    # Reach 1 - alpha of the total mass W + w_t using calibration mass only.
    target = (1.0 - alpha) * (total + test_w)
    tolerance = 1e-12 * (total + test_w)
    index = np.searchsorted(cumulative, target - tolerance, side="left")
    quantiles = np.full(test_w.size, np.inf)
    finite = index < values.size
    quantiles[finite] = sorted_scores[index[finite]]
    if scalar:
        return float(quantiles[0])
    return quantiles


class WeightedConformalRegressor:
    """Split-conformal intervals that stay valid under covariate shift.

    Same contract as :class:`~conformal_kit.regression.SplitConformalRegressor`
    -- it wraps an already-fitted model and only sees true values and
    predictions -- plus a likelihood-ratio weight per point. ``fit`` takes
    the calibration weights ``w(x_i)``; ``predict_interval`` takes the test
    weights ``w(x_test)`` and returns one interval per test point, whose
    width depends on that point's weight.

    ``normalize=True`` divides residuals by a per-point difficulty estimate,
    exactly as in split conformal, so the two adaptations combine: weights
    correct *where* the test data lives, difficulty scales *how wide* each
    interval is.

    Intervals can be infinite. That happens when a test point's weight is
    so large relative to the calibration weight that no finite width can
    certify ``1 - alpha`` coverage -- typically a test point far outside
    the calibration support. ``effective_n_`` (Kish effective sample size
    of the calibration weights) is a quick read on how much the shift
    costs.
    """

    def __init__(self, alpha=0.1, normalize=False):
        self.alpha = _check_alpha(alpha)
        self.normalize = bool(normalize)
        self.scores_ = None
        self.weights_ = None
        self.n_calibration_ = 0
        self.effective_n_ = 0.0

    def fit(self, y_true, y_pred, weights, difficulty=None):
        """Store weighted nonconformity scores from a held-out calibration set."""
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        if observed.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")
        cal_w = _checked_weights(weights, observed.size, "calibration_weights")
        if float(cal_w.sum()) == 0.0:
            raise ValueError("calibration_weights must not all be zero")

        residuals = np.abs(observed - predicted)
        if self.normalize:
            residuals = residuals / self._checked_difficulty(difficulty, observed.size)
        elif difficulty is not None:
            raise ValueError("difficulty requires normalize=True")

        self.scores_ = residuals
        self.weights_ = cal_w
        self.n_calibration_ = int(observed.size)
        self.effective_n_ = effective_sample_size(cal_w)
        return self

    def quantiles(self, weights):
        """Per-test-point score quantiles for the given test weights."""
        if self.scores_ is None:
            raise RuntimeError("fit must be called before predicting")
        return weighted_conformal_quantile(
            self.scores_, self.weights_, np.atleast_1d(np.asarray(weights, dtype=float)),
            self.alpha,
        )

    def predict_interval(self, y_pred, weights, difficulty=None):
        """Return ``(lower, upper)`` arrays for new predictions and their weights."""
        if self.scores_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predictions must all be finite")
        test_w = _checked_weights(weights, predicted.size, "test_weights")
        quantiles = self.quantiles(test_w)

        if self.normalize:
            half_width = quantiles * self._checked_difficulty(difficulty, predicted.size)
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            half_width = quantiles
        return predicted - half_width, predicted + half_width

    @staticmethod
    def _checked_difficulty(difficulty, expected):
        if difficulty is None:
            raise ValueError("normalize=True requires difficulty estimates")
        scale = np.asarray(difficulty, dtype=float).ravel()
        if scale.size != expected:
            raise ValueError("difficulty must have the same length as the predictions")
        if not np.all(np.isfinite(scale)):
            raise ValueError("difficulty estimates must all be finite")
        if np.any(scale <= 0):
            raise ValueError("difficulty estimates must be strictly positive")
        return scale
