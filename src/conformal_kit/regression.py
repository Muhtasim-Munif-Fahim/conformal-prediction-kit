"""Split-conformal prediction intervals for regression.

Split (inductive) conformal prediction holds out a calibration set the model
never trained on, scores how wrong the model was on each calibration point,
and takes a quantile of those scores as the interval half-width. Because the
quantile is computed from real errors rather than an assumed error
distribution, the resulting interval covers the truth at least ``1 - alpha``
of the time for any model and any data distribution, provided the calibration
and test points are exchangeable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MondrianSplitConformalRegressor", "SplitConformalRegressor", "conformal_quantile", "mondrian_regression_quantiles"]


def conformal_quantile(scores, alpha):
    """Return the finite-sample-corrected ``1 - alpha`` quantile of scores.

    The correction is what separates conformal prediction from taking an
    empirical quantile. With ``n`` calibration scores the quantile is taken at
    rank ``ceil((n + 1) * (1 - alpha))`` rather than at ``1 - alpha`` of the
    way through, which is what makes the coverage guarantee hold at finite
    ``n`` instead of only asymptotically. Without it, coverage sits just below
    the target for small calibration sets.

    Raises ``ValueError`` when the calibration set is too small to support the
    requested level -- with ``n < 1/alpha - 1`` the required rank exceeds the
    number of scores and no finite quantile can guarantee the coverage.
    """
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must all be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    n = values.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        needed = int(np.ceil(1.0 / alpha)) - 1
        raise ValueError(
            f"alpha={alpha} needs at least {needed} calibration points, got {n}"
        )
    # Rank is one-based; np.partition is zero-based.
    return float(np.partition(values, rank - 1)[rank - 1])


class SplitConformalRegressor:
    """Turn point predictions into intervals with a coverage guarantee.

    Wraps an already-fitted model: it never sees features and never trains.
    ``fit`` takes the true values and the model's predictions on a calibration
    set the model did not train on, and ``predict_interval`` applies the
    calibrated width to new predictions.

    With ``normalize=False`` (the default) the score is the absolute residual
    and every interval has the same width. That is correct but blunt on
    heteroskedastic data, where it is too wide in easy regions and too narrow
    in hard ones -- while still covering on average, which is exactly what
    makes the failure easy to miss.

    With ``normalize=True`` the score is the residual divided by a per-point
    difficulty estimate (a predicted standard deviation, an ensemble spread,
    anything positive that tracks uncertainty), so the width scales with it.
    Marginal coverage is unchanged; the intervals just distribute their width
    where it is needed.

    When the model is a quantile regressor rather than a point predictor,
    see :class:`~conformal_kit.cqr.ConformalizedQuantileRegressor`, which
    conformalizes the lower and upper bounds directly. When holding out a
    calibration set is too expensive, see
    :class:`~conformal_kit.jackknife.JackknifePlusRegressor`, which refits
    leave-one-out or K-fold so every point still trains a model. When the
    stream is not exchangeable, see
    :class:`~conformal_kit.aci.AdaptiveConformalRegressor`, which adapts
    the miscoverage level online so long-run coverage tracks ``1 - alpha``.
    """

    def __init__(self, alpha=0.1, normalize=False):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self.normalize = bool(normalize)
        self.quantile_ = None
        self.n_calibration_ = 0

    def fit(self, y_true, y_pred, difficulty=None):
        """Calibrate the interval width on held-out predictions."""
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        if observed.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")

        residuals = np.abs(observed - predicted)
        if self.normalize:
            scale = self._checked_difficulty(difficulty, observed.size)
            scores = residuals / scale
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            scores = residuals

        self.quantile_ = conformal_quantile(scores, self.alpha)
        self.n_calibration_ = int(observed.size)
        return self

    def predict_interval(self, y_pred, difficulty=None):
        """Return ``(lower, upper)`` arrays for new point predictions."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predictions must all be finite")

        if self.normalize:
            scale = self._checked_difficulty(difficulty, predicted.size)
            half_width = self.quantile_ * scale
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            half_width = np.full(predicted.size, self.quantile_)

        return predicted - half_width, predicted + half_width

    @property
    def width(self):
        """Interval width for the unnormalized case, else the base width."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before reading width")
        return 2.0 * self.quantile_

    @staticmethod
    def _checked_difficulty(difficulty, expected):
        if difficulty is None:
            raise ValueError("normalize=True requires difficulty estimates")
        scale = np.asarray(difficulty, dtype=float).ravel()
        if scale.size != expected:
            raise ValueError("difficulty must have the same length as the predictions")
        if not np.all(np.isfinite(scale)):
            raise ValueError("difficulty estimates must all be finite")
        # A zero would divide a residual to infinity and blow up the quantile.
        if np.any(scale <= 0):
            raise ValueError("difficulty estimates must be strictly positive")
        return scale


def mondrian_regression_quantiles(scores, groups, alpha):
    """Return one finite-sample conformal quantile per discrete group.

    Partitions absolute (or normalized) residuals by a discrete group label
    and takes :func:`conformal_quantile` inside each partition. Prediction
    later looks up the quantile for the test point's group, which is what
    makes coverage hold *conditionally on the group* rather than only on
    average across groups.

    Groups may be any hashable labels (ints, strings). Every group that
    appears must have enough calibration points for ``alpha``.
    """
    values = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(groups).ravel()
    if values.size != labels.size:
        raise ValueError("scores and groups must have the same length")
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must all be finite")

    quantiles = {}
    counts = {}
    for group in np.unique(labels):
        mask = labels == group
        group_scores = values[mask]
        key = group.item() if hasattr(group, "item") else group
        counts[key] = int(group_scores.size)
        try:
            quantiles[key] = conformal_quantile(group_scores, alpha)
        except ValueError as exc:
            if "calibration points" in str(exc):
                needed = int(np.ceil(1.0 / alpha)) - 1
                raise ValueError(
                    f"group {key!r} needs at least {needed} calibration points "
                    f"for alpha={alpha}, got {group_scores.size}"
                ) from exc
            raise
    return quantiles, counts


class MondrianSplitConformalRegressor:
    """Group-conditional split conformal intervals for regression.

    Same wrap-predictions API as :class:`SplitConformalRegressor`, but
    calibration is Mondrian: absolute residuals are grouped by a discrete
    attribute (region, segment, time bucket, ...) and each group gets its
    own finite-sample quantile. At prediction time the test point's group
    selects the width.

    The guarantee is group-conditional: for every group ``g``,

        P(Y in C(X) | G = g) >= 1 - alpha

    at finite ``n``, under exchangeability within the group. Ordinary split
    conformal only promises the average of those rates, so a rare / hard
    group can be systematically under-covered while the overall rate still
    looks fine.

    With ``normalize=True`` the score is residual / difficulty inside each
    group, the same as the pooled estimator.
    """

    def __init__(self, alpha=0.1, normalize=False):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self.normalize = bool(normalize)
        self.quantiles_ = None
        self.n_calibration_ = 0
        self.n_calibration_per_group_ = None

    def fit(self, y_true, y_pred, groups, difficulty=None):
        """Calibrate one interval width per group on held-out predictions."""
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        labels = np.asarray(groups).ravel()
        if observed.size != predicted.size or observed.size != labels.size:
            raise ValueError("y_true, y_pred, and groups must have the same length")
        if observed.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")

        residuals = np.abs(observed - predicted)
        if self.normalize:
            scale = SplitConformalRegressor._checked_difficulty(difficulty, observed.size)
            scores = residuals / scale
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            scores = residuals

        self.quantiles_, self.n_calibration_per_group_ = mondrian_regression_quantiles(
            scores, labels, self.alpha
        )
        self.n_calibration_ = int(observed.size)
        return self

    def predict_interval(self, y_pred, groups, difficulty=None):
        """Return ``(lower, upper)`` arrays using each point's group quantile."""
        if self.quantiles_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        predicted = np.asarray(y_pred, dtype=float).ravel()
        labels = np.asarray(groups).ravel()
        if predicted.size != labels.size:
            raise ValueError("y_pred and groups must have the same length")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predictions must all be finite")

        widths = np.empty(predicted.size, dtype=float)
        for i, group in enumerate(labels):
            key = group.item() if hasattr(group, "item") else group
            if key not in self.quantiles_:
                raise KeyError(
                    f"unknown group {key!r}; seen groups: "
                    f"{sorted(self.quantiles_, key=lambda g: str(g))}"
                )
            widths[i] = self.quantiles_[key]

        if self.normalize:
            scale = SplitConformalRegressor._checked_difficulty(difficulty, predicted.size)
            widths = widths * scale
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")

        return predicted - widths, predicted + widths
