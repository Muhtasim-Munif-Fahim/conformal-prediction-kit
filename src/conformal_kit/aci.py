"""Adaptive conformal inference (ACI) for streaming residuals.

Split conformal, CQR, jackknife+ and Mondrian all assume the calibration
and test points are exchangeable. Under distribution shift that assumption
fails, and a frozen quantile under-covers once the residuals grow.

Adaptive conformal inference (Gibbs and Candes 2021) restores *long-run*
coverage by treating the miscoverage level itself as a state variable.
After each outcome, the level is updated

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where ``err_t`` is 1 if the prediction set missed and 0 if it covered.
A miss *lowers* ``alpha_t``, so the next finite-sample quantile is more
conservative; a hit raises it. The telescoping sum of the recursion is

    mean(err_1..T) = alpha - (alpha_{T+1} - alpha_1) / (gamma * T)

so the average miscoverage tracks ``alpha`` at rate ``O(1 / (gamma T))``,
with no exchangeability assumption. When ``alpha_t`` leaves ``(0, 1)``
the prediction set becomes the whole space (``alpha_t <= 0``) or a
degenerate set (``alpha_t >= 1``), which pushes ``alpha_t`` back inside.

This module is that update plus the wrap-predictions API used elsewhere
in the kit. :class:`AdaptiveConformalRegressor` conformalizes absolute
residuals (the streaming analogue of
:class:`~conformal_kit.regression.SplitConformalRegressor`).
:class:`AdaptiveConformalClassifier` conformalizes LAC or APS scores
(the streaming analogue of
:class:`~conformal_kit.classification.SplitConformalClassifier`).
Both score a rolling window at the *current* ``alpha_t``; ``gamma``
only adapts the level, it does not replace the conformal quantile.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .classification import SplitConformalClassifier
from .regression import SplitConformalRegressor, conformal_quantile

__all__ = [
    "AdaptiveConformalClassifier",
    "AdaptiveConformalRegressor",
    "AdaptiveConformalUpdater",
    "aci_update",
    "adaptive_conformal_quantile",
]


def aci_update(alpha_t, err, alpha, gamma):
    """Return the next ACI miscoverage level.

    ``err`` is 1 when the last prediction set missed, 0 when it covered,
    or any value in ``[0, 1]`` for a randomized set. A miss lowers the
    level (wider next set); a hit raises it (tighter next set).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite step size")
    if not np.isfinite(alpha_t):
        raise ValueError("alpha_t must be finite")
    miss = _as_error(err)
    return float(alpha_t) + float(gamma) * (float(alpha) - miss)


def adaptive_conformal_quantile(scores, alpha):
    """Finite-sample ``1 - alpha`` quantile with ACI's out-of-range rules.

    Inside ``(0, 1)`` this is :func:`~conformal_kit.regression.conformal_quantile`,
    except that a level too tight for the window (rank ``> n``) returns
    ``+inf`` rather than raising: ACI answers "cover everything" instead
    of refusing. Outside ``(0, 1)`` the conventions are the ones the
    coverage recursion needs:

    * ``alpha <= 0`` → ``+inf`` (the whole space; the next error is 0)
    * ``alpha >= 1`` → ``-inf`` (a degenerate set; the next error is 1)
    """
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha <= 0.0:
        return np.inf
    if alpha >= 1.0:
        return -np.inf

    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        return np.inf
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must all be finite")

    n = values.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return np.inf
    if rank < 1:
        return -np.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def _as_error(err):
    miss = float(err)
    if not np.isfinite(miss) or miss < 0.0 or miss > 1.0:
        raise ValueError("err must be a finite value in [0, 1]")
    return miss


def _checked_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(alpha)


def _checked_gamma(gamma):
    if not np.isfinite(gamma) or float(gamma) <= 0.0:
        raise ValueError("gamma must be a positive finite step size")
    return float(gamma)


def _checked_window_size(window_size):
    if window_size is None:
        return None
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise TypeError("window_size must be an integer or None")
    if int(window_size) < 1:
        raise ValueError("window_size must be at least 1")
    return int(window_size)


def _as_1d_finite(values, name):
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must all be finite")
    return array


def _residual_half_width(quantile, n, scale=None):
    """Map an ACI residual quantile to a length-``n`` half-width."""
    if scale is None:
        scale = np.ones(n, dtype=float)
    if np.isposinf(quantile):
        return np.full(n, np.inf)
    if np.isneginf(quantile) or quantile <= 0.0:
        return np.zeros(n)
    return float(quantile) * scale


class AdaptiveConformalUpdater:
    """Online tracker for the ACI miscoverage level ``alpha_t``.

    This is the Gibbs and Candes recursion with no projection onto
    ``[0, 1]``. Out-of-range levels are left as they are; the prediction
    set convention in :func:`adaptive_conformal_quantile` is what pulls
    them back. The same updater applies to regression residuals and to
    classification nonconformity scores -- it only sees 0/1 errors.

    ``gamma`` is the step size. Smaller values track slowly and oscillate
    less; larger values react faster after a shift. There is no default
    that is uniformly right; ``0.05`` is a responsive starting point for
    streams of a few thousand points.
    """

    def __init__(self, alpha=0.1, gamma=0.05, alpha_init=None):
        self.alpha = _checked_alpha(alpha)
        self.gamma = _checked_gamma(gamma)
        if alpha_init is None:
            self.alpha_t = self.alpha
        else:
            if not np.isfinite(alpha_init):
                raise ValueError("alpha_init must be finite")
            self.alpha_t = float(alpha_init)
        self.n_updates_ = 0
        self.n_misses_ = 0.0

    def update(self, err):
        """Fold in one or more coverage errors, in order. Return ``self``."""
        misses = np.asarray(err, dtype=float).ravel()
        if misses.size == 0:
            raise ValueError("at least one error is required")
        for miss in misses:
            self.alpha_t = aci_update(self.alpha_t, miss, self.alpha, self.gamma)
            self.n_updates_ += 1
            self.n_misses_ += _as_error(miss)
        return self

    @property
    def empirical_miscoverage(self):
        """Average of the errors seen so far, or ``None`` before any update."""
        if self.n_updates_ == 0:
            return None
        return float(self.n_misses_ / self.n_updates_)

    @property
    def empirical_coverage(self):
        """One minus :attr:`empirical_miscoverage`, or ``None`` before any update."""
        rate = self.empirical_miscoverage
        if rate is None:
            return None
        return 1.0 - rate


class AdaptiveConformalRegressor:
    """Streaming residual intervals whose level tracks distribution shift.

    Same wrap-predictions shape as
    :class:`~conformal_kit.regression.SplitConformalRegressor`: ``fit``
    on held-out ``y_true`` / ``y_pred``, ``predict_interval`` on new
    predictions. The difference is that the miscoverage level is not
    frozen. After each outcome, :meth:`update` (or :meth:`predict_update`)
    appends the residual to a rolling window and steps ``alpha_t`` with
    :class:`AdaptiveConformalUpdater`. The next interval is the
    finite-sample quantile of that window at the new level.

    ``fit`` is ordinary split conformal at the target ``alpha``: it
    seeds the window and does not run the ACI recursion. The online
    loop starts at the first :meth:`update`. Use :meth:`predict_interval`
    for a snapshot that shares one ``alpha_t`` across a batch; use
    :meth:`predict_update` when each point should see the level left
    by the previous point.

    With ``normalize=True`` the score is the residual divided by a
    per-point difficulty estimate, as in split conformal. ``window_size``
    caps the residual window (oldest dropped first); ``None`` keeps
    every score since ``fit``. A sliding window forgets pre-shift
    residuals faster; the ``alpha_t`` update already compensates even
    with an expanding window.

    The guarantee is long-run, not exchangeable-finite-sample: the
    average coverage along the stream tracks ``1 - alpha``. Any one
    window after a sudden shift can under-cover while ``alpha_t``
    catches up.
    """

    def __init__(self, alpha=0.1, gamma=0.05, window_size=None, normalize=False):
        self.alpha = _checked_alpha(alpha)
        self.gamma = _checked_gamma(gamma)
        self.window_size = _checked_window_size(window_size)
        self.normalize = bool(normalize)
        self.updater = AdaptiveConformalUpdater(alpha=self.alpha, gamma=self.gamma)
        self.quantile_ = None
        self.n_calibration_ = 0
        self._scores = deque(maxlen=self.window_size)

    @property
    def alpha_t(self):
        """Current ACI miscoverage level."""
        return self.updater.alpha_t

    @property
    def width(self):
        """Interval width for the unnormalized case, else the base width."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before reading width")
        if np.isposinf(self.quantile_):
            return np.inf
        if np.isneginf(self.quantile_) or self.quantile_ <= 0.0:
            return 0.0
        return 2.0 * float(self.quantile_)

    def fit(self, y_true, y_pred, difficulty=None):
        """Seed the residual window at the target ``alpha``."""
        scores = self._scores_from(y_true, y_pred, difficulty)
        # Refuse a window that cannot support the *target* level. ACI may
        # later ask for a tighter level and answer with +inf; the starting
        # point should still be a well-defined split-conformal quantile.
        if self.window_size is not None and scores.size > self.window_size:
            seeded = scores[-self.window_size :]
        else:
            seeded = scores
        conformal_quantile(seeded, self.alpha)

        self.updater = AdaptiveConformalUpdater(alpha=self.alpha, gamma=self.gamma)
        self._scores = deque(seeded.tolist(), maxlen=self.window_size)
        self.n_calibration_ = len(self._scores)
        self._refresh_quantile()
        return self

    def predict_interval(self, y_pred, difficulty=None):
        """Return ``(lower, upper)`` at the current ``alpha_t``, without updating."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        predicted = _as_1d_finite(y_pred, "predictions")
        if self.normalize:
            scale = self._checked_difficulty(difficulty, predicted.size)
            half_width = _residual_half_width(self.quantile_, predicted.size, scale)
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            half_width = _residual_half_width(self.quantile_, predicted.size)
        return predicted - half_width, predicted + half_width

    def update(self, y_true, y_pred, difficulty=None):
        """Observe outcomes in order, adapt ``alpha_t`` and the residual window."""
        self.predict_update(y_pred, y_true, difficulty=difficulty)
        return self

    def predict_update(self, y_pred, y_true, difficulty=None):
        """Issue an interval for each point, then update before the next.

        Returns the ``(lower, upper)`` arrays that were issued *before*
        seeing each label. That is the online loop: predict, observe,
        adapt. ``y_pred`` comes first because that is the order the
        stream arrives in.
        """
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_update")
        predicted = _as_1d_finite(y_pred, "predictions")
        observed = _as_1d_finite(y_true, "y_true")
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")

        if self.normalize:
            scale = self._checked_difficulty(difficulty, predicted.size)
        else:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            scale = np.ones(predicted.size, dtype=float)

        lower = np.empty(predicted.size, dtype=float)
        upper = np.empty(predicted.size, dtype=float)
        for i in range(predicted.size):
            half = _residual_half_width(self.quantile_, 1, scale[i : i + 1])[0]
            lower[i] = predicted[i] - half
            upper[i] = predicted[i] + half
            covered = observed[i] >= lower[i] and observed[i] <= upper[i]
            self._scores.append(float(np.abs(observed[i] - predicted[i]) / scale[i]))
            self.updater.update(0.0 if covered else 1.0)
            self._refresh_quantile()
        self.n_calibration_ = len(self._scores)
        return lower, upper

    def _refresh_quantile(self):
        self.quantile_ = adaptive_conformal_quantile(np.asarray(self._scores), self.alpha_t)

    def _scores_from(self, y_true, y_pred, difficulty):
        observed = _as_1d_finite(y_true, "y_true")
        predicted = _as_1d_finite(y_pred, "y_pred")
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        residuals = np.abs(observed - predicted)
        if self.normalize:
            return residuals / self._checked_difficulty(difficulty, observed.size)
        if difficulty is not None:
            raise ValueError("difficulty requires normalize=True")
        return residuals

    @staticmethod
    def _checked_difficulty(difficulty, expected):
        # Same checks as SplitConformalRegressor: a zero would send a
        # residual to infinity and collapse the quantile.
        return SplitConformalRegressor._checked_difficulty(difficulty, expected)


class AdaptiveConformalClassifier:
    """Streaming prediction sets whose level tracks distribution shift.

    Same wrap-probabilities API as
    :class:`~conformal_kit.classification.SplitConformalClassifier` --
    ``fit`` on integer labels and a ``(n_samples, n_classes)`` probability
    matrix, ``predict_set`` returns a boolean mask -- but the threshold is
    the finite-sample quantile of a rolling score window at a time-varying
    ``alpha_t``. Scoring rules are the same ``lac`` / ``aps`` trade-off.

    ``fit`` seeds the window at the target ``alpha`` and does not run
    ACI. :meth:`update` / :meth:`predict_update` are the online loop.
    The long-run share of labels that land in their set tracks
    ``1 - alpha`` under shift; per-class coverage is still not guaranteed.
    For class-conditional thresholds on exchangeable data, see
    :class:`~conformal_kit.classification.MondrianConformalClassifier`.
    """

    def __init__(self, alpha=0.1, gamma=0.05, window_size=None, method="aps"):
        self.alpha = _checked_alpha(alpha)
        self.gamma = _checked_gamma(gamma)
        self.window_size = _checked_window_size(window_size)
        if method not in {"lac", "aps"}:
            raise ValueError("method must be 'lac' or 'aps'")
        self.method = method
        self.updater = AdaptiveConformalUpdater(alpha=self.alpha, gamma=self.gamma)
        self.quantile_ = None
        self.n_classes_ = 0
        self.n_calibration_ = 0
        self._scores = deque(maxlen=self.window_size)

    @property
    def alpha_t(self):
        """Current ACI miscoverage level."""
        return self.updater.alpha_t

    def fit(self, y_true, probabilities):
        """Seed the score window at the target ``alpha``."""
        labels, matrix = self._checked_calibration(y_true, probabilities)
        scores = self._true_scores(matrix, labels)
        if self.window_size is not None and scores.size > self.window_size:
            seeded = scores[-self.window_size :]
        else:
            seeded = scores
        conformal_quantile(seeded, self.alpha)

        self.updater = AdaptiveConformalUpdater(alpha=self.alpha, gamma=self.gamma)
        self._scores = deque(seeded.tolist(), maxlen=self.window_size)
        self.n_classes_ = int(matrix.shape[1])
        self.n_calibration_ = len(self._scores)
        self._refresh_quantile()
        return self

    def predict_set(self, probabilities):
        """Return a boolean membership mask at the current ``alpha_t``."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_set")
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(f"expected {self.n_classes_} classes, got {matrix.shape[1]}")
        return self._mask_from_quantile(matrix, self.quantile_)

    def predict_labels(self, probabilities):
        """Return the prediction set for each row as a list of class indices."""
        mask = self.predict_set(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, probabilities):
        """Return how many classes each row's prediction set holds."""
        return self.predict_set(probabilities).sum(axis=1)

    def update(self, y_true, probabilities):
        """Observe labels in order, adapt ``alpha_t`` and the score window."""
        self.predict_update(probabilities, y_true)
        return self

    def predict_update(self, probabilities, y_true):
        """Issue a set for each row, then update before the next.

        Returns the boolean mask that was issued *before* seeing each
        label. ``probabilities`` come first because that is the order
        the stream arrives in.
        """
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_update")
        labels, matrix = self._checked_calibration(y_true, probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(f"expected {self.n_classes_} classes, got {matrix.shape[1]}")

        mask = np.zeros_like(matrix, dtype=bool)
        for i in range(labels.size):
            row_mask = self._mask_from_quantile(matrix[i : i + 1], self.quantile_)
            mask[i] = row_mask[0]
            covered = bool(row_mask[0, labels[i]])
            self._scores.append(float(self._true_scores(matrix[i : i + 1], labels[i : i + 1])[0]))
            self.updater.update(0.0 if covered else 1.0)
            self._refresh_quantile()
        self.n_calibration_ = len(self._scores)
        return mask

    def _refresh_quantile(self):
        self.quantile_ = adaptive_conformal_quantile(np.asarray(self._scores), self.alpha_t)

    def _true_scores(self, matrix, labels):
        rows = np.arange(labels.size)
        if self.method == "lac":
            return 1.0 - matrix[rows, labels]
        return SplitConformalClassifier._aps_scores(matrix, labels)

    def _mask_from_quantile(self, matrix, quantile):
        if self.method == "lac":
            if np.isposinf(quantile):
                mask = np.ones_like(matrix, dtype=bool)
            elif np.isneginf(quantile):
                mask = np.zeros_like(matrix, dtype=bool)
            else:
                mask = matrix >= 1.0 - quantile
            SplitConformalClassifier._ensure_non_empty(mask, matrix)
            return mask

        order = np.argsort(-matrix, axis=1)
        ordered = np.take_along_axis(matrix, order, axis=1)
        cumulative = np.cumsum(ordered, axis=1)
        if np.isposinf(quantile):
            included = np.ones_like(cumulative, dtype=bool)
        elif np.isneginf(quantile):
            included = np.zeros_like(cumulative, dtype=bool)
        else:
            included = cumulative <= quantile
        included[:, 0] = True
        mask = np.zeros_like(matrix, dtype=bool)
        np.put_along_axis(mask, order, included, axis=1)
        return mask

    @staticmethod
    def _checked_calibration(y_true, probabilities):
        labels = np.asarray(y_true).ravel()
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if labels.size != matrix.shape[0]:
            raise ValueError("y_true and probabilities must have the same length")
        if labels.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y_true must contain integer class indices")
        if labels.min() < 0 or labels.max() >= matrix.shape[1]:
            raise ValueError("y_true contains a class index outside the probability matrix")
        return labels, matrix
