"""Split conformal predictive distributions (conformal predictive systems).

An interval answers one question at one level. A *predictive distribution*
answers all of them at once: it is a full CDF over the label, from which any
interval, quantile, exceedance probability or proper score can be read off.

Split conformal predictive systems (Vovk, Shen, Manokhin and Xie, 2017;
Vovk et al., "Nonparametric predictive distributions based on conformal
prediction", Machine Learning 2019) build that CDF from signed calibration
residuals ``r_i = (y_i - yhat_i) / sigma_i``. For a test prediction ``yhat``
with difficulty ``sigma`` the randomized conformal predictive distribution is

    Q(y, tau) = (#{i : r_i < s} + tau * (#{i : r_i = s} + 1)) / (n + 1),
    s = (y - yhat) / sigma,  tau ~ Uniform(0, 1).

Its key property is *calibration in probability*: under exchangeability the
value ``Q(y_test, tau)`` is exactly uniform on ``[0, 1]``, so every quantile of
the distribution has the advertised frequency on new data -- for any model.
The thin step function ``Q(y, 0)`` / ``Q(y, 1)`` pair brackets the CDF.

Use :class:`SplitConformalPredictiveSystem` to wrap an already-fitted point
model, exactly like :class:`~conformal_kit.regression.SplitConformalRegressor`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SplitConformalPredictiveSystem",
    "crps_empirical",
    "pit_uniformity_report",
]


def crps_empirical(samples, y):
    """Continuous ranked probability score of an equally-weighted sample.

    Uses the energy form ``E|X - y| - 0.5 E|X - X'|`` evaluated exactly in
    ``O(n log n)`` via the sorted-sample identity
    ``E|X - X'| = 2 / n^2 * sum_i (2i - n - 1) x_(i)`` (one-based ``i``).
    Lower is better; the score is in the units of ``y``.
    """
    values = np.sort(np.asarray(samples, dtype=float).ravel())
    if values.size == 0:
        raise ValueError("at least one sample is required")
    n = values.size
    term1 = np.mean(np.abs(values - float(y)))
    ranks = np.arange(1, n + 1, dtype=float)
    spread = 2.0 / n**2 * np.sum((2.0 * ranks - n - 1.0) * values)
    return float(term1 - 0.5 * spread)


def pit_uniformity_report(pit, n_bins=10):
    """Histogram and Kolmogorov-Smirnov distance of PIT values from uniform.

    Returns a dict with ``ks_statistic`` (sup distance between the empirical
    PIT CDF and the identity), ``bin_frequencies`` (share per equal-width bin,
    ideally ``1 / n_bins`` each) and ``max_bin_deviation``.
    """
    values = np.sort(np.asarray(pit, dtype=float).ravel())
    if values.size == 0:
        raise ValueError("at least one PIT value is required")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("PIT values must lie in [0, 1]")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be at least 1")
    n = values.size
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    ks = float(max(np.max(upper - values), np.max(values - lower)))
    counts, _ = np.histogram(values, bins=int(n_bins), range=(0.0, 1.0))
    freqs = counts / n
    return {
        "n": int(n),
        "ks_statistic": ks,
        "bin_frequencies": freqs.tolist(),
        "max_bin_deviation": float(np.max(np.abs(freqs - 1.0 / int(n_bins)))),
    }


class SplitConformalPredictiveSystem:
    """Full conformal predictive CDFs from held-out signed residuals.

    ``fit(y_true, y_pred, difficulty=None)`` stores the sorted (optionally
    difficulty-normalized) signed residuals. Then, for new predictions:

    * :meth:`cdf` -- ``Q(y, tau)`` for given candidate labels.
    * :meth:`pit` -- randomized PIT values ``Q(y_true, tau)`` (uniform under
      exchangeability; use :func:`pit_uniformity_report` to check).
    * :meth:`quantile` -- conservative quantiles of the predictive
      distribution (``+inf`` when the calibration set is too small to resolve
      the requested upper level).
    * :meth:`predict_interval` -- central ``1 - alpha`` intervals read off the
      distribution, with the usual split-conformal coverage guarantee.
    * :meth:`crps` -- CRPS of the calibrated residual distribution, a proper
      scoring rule for comparing models' full predictive distributions.
    """

    def __init__(self, normalize=False):
        self.normalize = bool(normalize)
        self.residuals_ = None

    @property
    def n_calibration_(self):
        return 0 if self.residuals_ is None else int(self.residuals_.size)

    def fit(self, y_true, y_pred, difficulty=None):
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        if observed.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("calibration values must all be finite")
        scale = self._scale(difficulty, observed.size)
        self.residuals_ = np.sort((observed - predicted) / scale)
        return self

    def cdf(self, y, y_pred, difficulty=None, tau=0.5):
        """Evaluate ``Q(y, tau)`` elementwise for broadcastable ``y``/``y_pred``.

        ``tau=0`` gives the lower and ``tau=1`` the upper thin CDF; any value
        in ``[0, 1]`` (or an array of them) gives the randomized CDF.
        """
        self._check_fitted()
        predicted = np.asarray(y_pred, dtype=float)
        candidate = np.asarray(y, dtype=float)
        scale = self._broadcast_scale(difficulty)
        tau_arr = np.asarray(tau, dtype=float)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must lie in [0, 1]")
        s = (candidate - predicted) / scale
        below = np.searchsorted(self.residuals_, s, side="left")
        at_or_below = np.searchsorted(self.residuals_, s, side="right")
        ties = at_or_below - below
        n = self.residuals_.size
        return (below + tau_arr * (ties + 1)) / (n + 1)

    def pit(self, y_true, y_pred, difficulty=None, seed=None):
        """Randomized probability integral transform of observed labels."""
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        tau = np.random.default_rng(seed).uniform(size=observed.size)
        return self.cdf(observed, predicted, difficulty=difficulty, tau=tau)

    def quantile(self, levels, y_pred, difficulty=None):
        """Quantiles of the predictive distribution, shape ``(n_pred, n_levels)``.

        The level-``p`` quantile is ``yhat + sigma * r_(k)`` with
        ``k = ceil(p * (n + 1))``, the upper end of the quantile set of the
        conformal predictive distribution, so ``P(Y <= q_p) >= p`` on new
        exchangeable data. Levels with ``k > n`` return ``+inf``, the honest
        answer when the calibration set cannot resolve that tail.
        """
        self._check_fitted()
        probs = np.atleast_1d(np.asarray(levels, dtype=float))
        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        predicted = np.asarray(y_pred, dtype=float).ravel()
        scale = self._scale(difficulty, predicted.size)
        n = self.residuals_.size
        ranks = np.ceil(probs * (n + 1) - 1e-12).astype(int)
        offsets = np.empty(probs.size)
        for j, k in enumerate(ranks):
            if k > n:
                offsets[j] = np.inf
            else:
                offsets[j] = self.residuals_[k - 1]
        return predicted[:, None] + scale[:, None] * offsets[None, :]

    def predict_interval(self, y_pred, alpha=0.1, difficulty=None):
        """Central interval ``[Q^-1(alpha/2), Q^-1(1 - alpha/2)]``.

        Uses ``floor`` / ``ceil`` ranks so each tail misses at most
        ``alpha / 2`` and coverage is at least ``1 - alpha``.
        """
        self._check_fitted()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        predicted = np.asarray(y_pred, dtype=float).ravel()
        scale = self._scale(difficulty, predicted.size)
        n = self.residuals_.size
        k_lo = int(np.floor((n + 1) * alpha / 2.0))
        k_hi = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
        lo = -np.inf if k_lo < 1 else self.residuals_[k_lo - 1]
        hi = np.inf if k_hi > n else self.residuals_[k_hi - 1]
        return predicted + scale * lo, predicted + scale * hi

    def crps(self, y_true, y_pred, difficulty=None):
        """Per-point CRPS of ``yhat + sigma * residuals`` against ``y_true``."""
        self._check_fitted()
        observed = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if observed.size != predicted.size:
            raise ValueError("y_true and y_pred must have the same length")
        scale = self._scale(difficulty, predicted.size)
        return np.array(
            [
                crps_empirical(p + s * self.residuals_, y)
                for y, p, s in zip(observed, predicted, scale)
            ]
        )

    def _check_fitted(self):
        if self.residuals_ is None:
            raise RuntimeError("fit must be called first")

    def _broadcast_scale(self, difficulty):
        if not self.normalize:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            return 1.0
        if difficulty is None:
            raise ValueError("normalize=True requires difficulty estimates")
        scale = np.asarray(difficulty, dtype=float)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("difficulty estimates must be finite and strictly positive")
        return scale

    def _scale(self, difficulty, expected):
        if not self.normalize:
            if difficulty is not None:
                raise ValueError("difficulty requires normalize=True")
            return np.ones(expected)
        if difficulty is None:
            raise ValueError("normalize=True requires difficulty estimates")
        scale = np.asarray(difficulty, dtype=float).ravel()
        if scale.size != expected:
            raise ValueError("difficulty must have the same length as the predictions")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("difficulty estimates must be finite and strictly positive")
        return scale
