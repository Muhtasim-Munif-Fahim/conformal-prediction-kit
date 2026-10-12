"""Conformal outlier detection with marginal or false-discovery-rate control.

Any anomaly score turns into a valid p-value once it is compared with scores
from a held-out set of *inliers* (Vovk et al., 2005; Bates, Candès, Lei,
Romano & Sesia, "Testing for outliers with conformal p-values", Annals of
Statistics 2023). With ``n`` calibration scores, where a larger score means
more anomalous,

    p(x) = (1 + #{i : s_i >= s(x)}) / (n + 1).

If ``x`` is exchangeable with the calibration inliers, then
``P(p(x) <= t) <= t`` for every ``t``. There are two ways to use this:

* ``method="marginal"``: flag ``p <= alpha``. Each inlier is flagged with
  probability at most ``alpha``.
* ``method="bh"``: apply Benjamini–Hochberg at level ``alpha`` across a
  batch of test points. Conformal p-values that share one calibration set
  are positively dependent (PRDS), so BH still controls the false discovery
  rate ``E[#false flags / #flags] <= alpha`` (Bates et al., Theorem 2).

``randomized=True`` breaks ties with independent uniforms, which makes the
p-values exactly uniform under the null rather than only conservative.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ConformalOutlierDetector",
    "benjamini_hochberg",
    "conformal_p_values",
]


def _as_scores(values, name):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must all be finite")
    return arr


def _checked_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(alpha)


def conformal_p_values(calibration_scores, test_scores, *, randomized=False, random_state=None):
    """Conformal p-values of test scores against inlier calibration scores.

    Larger scores mean more anomalous. The deterministic p-value is
    ``(1 + #{s_i >= s}) / (n + 1)``. The randomized version is
    ``(#{s_i > s} + U * (1 + #{s_i == s})) / (n + 1)`` with ``U ~ Uniform(0, 1)``.
    """
    cal = np.sort(_as_scores(calibration_scores, "calibration_scores"))
    test = _as_scores(test_scores, "test_scores")
    n = cal.size
    greater_equal = n - np.searchsorted(cal, test, side="left")
    if not randomized:
        return (1.0 + greater_equal) / (n + 1.0)
    greater = n - np.searchsorted(cal, test, side="right")
    ties = greater_equal - greater
    rng = np.random.default_rng(random_state)
    u = rng.uniform(size=test.size)
    return (greater + u * (1.0 + ties)) / (n + 1.0)


def benjamini_hochberg(p_values, alpha):
    """Benjamini–Hochberg step-up procedure.

    Returns ``(rejected, adjusted)``. ``rejected`` is a boolean mask of the
    hypotheses rejected at FDR level ``alpha``, and ``adjusted`` holds the BH
    adjusted p-values (q-values), in the input order.
    """
    alpha = _checked_alpha(alpha)
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    if np.any((p < 0.0) | (p > 1.0)) or not np.all(np.isfinite(p)):
        raise ValueError("p_values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    below = np.flatnonzero(p[order] <= alpha * np.arange(1, m + 1) / m)
    rejected = np.zeros(m, dtype=bool)
    if below.size:
        rejected[order[: below[-1] + 1]] = True
    return rejected, adjusted


class ConformalOutlierDetector:
    """Calibrate an anomaly score on inliers and flag outliers with error control.

    ``fit`` takes anomaly scores of held-out *inliers* (points the scorer was
    not trained on), where larger means more anomalous. ``predict`` returns 1
    for flagged outliers and 0 for inliers.

    Parameters
    ----------
    alpha:
        Per-point false-alarm rate (``method="marginal"``) or target FDR
        (``method="bh"``).
    method:
        ``"bh"`` (default) or ``"marginal"``.
    randomized:
        Use randomized p-values with tie-breaking.
    random_state:
        Seed for the randomized p-values.
    """

    def __init__(self, alpha=0.1, method="bh", randomized=False, random_state=None):
        self.alpha = _checked_alpha(alpha)
        if method not in ("bh", "marginal"):
            raise ValueError("method must be 'bh' or 'marginal'")
        self.method = method
        self.randomized = bool(randomized)
        self.random_state = random_state

    def fit(self, calibration_scores):
        self.calibration_scores_ = np.sort(_as_scores(calibration_scores, "calibration_scores"))
        self.n_calibration_ = int(self.calibration_scores_.size)
        if 1.0 / (self.n_calibration_ + 1.0) > self.alpha:
            raise ValueError(
                f"alpha={self.alpha} needs at least {int(np.ceil(1.0 / self.alpha)) - 1} "
                f"calibration scores to flag anything, got {self.n_calibration_}"
            )
        return self

    def _check_fitted(self):
        if not hasattr(self, "calibration_scores_"):
            raise RuntimeError("call fit() before scoring test points")

    def p_values(self, test_scores):
        self._check_fitted()
        return conformal_p_values(
            self.calibration_scores_,
            test_scores,
            randomized=self.randomized,
            random_state=self.random_state,
        )

    def predict(self, test_scores):
        p = self.p_values(test_scores)
        if self.method == "marginal":
            flagged = p <= self.alpha
        else:
            flagged, _ = benjamini_hochberg(p, self.alpha)
        return flagged.astype(int)

    def decision_report(self, test_scores):
        """Dict with p-values, BH q-values, flags, and the method's settings."""
        p = self.p_values(test_scores)
        rejected, q = benjamini_hochberg(p, self.alpha)
        flags = rejected if self.method == "bh" else p <= self.alpha
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration_,
            "n_test": int(p.size),
            "n_flagged": int(np.sum(flags)),
            "p_values": p,
            "q_values": q,
            "flagged": flags.astype(int),
        }
