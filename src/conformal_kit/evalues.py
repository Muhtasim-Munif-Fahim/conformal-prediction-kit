"""Conformal e-values and the e-BH procedure.

E-values are non-negative statistics with ``E[e] <= 1`` under the null.
Unlike p-values they can be averaged (any average of e-values is an
e-value), which makes them the natural tool for *derandomising* conformal
inference across several random calibration splits, and the e-BH
procedure (Wang & Ramdas, JRSS-B 2022) controls the FDR under arbitrary
dependence between the e-values.

``conformal_e_values`` implements the BH-thresholded conformal e-values of
Bashari, Epstein, Romano & Sesia ("Derandomized novelty detection with FDR
control via conformal e-values", NeurIPS 2023). Larger scores mean more
anomalous. With ``n`` calibration and ``m`` test scores the threshold is::

    T = min{ t in cal ∪ test :
             m/(n+1) * (1 + #{cal >= t}) / max(1, #{test >= t}) <= alpha }

and each test point receives ``e_j = (n+1) * 1{s_j >= T} / (1 + #{cal >= T})``.
e-BH at level ``alpha`` applied to these e-values rejects exactly the same
set as BH applied to the conformal p-values; averaging them over ``K``
splits and running e-BH gives a derandomised, still FDR-controlling result.
"""

from __future__ import annotations

import numpy as np

from .outliers import _as_scores, _checked_alpha

__all__ = [
    "conformal_e_values",
    "derandomized_e_values",
    "e_bh",
    "e_to_p",
    "p_to_e",
]


def _cal_ge(cal_sorted, t):
    return cal_sorted.size - np.searchsorted(cal_sorted, t, side="left")


def conformal_e_values(calibration_scores, test_scores, alpha=0.1, *, threshold=None):
    """Conformal e-values for outlier detection.

    Parameters
    ----------
    calibration_scores, test_scores:
        Non-conformity scores (larger = more anomalous). Calibration scores
        must come from inliers not used to train the score function.
    alpha:
        Target FDR level used to pick the data-driven threshold ``T``.
    threshold:
        Optional fixed threshold ``T`` chosen independently of the data; the
        resulting e-values are then valid marginally as well.

    Returns
    -------
    (e_values, T)
    """
    cal = np.sort(_as_scores(calibration_scores, "calibration_scores"))
    test = _as_scores(test_scores, "test_scores")
    n, m = cal.size, test.size
    if threshold is None:
        alpha = _checked_alpha(alpha)
        candidates = np.unique(np.concatenate([cal, test]))
        test_sorted = np.sort(test)
        cal_counts = _cal_ge(cal, candidates)
        test_counts = m - np.searchsorted(test_sorted, candidates, side="left")
        fdp_hat = (m / (n + 1.0)) * (1.0 + cal_counts) / np.maximum(1, test_counts)
        ok = np.flatnonzero(fdp_hat <= alpha)
        T = float(candidates[ok[0]]) if ok.size else float("inf")
    else:
        T = float(threshold)
    denom = 1.0 + _cal_ge(cal, T)
    e = np.where(test >= T, (n + 1.0) / denom, 0.0)
    return e, T


def e_bh(e_values, alpha):
    """e-BH procedure (Wang & Ramdas 2022).

    Rejects the ``k*`` hypotheses with the largest e-values, where
    ``k* = max{k : e_(k) >= m / (alpha * k)}`` and ``e_(1) >= e_(2) >= ...``.
    Controls FDR at ``alpha`` under arbitrary dependence. Returns a boolean
    rejection mask in input order.
    """
    alpha = _checked_alpha(alpha)
    e = np.asarray(e_values, dtype=float).ravel()
    if e.size == 0:
        return np.zeros(0, dtype=bool)
    if np.any(e < 0) or np.any(np.isnan(e)):
        raise ValueError("e_values must be non-negative")
    m = e.size
    order = np.argsort(-e, kind="stable")
    k = np.arange(1, m + 1)
    ok = np.flatnonzero(e[order] * alpha * k >= m)
    rejected = np.zeros(m, dtype=bool)
    if ok.size:
        rejected[order[: ok[-1] + 1]] = True
    return rejected


def derandomized_e_values(e_value_runs, weights=None):
    """Weighted average of e-values from several calibration splits.

    ``e_value_runs`` has shape ``(K, m)``. Any convex combination of
    e-values is an e-value, so the result is valid for e-BH.
    """
    E = np.asarray(e_value_runs, dtype=float)
    if E.ndim != 2 or E.shape[0] == 0:
        raise ValueError("e_value_runs must be a non-empty 2-D array (K, m)")
    if np.any(E < 0):
        raise ValueError("e-values must be non-negative")
    if weights is None:
        w = np.full(E.shape[0], 1.0 / E.shape[0])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != E.shape[0] or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per run, not all zero")
        w = w / w.sum()
    return w @ E


def p_to_e(p_values, kappa=0.5):
    """Power calibrator ``e = kappa * p**(kappa - 1)`` for ``kappa`` in (0, 1)."""
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must be strictly between 0 and 1")
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        return kappa * np.power(p, kappa - 1.0)


def e_to_p(e_values):
    """Markov calibrator ``p = min(1, 1/e)``."""
    e = np.asarray(e_values, dtype=float)
    if np.any(e < 0):
        raise ValueError("e_values must be non-negative")
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, 1.0 / e)
