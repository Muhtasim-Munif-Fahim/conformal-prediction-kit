"""Conformal selection: pick test units whose outcome exceeds a threshold with FDR control.

Jin & Candès, "Selection by Prediction with Conformal p-values" (JMLR 2023).
Given a fitted predictor ``mu`` and a labelled calibration set, we want to
select test units ``j`` whose *unobserved* outcome satisfies ``Y_j > c_j``
(e.g. "candidates whose binding affinity exceeds c") while controlling the
false discovery rate

    FDR = E[ #{j selected : Y_j <= c_j} / max(1, #selected) ] <= q.

Each test unit gets the null hypothesis ``H_j : Y_j <= c_j``. With a score
``V(x, y)`` that is monotone non-decreasing in ``y``, the calibration scores
``V_i = V(X_i, Y_i)`` are compared with the test statistic
``V_hat_j = V(X_j, c_j)``:

    p_j = (#{i : V_i < V_hat_j} + U_j * (1 + #{i : V_i = V_hat_j})) / (n + 1)

(``U_j = 1`` gives the deterministic, slightly conservative version). Under
``H_j`` these p-values are super-uniform, and Benjamini–Hochberg at level
``q`` controls the FDR (Jin & Candès, Theorem 2.3).

Two scores are provided:

* ``"clipped"`` (default, recommended):
  ``V(x, y) = M * 1{y > c} + c - mu(x)``. Calibration units that are
  already "positive" (``Y_i > c_i``) get a huge score and never count
  against a test unit, which makes the procedure much more powerful.
* ``"residual"``: ``V(x, y) = y - mu(x)``.
"""

from __future__ import annotations

import numpy as np

from .outliers import benjamini_hochberg

__all__ = [
    "ConformalSelector",
    "conformal_selection",
    "selection_p_values",
    "selection_report",
]

_SCORES = ("clipped", "residual")


def _as_vector(values, name):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must all be finite")
    return arr


def _broadcast_thresholds(thresholds, n, name):
    c = np.asarray(thresholds, dtype=float)
    if c.ndim == 0:
        c = np.full(n, float(c))
    c = c.ravel()
    if c.size != n:
        raise ValueError(f"{name} must be a scalar or have length {n}, got {c.size}")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"{name} must all be finite")
    return c


def _check_score(score):
    if score not in _SCORES:
        raise ValueError(f"score must be one of {_SCORES}, got {score!r}")
    return score


def _scores(score, y_cal, mu_cal, c_cal, mu_test, c_test):
    """Return calibration scores V_i and test statistics V_hat_j."""
    if score == "residual":
        return y_cal - mu_cal, c_test - mu_test
    # Any M exceeding the spread of ``c - mu`` keeps positives above every
    # test statistic; use a data-driven bound instead of +inf so ties and
    # arithmetic stay finite.
    spread = np.concatenate([c_cal - mu_cal, c_test - mu_test])
    big = 2.0 * (np.max(np.abs(spread)) + 1.0)
    v_cal = big * (y_cal > c_cal) + c_cal - mu_cal
    v_test = c_test - mu_test
    return v_cal, v_test


def selection_p_values(
    y_calibration,
    pred_calibration,
    pred_test,
    threshold,
    *,
    test_threshold=None,
    score="clipped",
    randomized=False,
    random_state=None,
):
    """Conformal p-values for ``H_j : Y_j <= c_j`` on each test unit.

    ``threshold`` is ``c`` for the calibration units (scalar or per-unit);
    ``test_threshold`` defaults to the same scalar. Small p-values are
    evidence that the test outcome exceeds its threshold.
    """
    score = _check_score(score)
    y_cal = _as_vector(y_calibration, "y_calibration")
    mu_cal = _as_vector(pred_calibration, "pred_calibration")
    if y_cal.size != mu_cal.size:
        raise ValueError("y_calibration and pred_calibration must have the same length")
    mu_test = _as_vector(pred_test, "pred_test")
    c_cal = _broadcast_thresholds(threshold, y_cal.size, "threshold")
    if test_threshold is None:
        if np.ndim(threshold) != 0:
            raise ValueError("pass test_threshold when threshold is per-unit")
        test_threshold = threshold
    c_test = _broadcast_thresholds(test_threshold, mu_test.size, "test_threshold")

    v_cal, v_test = _scores(score, y_cal, mu_cal, c_cal, mu_test, c_test)
    n = v_cal.size
    v_sorted = np.sort(v_cal)
    less = np.searchsorted(v_sorted, v_test, side="left")
    less_equal = np.searchsorted(v_sorted, v_test, side="right")
    ties = less_equal - less
    if randomized:
        u = np.random.default_rng(random_state).uniform(size=v_test.size)
    else:
        u = np.ones(v_test.size)
    return np.minimum((less + u * (1.0 + ties)) / (n + 1.0), 1.0)


def conformal_selection(
    y_calibration,
    pred_calibration,
    pred_test,
    threshold,
    q=0.1,
    *,
    test_threshold=None,
    score="clipped",
    randomized=False,
    random_state=None,
):
    """Select test units with ``Y_j > c_j`` at FDR level ``q``.

    Returns ``(selected, p_values, q_values)`` where ``selected`` is a
    boolean mask in test order.
    """
    p = selection_p_values(
        y_calibration,
        pred_calibration,
        pred_test,
        threshold,
        test_threshold=test_threshold,
        score=score,
        randomized=randomized,
        random_state=random_state,
    )
    selected, adjusted = benjamini_hochberg(p, q)
    return selected, p, adjusted


def selection_report(selected, y_true, threshold):
    """Realised false discovery proportion and power of a selection.

    ``threshold`` is a scalar or per-unit ``c_j``. ``power`` is the share of
    true positives (``Y_j > c_j``) that were selected (``nan`` when there are
    none).
    """
    sel = np.asarray(selected, dtype=bool).ravel()
    y = _as_vector(y_true, "y_true")
    if sel.size != y.size:
        raise ValueError("selected and y_true must have the same length")
    c = _broadcast_thresholds(threshold, y.size, "threshold")
    positive = y > c
    n_sel = int(sel.sum())
    false_sel = int(np.sum(sel & ~positive))
    n_pos = int(positive.sum())
    return {
        "n_test": int(y.size),
        "n_selected": n_sel,
        "n_false_discoveries": false_sel,
        "fdp": false_sel / max(n_sel, 1),
        "n_positives": n_pos,
        "power": (int(np.sum(sel & positive)) / n_pos) if n_pos else float("nan"),
    }


class ConformalSelector:
    """Estimator-style wrapper around :func:`conformal_selection`.

    >>> sel = ConformalSelector(threshold=0.0, q=0.1).fit(y_cal, pred_cal)
    >>> mask = sel.select(pred_test)
    """

    def __init__(self, threshold, q=0.1, score="clipped", randomized=False, random_state=None):
        if not 0.0 < q < 1.0:
            raise ValueError("q must be strictly between 0 and 1")
        self.threshold = threshold
        self.q = float(q)
        self.score = _check_score(score)
        self.randomized = randomized
        self.random_state = random_state

    def fit(self, y_calibration, pred_calibration):
        y = _as_vector(y_calibration, "y_calibration")
        mu = _as_vector(pred_calibration, "pred_calibration")
        if y.size != mu.size:
            raise ValueError("y_calibration and pred_calibration must have the same length")
        _broadcast_thresholds(self.threshold, y.size, "threshold")
        self.y_calibration_ = y
        self.pred_calibration_ = mu
        return self

    def _check_fitted(self):
        if not hasattr(self, "y_calibration_"):
            raise RuntimeError("ConformalSelector is not fitted; call fit() first")

    def p_values(self, pred_test, test_threshold=None):
        self._check_fitted()
        return selection_p_values(
            self.y_calibration_,
            self.pred_calibration_,
            pred_test,
            self.threshold,
            test_threshold=test_threshold,
            score=self.score,
            randomized=self.randomized,
            random_state=self.random_state,
        )

    def select(self, pred_test, test_threshold=None):
        selected, _ = benjamini_hochberg(self.p_values(pred_test, test_threshold), self.q)
        return selected

    def decision_report(self, pred_test, test_threshold=None):
        p = self.p_values(pred_test, test_threshold)
        selected, adjusted = benjamini_hochberg(p, self.q)
        return {
            "score": self.score,
            "q": self.q,
            "n_calibration": int(self.y_calibration_.size),
            "n_test": int(p.size),
            "n_selected": int(selected.sum()),
            "p_values": p,
            "q_values": adjusted,
            "selected": selected,
        }
