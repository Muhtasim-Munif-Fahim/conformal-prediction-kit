"""Conformal risk control (Angelopoulos, Bates, Fisch, Lei & Schuster, 2022).

Split conformal prediction controls the *miscoverage* probability. Conformal
risk control generalises it to the expected value of any bounded loss that
shrinks as a tuning parameter ``lambda`` grows: false negative rate of a
multilabel set, missed pixels in a segmentation mask, token-level recall, and
so on. With ``n`` exchangeable calibration points whose losses
``L_i(lambda)`` are non-increasing in ``lambda`` and bounded by ``B``, the
threshold

    lambda_hat = inf { lambda : (n / (n + 1)) * R_n(lambda) + B / (n + 1) <= alpha }

(``R_n`` the empirical risk) satisfies ``E[L_test(lambda_hat)] <= alpha``.
Taking ``L = 1{y not in C_lambda}`` recovers ordinary split conformal.

:func:`conformal_risk_threshold` implements the rule for any loss matrix.
:class:`MultilabelRiskController` applies it to multilabel classification,
returning label sets ``{k : p_k >= 1 - lambda}`` whose expected false
negative rate is at most ``alpha``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MultilabelRiskController",
    "conformal_risk_threshold",
    "false_negative_rate",
]


def _checked_alpha(alpha):
    try:
        value = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be strictly between 0 and 1") from None
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return value


def conformal_risk_threshold(losses, lambdas, alpha, loss_bound=1.0):
    """Smallest ``lambda`` whose corrected calibration risk is at most ``alpha``.

    Parameters
    ----------
    losses : array of shape (n, m)
        ``losses[i, j]`` is calibration point ``i``'s loss at ``lambdas[j]``.
        Every row must be non-increasing in ``j`` and lie in
        ``[0, loss_bound]``.
    lambdas : array of shape (m,)
        Strictly increasing candidate thresholds.
    alpha : float
        Target expected loss, in ``(0, 1)`` (relative to ``loss_bound=1``).
    loss_bound : float
        Upper bound ``B`` on any single loss.

    Raises ``ValueError`` when no candidate meets the bound, which happens
    when even the largest ``lambda`` leaves too much loss or the calibration
    set is too small (``B / (n + 1) > alpha``).
    """
    alpha = _checked_alpha(alpha)
    bound = float(loss_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("loss_bound must be a positive finite number")
    grid = np.asarray(lambdas, dtype=float).ravel()
    matrix = np.asarray(losses, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != grid.size or grid.size == 0:
        raise ValueError("losses must have shape (n_calibration, len(lambdas))")
    if matrix.shape[0] == 0:
        raise ValueError("at least one calibration point is required")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("lambdas must be finite and strictly increasing")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("losses must all be finite")
    if np.any(matrix < 0.0) or np.any(matrix > bound + 1e-12):
        raise ValueError("losses must lie in [0, loss_bound]")
    if np.any(np.diff(matrix, axis=1) > 1e-12):
        raise ValueError("each loss row must be non-increasing in lambda")
    n = matrix.shape[0]
    corrected = (n * matrix.mean(axis=0) + bound) / (n + 1)
    ok = np.flatnonzero(corrected <= alpha)
    if ok.size == 0:
        raise ValueError(
            f"no lambda achieves risk <= {alpha}; the smallest corrected risk is "
            f"{corrected.min():.4f} (n={n})"
        )
    return float(grid[ok[0]])


def _checked_multilabel(y_true, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty 2D array (n, n_labels)")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if y_true is None:
        return None, p
    y = np.asarray(y_true)
    if y.shape != p.shape:
        raise ValueError("y_true must be a multi-hot array with the same shape as probabilities")
    if not np.all(np.isin(y, (0, 1, True, False))):
        raise ValueError("y_true must contain only 0/1 indicators")
    return y.astype(bool), p


def false_negative_rate(y_true, prediction_sets):
    """Mean per-example false negative rate of multilabel prediction sets.

    For each row, FNR is ``1 - |Y ∩ C| / |Y|``; rows with no positive labels
    contribute 0. Returns the average over rows.
    """
    y = np.asarray(y_true).astype(bool)
    sets = np.asarray(prediction_sets).astype(bool)
    if y.shape != sets.shape or y.ndim != 2:
        raise ValueError("y_true and prediction_sets must be 2D arrays of the same shape")
    positives = y.sum(axis=1)
    hits = (y & sets).sum(axis=1)
    fnr = np.where(positives > 0, 1.0 - hits / np.maximum(positives, 1), 0.0)
    return float(fnr.mean())


class MultilabelRiskController:
    """Multilabel prediction sets with expected false negative rate <= ``alpha``.

    Fit on held-out multi-hot labels and per-label probabilities (e.g.
    independent sigmoids). The set for a test row is
    ``{k : p_k >= 1 - lambda_hat}``, where ``lambda_hat`` is chosen by
    conformal risk control on the FNR loss. Because FNR only changes at the
    points ``1 - p_ik`` of the true labels, the search over ``lambda`` is
    exact rather than on a grid.

    Parameters
    ----------
    alpha : float
        Target expected false negative rate.

    Attributes
    ----------
    lambda_ : float
        Calibrated threshold in ``[0, 1]``.
    threshold_ : float
        Probability cut-off ``1 - lambda_``.
    calibration_risk_ : float
        Empirical calibration FNR at ``lambda_``.
    """

    def __init__(self, alpha=0.1):
        self.alpha = _checked_alpha(alpha)
        self.lambda_ = None
        self.threshold_ = None
        self.calibration_risk_ = None
        self.n_labels_ = None

    def fit(self, y_true, probabilities):
        """Calibrate ``lambda_`` on held-out labels and probabilities."""
        y, p = _checked_multilabel(y_true, probabilities)
        n = y.shape[0]
        positives = y.sum(axis=1)
        # Each true label (i, k) adds 1 / |Y_i| to row i's loss while
        # lambda < 1 - p_ik, i.e. while that label is still excluded.
        rows, cols = np.nonzero(y)
        cut = 1.0 - p[rows, cols]
        weight = 1.0 / positives[rows]
        order = np.argsort(cut, kind="stable")
        cut, weight = cut[order], weight[order]
        candidates = np.unique(np.concatenate([[0.0], cut]))
        tail = np.concatenate([np.cumsum(weight[::-1])[::-1], [0.0]])
        # Loss mass still excluded at lambda = c: labels with cut > c.
        first_above = np.searchsorted(cut, candidates, side="right")
        risk = tail[first_above] / n
        corrected = (n * risk + 1.0) / (n + 1)
        ok = np.flatnonzero(corrected <= self.alpha)
        if ok.size == 0:
            needed = int(np.ceil(1.0 / self.alpha)) - 1
            raise ValueError(
                f"alpha={self.alpha} needs at least {needed} calibration rows, got {n}"
            )
        self.lambda_ = float(candidates[ok[0]])
        self.threshold_ = 1.0 - self.lambda_
        self.calibration_risk_ = float(risk[ok[0]])
        self.n_labels_ = p.shape[1]
        return self

    def _check_fitted(self):
        if self.lambda_ is None:
            raise RuntimeError("call fit before predicting")

    def predict_set(self, probabilities):
        """Boolean ``(n, n_labels)`` mask of the labels in each set."""
        self._check_fitted()
        _, p = _checked_multilabel(None, probabilities)
        if p.shape[1] != self.n_labels_:
            raise ValueError(f"expected {self.n_labels_} label columns, got {p.shape[1]}")
        return p >= self.threshold_ - 1e-12

    def predict_labels(self, probabilities):
        """List of label-index arrays, one per row."""
        return [np.flatnonzero(row) for row in self.predict_set(probabilities)]

    def set_sizes(self, probabilities):
        """Number of labels in each prediction set."""
        return self.predict_set(probabilities).sum(axis=1)
