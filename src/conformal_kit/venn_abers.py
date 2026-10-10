"""Inductive Venn-Abers predictors: calibrated probabilities with validity.

Conformal sets answer "which labels are plausible?". Venn-Abers predictors
(Vovk & Petej, 2014; Vovk, Petej & Fedorova, 2015) answer "what is the
probability of the positive class?" with a guarantee that plain Platt or
isotonic calibration lacks: under exchangeability, *one* of the two
probabilities they output is perfectly calibrated.

For a test score ``s`` the calibration set is augmented twice, once with
``(s, 0)`` and once with ``(s, 1)``, and an isotonic regression of label on
score is fitted to each augmented set. The two fitted values at ``s`` are a
multiprobability prediction ``(p0, p1)`` with ``p0 <= p1``. The gap
``p1 - p0`` shrinks as the calibration set grows and is wide where the
calibration data are thin, so it doubles as an epistemic-uncertainty signal.

A single probability for log loss is obtained by the minimax merge

    p = p1 / (1 - p0 + p1)

which always lies inside ``[p0, p1]``.

:func:`venn_abers_probabilities` implements the binary predictor on raw
scores. :class:`VennAbersCalibrator` wraps it for binary scores or
probability matrices; for more than two classes it calibrates each column
one-vs-rest and renormalizes the merged probabilities.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "VennAbersCalibrator",
    "venn_abers_probabilities",
]


def _pav_value_at(values, weights, target):
    """Pool-adjacent-violators fit; return the fitted value of block ``target``.

    ``values`` are block means sorted by score, ``weights`` their counts.
    """
    stack_mean = []
    stack_weight = []
    stack_start = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        mean = float(value)
        total = float(weight)
        start = idx
        while stack_mean and stack_mean[-1] >= mean:
            prev_mean = stack_mean.pop()
            prev_weight = stack_weight.pop()
            start = stack_start.pop()
            mean = (prev_mean * prev_weight + mean * total) / (prev_weight + total)
            total += prev_weight
        stack_mean.append(mean)
        stack_weight.append(total)
        stack_start.append(start)
    # locate the block that contains ``target``
    for k in range(len(stack_start) - 1, -1, -1):
        if stack_start[k] <= target:
            return stack_mean[k]
    return stack_mean[0]  # pragma: no cover - target >= 0 always matches


def _grouped(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order].astype(float)
    uniq, start, counts = np.unique(s, return_index=True, return_counts=True)
    sums = np.add.reduceat(y, start)
    return uniq, sums, counts.astype(float)


def _checked_binary(scores, labels):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    if s.size == 0:
        raise ValueError("calibration set must not be empty")
    if s.size != y.size:
        raise ValueError("scores and labels must have the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must all be finite")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be 0 or 1")
    return s, y.astype(int)


def venn_abers_probabilities(cal_scores, cal_labels, test_scores):
    """Return the Venn-Abers pair ``(p0, p1)`` for each test score.

    ``cal_scores`` are any real-valued scores where larger means "more likely
    positive" (probabilities, margins, logits), ``cal_labels`` the binary
    calibration labels, and ``test_scores`` the scores to calibrate.
    """
    s, y = _checked_binary(cal_scores, cal_labels)
    test = np.asarray(test_scores, dtype=float).ravel()
    if not np.all(np.isfinite(test)):
        raise ValueError("test scores must all be finite")
    uniq, sums, counts = _grouped(s, y)
    p0 = np.empty(test.size)
    p1 = np.empty(test.size)
    for i, score in enumerate(test):
        pos = int(np.searchsorted(uniq, score))
        exists = pos < uniq.size and uniq[pos] == score
        for label, out in ((0.0, p0), (1.0, p1)):
            if exists:
                g_sums = sums.copy()
                g_counts = counts.copy()
                g_sums[pos] += label
                g_counts[pos] += 1.0
            else:
                g_sums = np.insert(sums, pos, label)
                g_counts = np.insert(counts, pos, 1.0)
            out[i] = _pav_value_at(g_sums / g_counts, g_counts, pos)
    return p0, p1


def _merge(p0, p1):
    return p1 / (1.0 - p0 + p1)


class VennAbersCalibrator:
    """Inductive Venn-Abers calibration for binary or multiclass classifiers.

    ``fit(scores, y)`` stores a held-out calibration set. ``scores`` is either
    a 1-D array of positive-class scores (binary ``y`` in ``{0, 1}``) or a
    ``(n_samples, n_classes)`` probability / score matrix with integer
    labels. ``predict_interval`` returns the multiprobability ``(p0, p1)``
    for the positive class (binary) or for every class one-vs-rest;
    ``predict_proba`` returns merged probabilities, renormalized across
    classes in the multiclass case.
    """

    def __init__(self):
        self.n_classes_ = 0
        self.n_calibration_ = 0
        self._scores = None
        self._labels = None
        self._binary_1d = False

    def fit(self, scores, y):
        matrix = np.asarray(scores, dtype=float)
        labels = np.asarray(y).ravel()
        if matrix.ndim == 1:
            _checked_binary(matrix, labels)
            self._binary_1d = True
            self.n_classes_ = 2
        elif matrix.ndim == 2:
            if matrix.shape[1] < 2:
                raise ValueError("score matrix must have at least two columns")
            if matrix.shape[0] != labels.size:
                raise ValueError("scores and y must have the same length")
            if matrix.shape[0] == 0:
                raise ValueError("calibration set must not be empty")
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("y must contain integer class indices")
            if labels.min() < 0 or labels.max() >= matrix.shape[1]:
                raise ValueError("y contains a class index outside the score matrix")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("scores must all be finite")
            self._binary_1d = False
            self.n_classes_ = int(matrix.shape[1])
        else:
            raise ValueError("scores must be a 1-D array or a 2-D matrix")
        self._scores = matrix
        self._labels = labels.astype(int)
        self.n_calibration_ = int(labels.size)
        return self

    def _check_fitted(self):
        if self._scores is None:
            raise RuntimeError("fit must be called before predicting")

    def _test_matrix(self, scores):
        test = np.asarray(scores, dtype=float)
        if self._binary_1d:
            if test.ndim != 1:
                raise ValueError("calibrator was fitted on 1-D scores; pass 1-D scores")
            return test
        if test.ndim != 2 or test.shape[1] != self.n_classes_:
            raise ValueError(f"expected a (n, {self.n_classes_}) score matrix")
        return test

    def predict_interval(self, scores):
        """Return ``(p0, p1)``: arrays of shape ``(n,)`` (binary) or ``(n, K)``."""
        self._check_fitted()
        test = self._test_matrix(scores)
        if self._binary_1d:
            return venn_abers_probabilities(self._scores, self._labels, test)
        if self.n_classes_ == 2:
            return venn_abers_probabilities(
                self._scores[:, 1], self._labels, test[:, 1]
            )
        p0 = np.empty(test.shape)
        p1 = np.empty(test.shape)
        for k in range(self.n_classes_):
            lo, hi = venn_abers_probabilities(
                self._scores[:, k], (self._labels == k).astype(int), test[:, k]
            )
            p0[:, k] = lo
            p1[:, k] = hi
        return p0, p1

    def predict_proba(self, scores):
        """Merged calibrated probabilities, shape ``(n, n_classes)``."""
        p0, p1 = self.predict_interval(scores)
        merged = _merge(p0, p1)
        if merged.ndim == 1:
            return np.column_stack([1.0 - merged, merged])
        totals = merged.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1.0
        return merged / totals

    def interval_width(self, scores):
        """``p1 - p0`` per row (max over classes in the multiclass case)."""
        p0, p1 = self.predict_interval(scores)
        width = p1 - p0
        return width if width.ndim == 1 else width.max(axis=1)
