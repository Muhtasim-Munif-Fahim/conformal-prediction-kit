"""Split-conformal prediction sets for classification.

Where regression yields an interval, classification yields a *set* of labels.
The set is large where the model is unsure and small where it is confident,
and it contains the true label at least ``1 - alpha`` of the time without
assuming the model's probabilities are calibrated.
"""

from __future__ import annotations

import numpy as np

from .regression import conformal_quantile

__all__ = ["SplitConformalClassifier"]


class SplitConformalClassifier:
    """Turn predicted class probabilities into label sets with coverage.

    ``fit`` takes integer labels and a ``(n_samples, n_classes)`` probability
    matrix from a calibration set the model did not train on.
    ``predict_set`` returns, for each test row, a boolean mask over classes.

    Two scoring rules, and the choice between them is a real trade-off:

    ``lac``
        Least ambiguous classifier. The score is ``1 - p[true label]``, so the
        set holds every class whose probability clears a single threshold.
        Gives the smallest average set size of any method with valid marginal
        coverage, but the coverage it does give is uneven: easy points are
        over-covered and hard points under-covered.

    ``aps``
        Adaptive prediction sets. Classes are accumulated in descending
        probability until their total passes a threshold, so the score
        reflects how much probability mass had to be swept up to reach the
        truth. Sets are larger on average but adapt to each point, spreading
        coverage far more evenly across easy and hard inputs.

    Both guarantee at least ``1 - alpha`` marginal coverage. ``aps`` buys
    conditional behaviour with width; ``lac`` buys width with conditional
    behaviour.

    ``aps`` here is the deterministic variant, which over-covers rather than
    hitting the target exactly: sets never come back empty, and the score
    jumps by a whole class's probability at a time. On a 5-class problem at
    ``alpha=0.1`` that shows up as roughly 95% observed coverage against 90%
    asked for, where ``lac`` lands on 90%. Randomizing the final class would
    tighten it at the cost of non-reproducible sets.
    """

    def __init__(self, alpha=0.1, method="aps"):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if method not in {"lac", "aps"}:
            raise ValueError("method must be 'lac' or 'aps'")
        self.alpha = float(alpha)
        self.method = method
        self.quantile_ = None
        self.n_classes_ = 0
        self.n_calibration_ = 0

    def fit(self, y_true, probabilities):
        """Calibrate the score threshold on held-out probabilities."""
        labels = np.asarray(y_true).ravel()
        matrix = self._checked_probabilities(probabilities)
        if labels.size != matrix.shape[0]:
            raise ValueError("y_true and probabilities must have the same length")
        if labels.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y_true must contain integer class indices")
        if labels.min() < 0 or labels.max() >= matrix.shape[1]:
            raise ValueError("y_true contains a class index outside the probability matrix")

        rows = np.arange(labels.size)
        if self.method == "lac":
            scores = 1.0 - matrix[rows, labels]
        else:
            scores = self._aps_scores(matrix, labels)

        self.quantile_ = conformal_quantile(scores, self.alpha)
        self.n_classes_ = int(matrix.shape[1])
        self.n_calibration_ = int(labels.size)
        return self

    def predict_set(self, probabilities):
        """Return a boolean ``(n_samples, n_classes)`` membership mask."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict_set")
        matrix = self._checked_probabilities(probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(
                f"expected {self.n_classes_} classes, got {matrix.shape[1]}"
            )

        if self.method == "lac":
            mask = matrix >= 1.0 - self.quantile_
            # At a loose alpha the threshold can exclude every class. An
            # empty set is valid for coverage but useless to a caller, so
            # keep the top class; adding a class only raises coverage.
            self._ensure_non_empty(mask, matrix)
            return mask

        order = np.argsort(-matrix, axis=1)
        ordered = np.take_along_axis(matrix, order, axis=1)
        cumulative = np.cumsum(ordered, axis=1)
        # The calibration score is the cumulative mass *including* the true
        # class, so membership must use the same quantity or the set is
        # systematically larger than the threshold was calibrated for.
        included = cumulative <= self.quantile_
        # Keep the top class even when it alone exceeds the threshold: an
        # empty set carries no information and cannot cover anything.
        included[:, 0] = True

        mask = np.zeros_like(matrix, dtype=bool)
        np.put_along_axis(mask, order, included, axis=1)
        return mask

    def predict_labels(self, probabilities):
        """Return the prediction set for each row as a list of class indices."""
        mask = self.predict_set(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, probabilities):
        """Return how many classes each row's prediction set holds."""
        return self.predict_set(probabilities).sum(axis=1)

    @staticmethod
    def _ensure_non_empty(mask, matrix):
        """Add each empty row's most probable class, in place."""
        empty = ~mask.any(axis=1)
        if np.any(empty):
            mask[empty, np.argmax(matrix[empty], axis=1)] = True

    @staticmethod
    def _aps_scores(matrix, labels):
        """Cumulative probability swept up to and including the true class."""
        order = np.argsort(-matrix, axis=1)
        ordered = np.take_along_axis(matrix, order, axis=1)
        cumulative = np.cumsum(ordered, axis=1)
        # Where each row's true label landed once sorted by probability.
        position = np.argmax(order == labels[:, None], axis=1)
        return cumulative[np.arange(labels.size), position]

    @staticmethod
    def _checked_probabilities(probabilities):
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a 2-D (n_samples, n_classes) array")
        if matrix.shape[1] < 2:
            raise ValueError("probabilities must cover at least two classes")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("probabilities must all be finite")
        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise ValueError("probabilities must lie between 0 and 1")
        return matrix
