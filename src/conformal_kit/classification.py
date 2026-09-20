"""Split-conformal prediction sets for classification.

Where regression yields an interval, classification yields a *set* of labels.
The set is large where the model is unsure and small where it is confident,
and it contains the true label at least ``1 - alpha`` of the time without
assuming the model's probabilities are calibrated.
"""

from __future__ import annotations

import numpy as np

from .regression import conformal_quantile

__all__ = [
    "MondrianConformalClassifier",
    "SplitConformalClassifier",
    "mondrian_quantiles",
]


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

    Both guarantee at least ``1 - alpha`` *marginal* coverage. ``aps`` buys
    conditional behaviour with width; ``lac`` buys width with conditional
    behaviour. Neither promises coverage within a class -- a rare or hard
    label can be under-covered while easy labels make up the average. For
    a threshold per label, see :class:`MondrianConformalClassifier`.

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


def mondrian_quantiles(scores, labels, alpha, n_classes=None):
    """Return one finite-sample conformal quantile per class.

    Mondrian (class-conditional) conformal prediction partitions calibration
    scores by the true label and takes
    :func:`~conformal_kit.regression.conformal_quantile` inside each partition.
    The result is a length-``n_classes`` vector of thresholds: prediction
    later compares a candidate class against *that class's* quantile, which
    is what makes coverage hold given the true label rather than only on
    average.

    ``n_classes`` defaults to ``max(labels) + 1``. Pass the probability
    matrix width when a trailing class might be absent -- a missing class
    cannot get a threshold, and that is an error rather than a silent
    fallback. A class that appears but is smaller than the finite-sample
    requirement for ``alpha`` raises with the class index in the message.
    """
    values = np.asarray(scores, dtype=float).ravel()
    groups = np.asarray(labels).ravel()
    if values.size != groups.size:
        raise ValueError("scores and labels must have the same length")
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    if groups.min() < 0:
        raise ValueError("labels must be non-negative class indices")

    width = int(groups.max()) + 1 if n_classes is None else int(n_classes)
    if width < 1:
        raise ValueError("n_classes must be at least 1")
    if groups.max() >= width:
        raise ValueError("labels contain a class index outside n_classes")

    quantiles = np.empty(width, dtype=float)
    for cls in range(width):
        class_scores = values[groups == cls]
        if class_scores.size == 0:
            raise ValueError(
                f"class {cls} has no calibration examples; "
                "Mondrian conformal needs at least one example of every class"
            )
        try:
            quantiles[cls] = conformal_quantile(class_scores, alpha)
        except ValueError as exc:
            if "calibration points" in str(exc):
                needed = int(np.ceil(1.0 / alpha)) - 1
                raise ValueError(
                    f"class {cls} needs at least {needed} calibration points "
                    f"for alpha={alpha}, got {class_scores.size}"
                ) from exc
            raise
    return quantiles


def _aps_score_matrix(matrix):
    """APS score for every class: cumulative mass up to and including it."""
    order = np.argsort(-matrix, axis=1)
    ordered = np.take_along_axis(matrix, order, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    scores = np.empty_like(matrix)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


class MondrianConformalClassifier:
    """Class-conditional prediction sets: one conformal threshold per label.

    Same wrap-probabilities API as :class:`SplitConformalClassifier` --
    ``fit`` on integer labels and a ``(n_samples, n_classes)`` probability
    matrix, ``predict_set`` returns a boolean mask -- but calibration is
    *Mondrian*. Scores are grouped by the true class and each group gets
    its own finite-sample quantile. A test point includes class ``k`` when
    that class's score clears ``k``'s threshold, not a single shared one.

    The guarantee is class-conditional: for every label ``y``,

        P(Y in C(X) | Y = y) >= 1 - alpha

    at finite ``n``, under the same exchangeability assumption as split
    conformal. Split conformal only promises the average of those rates.
    That is enough when a missed class is no worse than a missed point;
    it is not enough when the rare diagnosis is the one that matters.

    Scoring rules are the same ``lac`` / ``aps`` trade-off. ``lac`` still
    gives smaller sets; ``aps`` still adapts to each point. The thresholds
    just no longer pool easy and hard labels together.

    The cost is data: every class needs enough calibration examples to
    support the finite-sample quantile. At ``alpha=0.1`` that is 9 points
    *per class*, not 9 points overall. A class that never appears cannot
    be given a threshold, and ``fit`` refuses rather than invent one.
    """

    def __init__(self, alpha=0.1, method="aps"):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if method not in {"lac", "aps"}:
            raise ValueError("method must be 'lac' or 'aps'")
        self.alpha = float(alpha)
        self.method = method
        self.quantiles_ = None
        self.n_classes_ = 0
        self.n_calibration_ = 0
        self.n_calibration_per_class_ = None

    def fit(self, y_true, probabilities):
        """Calibrate one score threshold per class on held-out probabilities."""
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

        rows = np.arange(labels.size)
        if self.method == "lac":
            scores = 1.0 - matrix[rows, labels]
        else:
            scores = SplitConformalClassifier._aps_scores(matrix, labels)

        self.quantiles_ = mondrian_quantiles(
            scores, labels, self.alpha, n_classes=matrix.shape[1]
        )
        self.n_classes_ = int(matrix.shape[1])
        self.n_calibration_ = int(labels.size)
        counts = np.bincount(labels, minlength=self.n_classes_)
        self.n_calibration_per_class_ = counts.astype(int)
        return self

    def predict_set(self, probabilities):
        """Return a boolean ``(n_samples, n_classes)`` membership mask."""
        if self.quantiles_ is None:
            raise RuntimeError("fit must be called before predict_set")
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(
                f"expected {self.n_classes_} classes, got {matrix.shape[1]}"
            )

        if self.method == "lac":
            mask = matrix >= 1.0 - self.quantiles_
        else:
            scores = _aps_score_matrix(matrix)
            mask = scores <= self.quantiles_
        # An empty set is valid for coverage but useless to a caller.
        # Adding the top class only raises coverage.
        SplitConformalClassifier._ensure_non_empty(mask, matrix)
        return mask

    def predict_labels(self, probabilities):
        """Return the prediction set for each row as a list of class indices."""
        mask = self.predict_set(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, probabilities):
        """Return how many classes each row's prediction set holds."""
        return self.predict_set(probabilities).sum(axis=1)
