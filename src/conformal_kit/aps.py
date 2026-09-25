"""Adaptive prediction sets and regularized APS for classification.

:class:`~conformal_kit.classification.SplitConformalClassifier` already
scores labels with ``lac`` or ``aps``. This module is that adaptive score,
and its regularized form, as their own classifiers. Both are split
conformal: calibrate a threshold on held-out probabilities, then return a
label set. Neither retrains the underlying model. The probabilities are a
softmax, or any row of class scores already scaled into ``[0, 1]``.

``APS`` (Romano, Sesia, Candès 2020) scores a label by the probability mass
accumulated, in descending order, up to and including that label. The set
grows until the cumulative mass clears a single quantile, so easy points
stay small and ambiguous points collect more labels.

``RAPS`` (Angelopoulos, Bates, Malik, Jordan 2021) adds a penalty once a
label's rank passes ``k_reg``. Classes that only just scraped into the APS
set get pushed back out, which shortens sets on diffuse probabilities. The
quantile is taken on the penalized scores, so the marginal coverage
guarantee is the same ``1 - alpha``. ``penalty=0`` is exactly APS.
"""

from __future__ import annotations

import numpy as np

from .classification import SplitConformalClassifier
from .regression import conformal_quantile

__all__ = [
    "APSClassifier",
    "RAPSClassifier",
    "aps_scores",
    "raps_scores",
]


def _checked_penalty(penalty):
    try:
        value = float(penalty)
    except (TypeError, ValueError):
        raise ValueError(
            "penalty must be a finite number greater than or equal to 0"
        ) from None
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("penalty must be a finite number greater than or equal to 0")
    return value


def _checked_k_reg(k_reg):
    # ``bool`` is an ``int`` subclass; a flag is not a rank cutoff.
    if isinstance(k_reg, bool) or not isinstance(k_reg, (int, np.integer)):
        raise TypeError("k_reg must be a non-negative integer")
    value = int(k_reg)
    if value < 0:
        raise ValueError("k_reg must be a non-negative integer")
    return value


def _checked_labels(y_true, matrix):
    labels = np.asarray(y_true).ravel()
    if labels.size != matrix.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    if labels.size == 0:
        raise ValueError("calibration set must not be empty")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true must contain integer class indices")
    if labels.min() < 0 or labels.max() >= matrix.shape[1]:
        raise ValueError("y_true contains a class index outside the probability matrix")
    return labels


def _ranking(matrix):
    """Sort each row by descending probability and return cumulative mass.

    ``order[i, r]`` is the class at 0-based rank ``r``. ``cumulative[i, r]``
    is the probability mass of the top ``r + 1`` classes, which is the APS
    nonconformity of the class sitting at that rank.
    """
    order = np.argsort(-matrix, axis=1)
    ordered = np.take_along_axis(matrix, order, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    return order, cumulative


def _penalized_cumulative(cumulative, penalty, k_reg):
    """Add ``penalty * max(rank - k_reg, 0)`` with ranks starting at 1.

    A zero penalty is returned unchanged so RAPS at ``penalty=0`` is
    bit-for-bit the APS cumulative mass, not a float that happens to be close.
    """
    if penalty == 0.0:
        return cumulative
    ranks = np.arange(1, cumulative.shape[1] + 1)
    return cumulative + penalty * np.maximum(ranks - k_reg, 0)


def _label_position(order, labels):
    """0-based rank of each row's true label inside ``order``."""
    return np.argmax(order == labels[:, None], axis=1)


def _nonconformity(matrix, labels, penalty, k_reg):
    order, cumulative = _ranking(matrix)
    position = _label_position(order, labels)
    scores = cumulative[np.arange(labels.size), position]
    if penalty == 0.0:
        return scores
    ranks = position + 1
    return scores + penalty * np.maximum(ranks - k_reg, 0)


def _prediction_mask(matrix, quantile, penalty, k_reg):
    order, cumulative = _ranking(matrix)
    scores = _penalized_cumulative(cumulative, penalty, k_reg)
    # Membership uses the same score calibration computed, including the
    # class itself. Comparing against the mass *before* the class would
    # admit one extra label on every row.
    included = scores <= quantile
    # The top class can sit above the quantile on its own. Dropping it
    # would leave an empty set, which cannot cover the truth and tells
    # the caller nothing. Keeping it only raises coverage.
    included[:, 0] = True
    mask = np.zeros(matrix.shape, dtype=bool)
    np.put_along_axis(mask, order, included, axis=1)
    return mask


def aps_scores(probabilities, y_true):
    """APS nonconformity: cumulative probability up to and including ``y_true``.

    Classes are taken in descending probability. The score is the softmax
    (or probability) mass of every class at least as likely as the true
    one, including the true one. A confident correct prediction scores near
    its top probability; a label buried down the ranking scores near 1.
    """
    matrix = SplitConformalClassifier._checked_probabilities(probabilities)
    labels = _checked_labels(y_true, matrix)
    return _nonconformity(matrix, labels, penalty=0.0, k_reg=0)


def raps_scores(probabilities, y_true, penalty=0.01, k_reg=1):
    """RAPS nonconformity: APS mass plus a penalty past rank ``k_reg``.

    With 1-based rank ``L`` of the true label,

        score = cumulative mass through L + penalty * max(L - k_reg, 0)

    ``k_reg=1`` leaves the most probable class unpenalized and charges
    ``penalty`` for the second, ``2 * penalty`` for the third, and so on.
    ``penalty=0`` returns :func:`aps_scores`.
    """
    penalty = _checked_penalty(penalty)
    k_reg = _checked_k_reg(k_reg)
    matrix = SplitConformalClassifier._checked_probabilities(probabilities)
    labels = _checked_labels(y_true, matrix)
    return _nonconformity(matrix, labels, penalty, k_reg)


class _CumulativeSetClassifier:
    """Split-conformal sets from a cumulative-probability score."""

    def __init__(self, alpha, penalty, k_reg):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self._penalty = _checked_penalty(penalty)
        self._k_reg = _checked_k_reg(k_reg)
        self.quantile_ = None
        self.n_classes_ = 0
        self.n_calibration_ = 0

    def fit(self, y_true, probabilities):
        """Calibrate the score threshold on held-out probabilities."""
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        labels = _checked_labels(y_true, matrix)
        scores = _nonconformity(matrix, labels, self._penalty, self._k_reg)
        self.quantile_ = conformal_quantile(scores, self.alpha)
        self.n_classes_ = int(matrix.shape[1])
        self.n_calibration_ = int(labels.size)
        return self

    def predict_set(self, probabilities):
        """Return a boolean ``(n_samples, n_classes)`` membership mask."""
        if self.quantile_ is None:
            raise RuntimeError("fit must be called before predict")
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(
                f"expected {self.n_classes_} classes, got {matrix.shape[1]}"
            )
        return _prediction_mask(matrix, self.quantile_, self._penalty, self._k_reg)

    def predict(self, probabilities):
        """Return prediction sets as a boolean membership mask.

        Alias of :meth:`predict_set`. The mask is the set: column ``k`` is
        true when class ``k`` is included.
        """
        return self.predict_set(probabilities)

    def predict_labels(self, probabilities):
        """Return the prediction set for each row as a list of class indices."""
        mask = self.predict_set(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, probabilities):
        """Return how many classes each row's prediction set holds."""
        return self.predict_set(probabilities).sum(axis=1)


class APSClassifier(_CumulativeSetClassifier):
    """Adaptive prediction sets from cumulative softmax mass.

    ``fit`` takes integer labels and a ``(n_samples, n_classes)`` probability
    matrix from a calibration set the classifier did not train on.
    ``predict_set`` and ``predict`` return a boolean mask over classes.

    The score is the cumulative probability of classes ranked ahead of the
    true label, including it. That is the deterministic APS score of Romano,
    Sesia, and Candès (2020), the same one
    :class:`~conformal_kit.classification.SplitConformalClassifier` computes
    for ``method="aps"``. Sets are never empty: a row whose leading class
    already exceeds the quantile still keeps that class.

    Marginal coverage is at least ``1 - alpha`` when calibration and test
    rows are exchangeable. The deterministic score over-covers rather than
    hitting the target exactly, because the score jumps by a whole class's
    probability at a time. Randomizing the last included class would tighten
    it and would make the sets depend on a draw; these sets stay fixed.

    For a penalty that discourages long tails, use :class:`RAPSClassifier`.
    For one threshold per label, use
    :class:`~conformal_kit.classification.MondrianConformalClassifier`.
    """

    def __init__(self, alpha=0.1):
        super().__init__(alpha=alpha, penalty=0.0, k_reg=0)


class RAPSClassifier(_CumulativeSetClassifier):
    """Regularized adaptive prediction sets.

    Same wrap-probabilities API as :class:`APSClassifier` -- ``fit`` on
    integer labels and a probability matrix, ``predict`` / ``predict_set``
    return a boolean mask -- with the RAPS score of Angelopoulos, Bates,
    Malik, and Jordan (2021). A label at 1-based rank ``L`` scores

        cumulative probability through L + penalty * max(L - k_reg, 0)

    The first ``k_reg`` classes pay nothing. Every rank after that pays
    ``penalty`` more than the rank before it, so a class that contributes
    little probability and sits deep in the ordering has to clear a higher
    bar. Calibration uses that same score, and the finite-sample quantile
    is what keeps coverage at ``1 - alpha`` whatever ``penalty`` and
    ``k_reg`` are. The knobs change the size of the sets, not the
    guarantee. A bad choice yields valid sets that are wider or, with a
    harsh penalty, sets that often collapse to the top class.

    ``penalty=0`` or ``k_reg`` greater than or equal to the number of
    classes removes the regularizer and reproduces :class:`APSClassifier`.
    The default ``penalty=0.01`` and ``k_reg=1`` is a light trim: only
    classes that barely cleared the APS threshold lose their place. Raise
    ``penalty`` when the probabilities are diffuse and the sets are still
    longer than you can act on.

    As with APS, the top class is always kept and the score is not
    randomized, so the sets are reproducible and slightly conservative.
    """

    def __init__(self, alpha=0.1, penalty=0.01, k_reg=1):
        super().__init__(alpha=alpha, penalty=penalty, k_reg=k_reg)
        self.penalty = self._penalty
        self.k_reg = self._k_reg
