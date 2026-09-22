"""Cross-conformal and CV+ prediction sets for classification.

Inductive split conformal classification, including adaptive prediction sets,
already lives on :class:`~conformal_kit.classification.SplitConformalClassifier`
(``method="aps"`` or ``"lac"``). That method holds out a calibration set the
model never trains on. This module is the alternative when that split is too
expensive: the same nonconformity scores, but every point is scored by a model
that held it out, and the prediction set is read off cross-conformal p-values.

``n_splits=None`` refits leave-one-out. ``n_splits=K`` refits K-fold, which is
the classification counterpart of CV+. Both are Vovk's cross-conformal
predictor (2015) with deterministic tie-breaking. The finite-sample guarantee
is the jackknife+ one, about ``1 - 2 * alpha``, not split conformal's
``1 - alpha``. Regularized APS (RAPS) is not implemented: split APS was
already present, and this module is the cross-validation route instead.
"""

from __future__ import annotations

import copy

import numpy as np

from .classification import SplitConformalClassifier, _aps_score_matrix
from .jackknife import _as_features, _fold_ids
from .regression import conformal_quantile

__all__ = [
    "CrossConformalClassifier",
    "cross_conformal_p_values",
    "cross_conformal_sets",
]


def cross_conformal_p_values(calibration_scores, fold_ids, test_scores):
    """Return cross-conformal p-values for each test point and candidate class.

    ``calibration_scores`` has length ``n``: the nonconformity of each training
    point under the model that held its fold out. ``fold_ids`` assigns those
    points to folds ``0 .. K-1``. ``test_scores`` has shape
    ``(K, n_test, n_classes)``; entry ``[k, j, c]`` is the nonconformity of
    class ``c`` for test point ``j`` under the model trained without fold ``k``.

    The p-value for class ``c`` at test point ``j`` is

        (1 + #{i : R_i >= s_{k(i)}(x_j, c)}) / (n + 1)

    which is Vovk's cross-conformal p-value written with nonconformity scores
    (higher is worse). A tie counts as ``R_i >= s``, so the p-value only goes
    up. That is the deterministic, reproducible choice: randomizing the tie
    would shrink sets slightly and make them depend on a draw.
    """
    scores = np.asarray(calibration_scores, dtype=float).ravel()
    folds = np.asarray(fold_ids).ravel()
    test = np.asarray(test_scores, dtype=float)
    if scores.size != folds.size:
        raise ValueError("calibration_scores and fold_ids must have the same length")
    if scores.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must contain integer fold indices")
    if folds.min() < 0:
        raise ValueError("fold_ids must be non-negative")
    if test.ndim != 3:
        raise ValueError("test_scores must have shape (n_splits, n_test, n_classes)")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(test)):
        raise ValueError("scores must all be finite")

    n_splits = int(folds.max()) + 1
    if test.shape[0] != n_splits:
        raise ValueError(
            f"test_scores has {test.shape[0]} folds, expected {n_splits}"
        )
    for fold in range(n_splits):
        if not np.any(folds == fold):
            raise ValueError(f"fold {fold} has no calibration examples")

    counts = np.zeros(test.shape[1:], dtype=float)
    for fold in range(n_splits):
        held = scores[folds == fold]
        counts += np.sum(held[:, None, None] >= test[fold][None, :, :], axis=0)
    return (1.0 + counts) / (scores.size + 1.0)


def cross_conformal_sets(
    calibration_scores, fold_ids, test_scores, alpha, preference=None
):
    """Return a boolean prediction-set mask from cross-conformal p-values.

    Class ``c`` is included at a test point when its p-value is strictly
    greater than ``alpha``. That comparison is the same finite-sample rule as
    :func:`~conformal_kit.regression.conformal_quantile` when every fold model
    agrees: the class is kept exactly when its score is at most the split
    conformal quantile of the calibration scores.

    An empty row is not left empty. The class with the highest ``preference``
    (by default, the highest p-value) is added; ties keep the smallest index.
    Adding a class only raises coverage. The same floor as
    :func:`~conformal_kit.regression.conformal_quantile` applies: ``alpha``
    has to be supportable by ``n`` scores.
    """
    p_values = cross_conformal_p_values(calibration_scores, fold_ids, test_scores)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    # Same finite-sample floor as split conformal and jackknife+.
    conformal_quantile(np.asarray(calibration_scores, dtype=float).ravel(), alpha)

    mask = p_values > alpha
    if preference is None:
        chosen = p_values
    else:
        chosen = np.asarray(preference, dtype=float)
        if chosen.shape != mask.shape:
            raise ValueError(
                "preference must have shape (n_test, n_classes), "
                f"got {chosen.shape}"
            )
        if not np.all(np.isfinite(chosen)):
            raise ValueError("preference values must all be finite")
    empty = ~mask.any(axis=1)
    if np.any(empty):
        rows = np.flatnonzero(empty)
        mask[rows, np.argmax(chosen[rows], axis=1)] = True
    return mask


def _class_nonconformity(probabilities, method):
    """Nonconformity of every class. Higher means less conforming."""
    if method == "lac":
        return 1.0 - probabilities
    return _aps_score_matrix(probabilities)


def _as_classifier(model):
    """Accept a fit/predict_proba estimator or a ``(X, y) -> predict_proba`` callable."""
    if hasattr(model, "fit") and hasattr(model, "predict_proba"):

        def train(X_train, y_train, _proto=model):
            fitted = copy.deepcopy(_proto)
            fitted.fit(X_train, y_train)
            return fitted.predict_proba

        return train
    if callable(model):

        def train(X_train, y_train, _factory=model):
            predictor = _factory(X_train, y_train)
            if not callable(predictor):
                raise TypeError(
                    "callable model must return a predictor: "
                    "model(X, y) -> predict_proba(X)"
                )
            return predictor

        return train
    raise ValueError(
        "model must be an estimator with fit and predict_proba, "
        "or a callable (X, y) -> predict_proba"
    )


def _checked_proba(values, n_rows, n_classes):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (n_rows, n_classes):
        raise ValueError(
            f"predict_proba must return shape ({n_rows}, {n_classes}), got {matrix.shape}"
        )
    return SplitConformalClassifier._checked_probabilities(matrix)


class CrossConformalClassifier:
    """Cross-conformal / CV+ label sets from out-of-fold probabilities.

    :class:`~conformal_kit.classification.SplitConformalClassifier` wraps
    probabilities a fitted model already produced on a held-out calibration
    set. This class refits. ``n_splits=None`` is leave-one-out cross-conformal
    (``n`` fits); ``n_splits=K`` is K-fold CV+ (``K`` fits). A candidate label
    is kept when its cross-conformal p-value exceeds ``alpha``.

    ``model`` is passed to :meth:`fit` so construction stays aligned with
    :class:`~conformal_kit.jackknife.JackknifePlusRegressor`: hyperparameters
    here, data there. It can be:

    * a duck-typed estimator with ``fit(X, y)`` and ``predict_proba(X)`` --
      cloned with ``deepcopy`` on each fold, no sklearn required;
    * a callable ``model(X_train, y_train) -> predict_proba``, where
      ``predict_proba(X)`` returns a ``(n_samples, n_classes)`` probability
      matrix.

    Class indices on ``y`` are ``0 .. n_classes - 1`` with
    ``n_classes = max(y) + 1``. ``predict_proba`` has to use that same width
    on every fold, including a fold that happened to miss a class. Pass the
    full class count into the estimator when a fold can miss a label.

    Scoring rules are the same ``lac`` / ``aps`` trade-off as split conformal.
    ``lac`` still tends to smaller sets; ``aps`` still adapts to how much
    probability mass had to be swept up. The sets are not required to be a
    prefix of one model's ranking, because each fold model casts its own vote.

    The coverage guarantee is at least ``1 - 2 * alpha``, up to a vanishing
    ``O(1 / sqrt(n))`` term that is zero for leave-one-out (Barber, Candès,
    Ramdas, Tibshirani 2021, Remark 1, applied to Vovk's cross-conformal
    p-values). That is weaker than split conformal's ``1 - alpha``, and it is
    the reason to prefer split conformal whenever a calibration set is
    affordable. In practice the sets usually sit close to the split target.
    """

    def __init__(self, alpha=0.1, method="aps", n_splits=None, random_state=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if method not in {"lac", "aps"}:
            raise ValueError("method must be 'lac' or 'aps'")
        if n_splits is not None:
            if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
                raise ValueError("n_splits must be an integer or None")
            if int(n_splits) < 2:
                raise ValueError("n_splits must be at least 2")
            n_splits = int(n_splits)
        self.alpha = float(alpha)
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state
        self.calibration_scores_ = None
        self.fold_ids_ = None
        self.n_calibration_ = 0
        self.n_splits_ = 0
        self.n_classes_ = 0
        self.n_features_ = 0
        self._predictors = None

    def fit(self, X, y, model):
        """Refit on each fold and store out-of-fold nonconformity scores."""
        features = _as_features(X)
        labels = np.asarray(y).ravel()
        if labels.size != features.shape[0]:
            raise ValueError("X and y must have the same length")
        if labels.size == 0:
            raise ValueError("training set must not be empty")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must contain integer class indices")
        if labels.min() < 0:
            raise ValueError("y must be non-negative class indices")
        if np.unique(labels).size < 2:
            raise ValueError("at least two classes are required")

        n = labels.size
        n_classes = int(labels.max()) + 1
        n_splits = n if self.n_splits is None else self.n_splits
        if n_splits > n:
            raise ValueError(f"n_splits={n_splits} cannot exceed n={n}")
        # Same finite-sample floor as split conformal: the p-value level has
        # to be supportable by n scores before any model is fit.
        conformal_quantile(np.zeros(n), self.alpha)

        trainer = _as_classifier(model)
        fold_ids = _fold_ids(n, n_splits, self.random_state)
        calibration_scores = np.empty(n, dtype=float)
        predictors = []
        for fold in range(n_splits):
            held_out = fold_ids == fold
            predictor = trainer(features[~held_out], labels[~held_out])
            n_held = int(held_out.sum())
            proba = _checked_proba(predictor(features[held_out]), n_held, n_classes)
            scores = _class_nonconformity(proba, self.method)
            calibration_scores[held_out] = scores[np.arange(n_held), labels[held_out]]
            predictors.append(predictor)

        self.calibration_scores_ = calibration_scores
        self.fold_ids_ = fold_ids
        self._predictors = predictors
        self.n_calibration_ = n
        self.n_splits_ = n_splits
        self.n_classes_ = n_classes
        self.n_features_ = int(features.shape[1])
        return self

    def predict_p_values(self, X):
        """Return cross-conformal p-values with shape ``(n_test, n_classes)``."""
        test_scores, _mean = self._fold_scores(X)
        return cross_conformal_p_values(
            self.calibration_scores_, self.fold_ids_, test_scores
        )

    def predict_set(self, X):
        """Return a boolean ``(n_samples, n_classes)`` membership mask."""
        test_scores, mean_proba = self._fold_scores(X)
        return cross_conformal_sets(
            self.calibration_scores_,
            self.fold_ids_,
            test_scores,
            self.alpha,
            preference=mean_proba,
        )

    def predict_labels(self, X):
        """Return the prediction set for each row as a list of class indices."""
        mask = self.predict_set(X)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, X):
        """Return how many classes each row's prediction set holds."""
        return self.predict_set(X).sum(axis=1)

    def _fold_scores(self, X):
        if self._predictors is None:
            raise RuntimeError("fit must be called before prediction")
        features = _as_features(X)
        if features.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {features.shape[1]}"
            )
        n_test = features.shape[0]
        test_scores = np.empty((self.n_splits_, n_test, self.n_classes_), dtype=float)
        mean_proba = np.zeros((n_test, self.n_classes_), dtype=float)
        for fold, predictor in enumerate(self._predictors):
            proba = _checked_proba(predictor(features), n_test, self.n_classes_)
            test_scores[fold] = _class_nonconformity(proba, self.method)
            mean_proba += proba
        mean_proba /= self.n_splits_
        return test_scores, mean_proba
