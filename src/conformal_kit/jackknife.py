"""Jackknife+ and CV+ prediction intervals for regression.

Split conformal holds out a calibration set the model never trains on. That
is cheap and gives a ``1 - alpha`` guarantee, but it spends data on
calibration that a small sample may not be able to spare.

Jackknife+ (Barber, Candes, Ramdas, Tibshirani 2021) keeps every point in
training by refitting leave-one-out. The residual at each point comes from a
model that never saw it, and the interval at a test point is built from those
leave-one-out predictions rather than from a single fitted model. CV+ is the
same construction with K-fold refits instead of ``n``.

The finite-sample guarantee is ``1 - 2 * alpha`` rather than ``1 - alpha``.
In practice the intervals usually sit close to the split-conformal target;
the extra ``alpha`` is the price of not holding data out. sklearn is not
required: pass a duck-typed ``fit`` / ``predict`` estimator or a trainer
callable that returns a predictor.
"""

from __future__ import annotations

import copy

import numpy as np

from .regression import conformal_quantile

__all__ = ["JackknifePlusRegressor", "jackknife_plus_interval"]


def jackknife_plus_interval(loo_predictions, residuals, alpha):
    """Return jackknife+ ``(lower, upper)`` from leave-one-out predictions.

    ``residuals`` has length ``n`` and must be nonnegative magnitudes.
    ``loo_predictions`` is either the
    leave-one-out prediction of a single test point (length ``n``) or a
    ``(n, n_test)`` matrix whose ``i``-th row is the prediction from the
    model trained without point ``i``.

    The upper endpoint is the ``ceil((n + 1) * (1 - alpha))``-th smallest
    value of ``prediction_i + residual_i``; the lower endpoint is the same
    quantile of the negated ``prediction_i - residual_i`` values. That is
    the finite-sample construction whose coverage is at least
    ``1 - 2 * alpha``. When the required rank exceeds ``n`` the same error
    as :func:`conformal_quantile` is raised -- no finite interval can
    guarantee the level.
    """
    scores = np.asarray(residuals, dtype=float).ravel()
    predictions = np.asarray(loo_predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    elif predictions.ndim != 2:
        raise ValueError("loo_predictions must be a 1-D or 2-D array")
    if predictions.shape[0] != scores.size:
        raise ValueError("loo_predictions rows must match the number of residuals")
    if scores.size == 0:
        raise ValueError("at least one residual is required")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(predictions)):
        raise ValueError("loo predictions and residuals must all be finite")
    if np.any(scores < 0):
        raise ValueError("residuals must be nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    # The rank check is the same finite-sample constraint as split conformal.
    conformal_quantile(scores, alpha)

    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    left = predictions - scores[:, None]
    right = predictions + scores[:, None]
    # Rank is one-based. The lower endpoint is -q_plus of the negated left
    # values, which is the (n - rank + 1)-th smallest of the left values.
    lower = np.partition(left, n - rank, axis=0)[n - rank]
    upper = np.partition(right, rank - 1, axis=0)[rank - 1]
    return lower, upper


def _as_features(X, name="X"):
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must all be finite")
    return array


def _as_trainer(model):
    """Accept a fit/predict estimator or a ``(X, y) -> predictor`` callable."""
    if hasattr(model, "fit") and hasattr(model, "predict"):

        def train(X_train, y_train, _proto=model):
            fitted = copy.deepcopy(_proto)
            fitted.fit(X_train, y_train)
            return fitted.predict

        return train
    if callable(model):

        def train(X_train, y_train, _factory=model):
            predictor = _factory(X_train, y_train)
            if not callable(predictor):
                raise TypeError(
                    "callable model must return a predictor: model(X, y) -> predict(X)"
                )
            return predictor

        return train
    raise ValueError(
        "model must be an estimator with fit and predict, "
        "or a callable (X, y) -> predictor"
    )


def _fold_ids(n, n_splits, random_state):
    sizes = np.full(n_splits, n // n_splits, dtype=int)
    sizes[: n % n_splits] += 1
    ids = np.repeat(np.arange(n_splits), sizes)
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        rng.shuffle(ids)
    return ids


def _as_predictions(values, expected, what):
    predicted = np.asarray(values, dtype=float).ravel()
    if predicted.size != expected:
        raise ValueError(f"{what} returned {predicted.size} predictions, expected {expected}")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("model predictions must all be finite")
    return predicted


class JackknifePlusRegressor:
    """Jackknife+ / CV+ intervals that train on every point.

    :class:`~conformal_kit.regression.SplitConformalRegressor` wraps
    already-fitted predictions. This class refits. ``n_splits=None`` is
    leave-one-out jackknife+ (``n`` fits); ``n_splits=K`` is CV+ (``K``
    fits). The interval at a test point is the finite-sample quantile of
    the leave-one-out (or out-of-fold) predictions plus or minus their
    residuals, not a single point prediction plus a shared half-width.

    ``model`` is passed to :meth:`fit` so construction stays aligned with
    :class:`~conformal_kit.regression.SplitConformalRegressor`:
    hyperparameters here, data there. It can be:

    * a duck-typed estimator with ``fit(X, y)`` and ``predict(X)`` --
      cloned with ``deepcopy`` on each fold, no sklearn required;
    * a callable ``model(X_train, y_train) -> predict``, where ``predict(X)``
      returns a 1-D array of predictions.

    The coverage guarantee is at least ``1 - 2 * alpha``. That is weaker
    than split conformal's ``1 - alpha``, and it is the reason to prefer
    split conformal whenever a calibration set is affordable.
    """

    def __init__(self, alpha=0.1, n_splits=None, random_state=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if n_splits is not None:
            if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
                raise ValueError("n_splits must be an integer or None")
            if int(n_splits) < 2:
                raise ValueError("n_splits must be at least 2")
            n_splits = int(n_splits)
        self.alpha = float(alpha)
        self.n_splits = n_splits
        self.random_state = random_state
        self.residuals_ = None
        self.fold_ids_ = None
        self.n_calibration_ = 0
        self.n_splits_ = 0
        self.n_features_ = 0
        self._predictors = None

    def fit(self, X, y, model):
        """Refit on each fold and store out-of-fold residuals."""
        features = _as_features(X)
        observed = np.asarray(y, dtype=float).ravel()
        if observed.size != features.shape[0]:
            raise ValueError("X and y must have the same length")
        if observed.size == 0:
            raise ValueError("training set must not be empty")
        if not np.all(np.isfinite(observed)):
            raise ValueError("y values must all be finite")

        n = observed.size
        n_splits = n if self.n_splits is None else self.n_splits
        if n_splits > n:
            raise ValueError(f"n_splits={n_splits} cannot exceed n={n}")
        # Same finite-sample floor as split conformal: the plus quantile
        # rank must land inside the n residuals.
        conformal_quantile(np.zeros(n), self.alpha)

        trainer = _as_trainer(model)
        fold_ids = _fold_ids(n, n_splits, self.random_state)
        residuals = np.empty(n, dtype=float)
        predictors = []
        for fold in range(n_splits):
            held_out = fold_ids == fold
            predictor = trainer(features[~held_out], observed[~held_out])
            out_pred = _as_predictions(
                predictor(features[held_out]), int(held_out.sum()), "model"
            )
            residuals[held_out] = np.abs(observed[held_out] - out_pred)
            predictors.append(predictor)

        self.residuals_ = residuals
        self.fold_ids_ = fold_ids
        self.n_calibration_ = n
        self.n_splits_ = n_splits
        self.n_features_ = int(features.shape[1])
        self._predictors = predictors
        return self

    def predict_interval(self, X):
        """Return ``(lower, upper)`` arrays for new rows of ``X``."""
        if self._predictors is None:
            raise RuntimeError("fit must be called before predict_interval")
        features = _as_features(X)
        if features.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {features.shape[1]}"
            )

        n_test = features.shape[0]
        fold_predictions = np.empty((self.n_splits_, n_test), dtype=float)
        for fold, predictor in enumerate(self._predictors):
            fold_predictions[fold] = _as_predictions(
                predictor(features), n_test, "model"
            )
        loo_predictions = fold_predictions[self.fold_ids_]
        return jackknife_plus_interval(loo_predictions, self.residuals_, self.alpha)
