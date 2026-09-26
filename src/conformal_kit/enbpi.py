"""Ensemble Batch Prediction Intervals (EnbPI) for sequential conformal prediction.

Split conformal, CQR and jackknife+ assume the calibration and test points are
exchangeable. Time series break that assumption: residuals drift, and a frozen
quantile under-covers once the process shifts.

Ensemble Batch Prediction Intervals (Xu and Xie, 2021) keep a sliding pool of
*leave-one-out style* residuals from an ensemble of bootstrap models and rebuild
the prediction interval at every step from the most recent residuals. After each
new outcome arrives the oldest residual is dropped and the fresh residual is
appended, so the interval adapts without refitting the whole ensemble.

This module is that construction:

* :func:`enbpi_interval` builds ``(lower, upper)`` from a point prediction and a
  residual pool at level ``1 - alpha``.
* :class:`EnbPIRegressor` fits bootstrap ensemble members, records in-sample
  leave-one-bootstrap residuals, and exposes streaming
  :meth:`~EnbPIRegressor.predict_interval` /
  :meth:`~EnbPIRegressor.update` for sequential use.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

from .regression import conformal_quantile

__all__ = ["EnbPIRegressor", "enbpi_interval", "enbpi_quantile"]


def enbpi_quantile(residuals, alpha):
    """Finite-sample ``1 - alpha`` residual quantile used by EnbPI.

    Residuals are treated as absolute nonconformity scores. The returned value
    ``q`` is the same conformal quantile used by split conformal, so the
    interval ``[yhat - q, yhat + q]`` has the usual finite-sample coverage
    guarantee when residuals are exchangeable with the next error. EnbPI keeps
    that construction but refreshes the residual pool over time.
    """
    scores = np.asarray(residuals, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("at least one residual is required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("residuals must all be finite")
    if np.any(scores < 0):
        raise ValueError("residuals must be nonnegative")
    return conformal_quantile(scores, alpha)


def enbpi_interval(y_hat, residuals, alpha):
    """Return EnbPI ``(lower, upper)`` for point prediction(s) ``y_hat``.

    ``residuals`` is a 1-D pool of nonnegative absolute residuals. ``y_hat`` may
    be a scalar or an array; the same half-width is applied to every entry.
    """
    q = enbpi_quantile(residuals, alpha)
    point = np.asarray(y_hat, dtype=float)
    if point.ndim > 1:
        raise ValueError("y_hat must be a scalar or 1-D array")
    lower = point - q
    upper = point + q
    if point.ndim == 0:
        return float(lower), float(upper)
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


def _as_target(y, n_rows):
    array = np.asarray(y, dtype=float).ravel()
    if array.shape[0] != n_rows:
        raise ValueError("y must have one value per row of X")
    if not np.all(np.isfinite(array)):
        raise ValueError("y values must all be finite")
    return array


def _as_trainer(model):
    if hasattr(model, "fit") and hasattr(model, "predict"):

        def train(X_train, y_train, _proto=model):
            fitted = copy.deepcopy(_proto)
            fitted.fit(X_train, y_train)
            return fitted

        return train
    if callable(model):

        def train(X_train, y_train, _factory=model):
            fitted = _factory(X_train, y_train)
            if not hasattr(fitted, "predict"):
                raise TypeError(
                    "callable model must return an estimator with predict(X)"
                )
            return fitted

        return train
    raise ValueError(
        "model must be an estimator with fit and predict, "
        "or a callable (X, y) -> estimator"
    )


class EnbPIRegressor:
    """Bootstrap-ensemble EnbPI for streaming regression intervals.

    Parameters
    ----------
    model :
        Template estimator with ``fit`` / ``predict``, or a callable
        ``(X, y) -> estimator``.
    alpha :
        Target miscoverage level. Intervals aim for coverage ``1 - alpha``.
    n_estimators :
        Number of bootstrap ensemble members.
    max_resid :
        Length of the sliding residual pool. ``None`` keeps every in-sample
        LOO residual from training (and still slides as :meth:`update` is
        called when ``max_resid`` is set).
    random_state :
        Seed for bootstrap draws.
    """

    def __init__(
        self,
        model,
        alpha=0.1,
        n_estimators=20,
        max_resid=None,
        random_state=None,
    ):
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be a positive integer")
        if max_resid is not None and int(max_resid) < 1:
            raise ValueError("max_resid must be a positive integer or None")
        self.model = model
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.max_resid = None if max_resid is None else int(max_resid)
        self.random_state = random_state
        self.estimators_ = None
        self.residuals_ = None
        self.n_features_in_ = None
        self._train = _as_trainer(model)

    def fit(self, X, y):
        """Fit the bootstrap ensemble and initialise the residual pool."""
        X = _as_features(X)
        y = _as_target(y, X.shape[0])
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        estimators = []
        # Leave-one-bootstrap aggregate: for each training row, average the
        # predictions of members that did *not* draw that row.
        agg = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=float)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            fitted = self._train(X[idx], y[idx])
            estimators.append(fitted)
            oob = np.ones(n, dtype=bool)
            oob[idx] = False
            if not np.any(oob):
                # Extremely small n can draw every index; fall back to all rows.
                oob[:] = True
            pred = np.asarray(fitted.predict(X[oob]), dtype=float).ravel()
            if pred.shape[0] != int(oob.sum()):
                raise ValueError("base estimator predict returned the wrong shape")
            agg[oob] += pred
            counts[oob] += 1.0

        missing = counts == 0
        if np.any(missing):
            # Rows never left out: use the full-ensemble mean as a fallback.
            full = np.mean(
                [np.asarray(est.predict(X), dtype=float).ravel() for est in estimators],
                axis=0,
            )
            agg[missing] = full[missing]
            counts[missing] = 1.0
        loo_pred = agg / counts
        residuals = np.abs(y - loo_pred)
        if self.max_resid is not None and residuals.size > self.max_resid:
            residuals = residuals[-self.max_resid :]

        self.estimators_ = estimators
        self.residuals_ = deque(residuals.tolist(), maxlen=self.max_resid)
        self.n_features_in_ = int(X.shape[1])
        self.n_train_ = n
        return self

    def _check_fitted(self):
        if self.estimators_ is None or self.residuals_ is None:
            raise RuntimeError("EnbPIRegressor is not fitted yet")

    def predict(self, X):
        """Ensemble-mean point prediction."""
        self._check_fitted()
        X = _as_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but EnbPIRegressor is expecting "
                f"{self.n_features_in_} features as input"
            )
        preds = np.column_stack(
            [np.asarray(est.predict(X), dtype=float).ravel() for est in self.estimators_]
        )
        return preds.mean(axis=1)

    def predict_interval(self, X):
        """Return ``(lower, upper)`` intervals from the current residual pool."""
        y_hat = self.predict(X)
        lower, upper = enbpi_interval(y_hat, np.asarray(self.residuals_, dtype=float), self.alpha)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def update(self, y_true, y_pred=None):
        """Slide the residual pool after observing a new outcome.

        ``y_true`` is the realized label. ``y_pred`` defaults to the most recent
        ensemble prediction you already computed; pass it explicitly when the
        interval was built from a known point forecast.
        """
        self._check_fitted()
        if y_pred is None:
            raise ValueError("y_pred is required when updating the residual pool")
        resid = abs(float(y_true) - float(y_pred))
        if not np.isfinite(resid):
            raise ValueError("residual must be finite")
        self.residuals_.append(resid)
        return resid

    @property
    def width(self):
        """Current interval half-width ``2 * q`` from the residual pool."""
        self._check_fitted()
        q = enbpi_quantile(np.asarray(self.residuals_, dtype=float), self.alpha)
        return 2.0 * q
