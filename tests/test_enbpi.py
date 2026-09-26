"""Tests for Ensemble Batch Prediction Intervals (EnbPI)."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.enbpi import EnbPIRegressor, enbpi_interval, enbpi_quantile
from conformal_kit.regression import conformal_quantile


class _MeanRegressor:
    """Predicts the training-label mean for every row."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.n_features_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = X.shape[0] if X.ndim == 2 else 1
        return np.full(n, self.mean_, dtype=float)


class TestEnbpiQuantile:
    def test_matches_split_conformal_quantile(self):
        residuals = np.array([0.1, 0.4, 0.2, 0.9, 0.3])
        assert enbpi_quantile(residuals, 0.2) == pytest.approx(
            conformal_quantile(residuals, 0.2)
        )

    def test_rejects_negative_residuals(self):
        with pytest.raises(ValueError, match="nonnegative"):
            enbpi_quantile([-0.1, 0.2], 0.1)


class TestEnbpiInterval:
    def test_symmetric_around_point(self):
        lower, upper = enbpi_interval(5.0, [1.0, 2.0, 3.0, 0.5], alpha=0.25)
        assert upper - 5.0 == pytest.approx(5.0 - lower)
        assert lower < 5.0 < upper

    def test_vectorized_predictions(self):
        lower, upper = enbpi_interval([1.0, 2.0], [0.5, 1.0, 1.5, 2.0], alpha=0.2)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(upper > lower)


class TestEnbPIRegressor:
    def test_fit_predict_interval_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 2))
        y = X @ np.array([1.0, -0.5]) + 0.1 * rng.normal(size=80)
        model = EnbPIRegressor(_MeanRegressor(), alpha=0.1, n_estimators=15, random_state=0)
        model.fit(X, y)
        lower, upper = model.predict_interval(X[:10])
        assert lower.shape == (10,)
        assert upper.shape == (10,)
        assert np.all(upper >= lower)
        assert model.width > 0.0

    def test_update_slides_residual_pool(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 1))
        y = 2.0 * X[:, 0] + 0.05 * rng.normal(size=40)
        model = EnbPIRegressor(
            _MeanRegressor(), alpha=0.1, n_estimators=10, max_resid=10, random_state=1
        )
        model.fit(X, y)
        before = list(model.residuals_)
        y_hat = float(model.predict(X[-1:])[0])
        model.update(y_true=y[-1] + 5.0, y_pred=y_hat)
        after = list(model.residuals_)
        assert len(after) == len(before) == 10
        assert after[-1] == pytest.approx(abs((y[-1] + 5.0) - y_hat))
        assert after[:-1] == before[1:]

    def test_streaming_coverage_on_noisy_mean(self):
        rng = np.random.default_rng(2)
        n_train, n_test = 60, 40
        X_all = rng.normal(size=(n_train + n_test, 1))
        y_all = 0.0 * X_all[:, 0] + rng.normal(0, 1.0, size=n_train + n_test)
        model = EnbPIRegressor(
            _MeanRegressor(), alpha=0.2, n_estimators=20, max_resid=30, random_state=2
        )
        model.fit(X_all[:n_train], y_all[:n_train])
        hits = []
        for i in range(n_train, n_train + n_test):
            lower, upper = model.predict_interval(X_all[i : i + 1])
            y_hat = float(model.predict(X_all[i : i + 1])[0])
            hits.append(float(lower[0] <= y_all[i] <= upper[0]))
            model.update(y_true=y_all[i], y_pred=y_hat)
        assert np.mean(hits) >= 0.6  # soft check around 1 - alpha = 0.8

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            EnbPIRegressor(_MeanRegressor()).predict([[0.0]])
