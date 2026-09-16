"""Tests for jackknife+ / CV+ regression intervals."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.jackknife import JackknifePlusRegressor, jackknife_plus_interval
from conformal_kit.regression import conformal_quantile


class MeanRegressor:
    """Duck-typed estimator: predict the training mean, ignore features."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), self.mean_)


def mean_trainer(X_train, y_train):
    mu = float(np.mean(y_train))

    def predict(X):
        return np.full(len(np.asarray(X)), mu)

    return predict


def ols_trainer(X_train, y_train):
    features = np.asarray(X_train, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    design = np.column_stack([np.ones(len(features)), features])
    coef, *_ = np.linalg.lstsq(design, np.asarray(y_train, dtype=float).ravel(), rcond=None)

    def predict(X):
        rows = np.asarray(X, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        return np.column_stack([np.ones(len(rows)), rows]) @ coef

    return predict


def _linear_problem(n_train, n_test, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 1))
    y = 2.5 * X[:, 0] + rng.normal(scale=noise, size=n_train + n_test)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class TestJackknifePlusInterval:
    def test_constant_predictions_match_the_conformal_quantile(self):
        residuals = np.arange(1.0, 10.0)
        lower, upper = jackknife_plus_interval(np.zeros(9), residuals, 0.1)
        half = conformal_quantile(residuals, 0.1)
        assert lower == pytest.approx([-half])
        assert upper == pytest.approx([half])

    def test_uses_the_finite_sample_corrected_rank(self):
        # n=9, alpha=0.1 -> rank 9, the largest shifted prediction.
        residuals = np.arange(1.0, 10.0)
        lower, upper = jackknife_plus_interval(np.zeros(9), residuals, 0.1)
        assert lower[0] == pytest.approx(-9.0)
        assert upper[0] == pytest.approx(9.0)

    def test_a_matrix_of_test_points_is_scored_columnwise(self):
        residuals = np.arange(1.0, 10.0)
        loo = np.column_stack([np.zeros(9), np.ones(9)])
        lower, upper = jackknife_plus_interval(loo, residuals, 0.1)
        assert lower == pytest.approx([-9.0, -8.0])
        assert upper == pytest.approx([9.0, 10.0])

    def test_too_few_points_for_the_level_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19 calibration points"):
            jackknife_plus_interval(np.zeros(5), np.arange(5.0), 0.05)

    def test_row_count_must_match_residuals(self):
        with pytest.raises(ValueError, match="rows must match"):
            jackknife_plus_interval(np.zeros((4, 2)), np.arange(5.0), 0.1)

    def test_non_finite_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            jackknife_plus_interval([0.0, np.inf] * 5, np.ones(10), 0.1)

    def test_negative_residuals_are_rejected(self):
        residuals = np.arange(1.0, 10.0)
        residuals[0] = -1.0
        with pytest.raises(ValueError, match="residuals must be nonnegative"):
            jackknife_plus_interval(np.zeros(9), residuals, 0.1)


class TestJackknifePlusRegressor:
    def test_fit_returns_self_for_chaining(self):
        X, y, _, _ = _linear_problem(40, 10)
        model = JackknifePlusRegressor(alpha=0.1)
        assert model.fit(X, y, mean_trainer) is model
        assert model.n_calibration_ == 40
        assert model.n_splits_ == 40

    def test_leave_one_out_residuals_never_see_the_held_out_point(self):
        y = np.arange(20.0)
        X = np.zeros((20, 1))
        model = JackknifePlusRegressor(alpha=0.1).fit(X, y, MeanRegressor())
        loo_means = (y.sum() - y) / (len(y) - 1)
        assert model.residuals_ == pytest.approx(np.abs(y - loo_means))

    def test_interval_width_varies_across_test_points(self):
        X_train, y_train, _, _ = _linear_problem(40, 5, seed=1)
        model = JackknifePlusRegressor(alpha=0.1).fit(X_train, y_train, ols_trainer)
        lower, upper = model.predict_interval(np.array([[0.0], [3.0], [-5.0]]))
        widths = upper - lower
        # Split conformal would share one width. Jackknife+ widths follow
        # how much the leave-one-out models disagree at x.
        assert not np.allclose(widths, widths[0])

    def test_empirical_coverage_meets_the_jackknife_plus_bound(self):
        # Theorem: at least 1 - 2*alpha = 80% at alpha=0.1.
        X_train, y_train, X_test, y_test = _linear_problem(80, 400, seed=4)
        model = JackknifePlusRegressor(alpha=0.1).fit(X_train, y_train, ols_trainer)
        lower, upper = model.predict_interval(X_test)
        covered = np.mean((y_test >= lower) & (y_test <= upper))
        assert covered >= 0.80

    def test_empirical_coverage_is_close_to_the_nominal_level(self):
        X_train, y_train, X_test, y_test = _linear_problem(120, 600, seed=5)
        model = JackknifePlusRegressor(alpha=0.1).fit(X_train, y_train, ols_trainer)
        lower, upper = model.predict_interval(X_test)
        covered = np.mean((y_test >= lower) & (y_test <= upper))
        assert covered >= 0.85

    def test_cv_plus_coverage_meets_the_bound(self):
        X_train, y_train, X_test, y_test = _linear_problem(200, 400, seed=6)
        model = JackknifePlusRegressor(alpha=0.1, n_splits=10).fit(
            X_train, y_train, ols_trainer
        )
        lower, upper = model.predict_interval(X_test)
        covered = np.mean((y_test >= lower) & (y_test <= upper))
        assert covered >= 0.80
        assert model.n_splits_ == 10

    def test_n_splits_equal_to_n_matches_leave_one_out(self):
        X, y, X_test, _ = _linear_problem(25, 8, seed=7)
        loo = JackknifePlusRegressor(alpha=0.1).fit(X, y, ols_trainer)
        cvn = JackknifePlusRegressor(alpha=0.1, n_splits=len(y)).fit(X, y, ols_trainer)
        left = loo.predict_interval(X_test)
        right = cvn.predict_interval(X_test)
        assert left[0] == pytest.approx(right[0])
        assert left[1] == pytest.approx(right[1])

    def test_a_tighter_alpha_widens_the_interval(self):
        X, y, X_test, _ = _linear_problem(60, 15, seed=8)
        wide = JackknifePlusRegressor(alpha=0.05).fit(X, y, ols_trainer)
        narrow = JackknifePlusRegressor(alpha=0.2).fit(X, y, ols_trainer)
        wide_lo, wide_hi = wide.predict_interval(X_test)
        narrow_lo, narrow_hi = narrow.predict_interval(X_test)
        assert np.mean(wide_hi - wide_lo) > np.mean(narrow_hi - narrow_lo)

    def test_noisier_data_gets_wider_intervals(self):
        X_quiet, y_quiet, X_test, _ = _linear_problem(60, 20, noise=0.1, seed=9)
        X_loud, y_loud, _, _ = _linear_problem(60, 20, noise=2.0, seed=9)
        quiet = JackknifePlusRegressor(alpha=0.1).fit(X_quiet, y_quiet, ols_trainer)
        loud = JackknifePlusRegressor(alpha=0.1).fit(X_loud, y_loud, ols_trainer)
        quiet_lo, quiet_hi = quiet.predict_interval(X_test)
        loud_lo, loud_hi = loud.predict_interval(X_test)
        assert np.mean(loud_hi - loud_lo) > np.mean(quiet_hi - quiet_lo)

    def test_a_callable_trainer_and_an_estimator_agree(self):
        y = np.linspace(-2.0, 2.0, 24)
        X = np.ones((24, 1))
        X_test = np.zeros((4, 1))
        from_callable = JackknifePlusRegressor(alpha=0.1).fit(X, y, mean_trainer)
        from_estimator = JackknifePlusRegressor(alpha=0.1).fit(X, y, MeanRegressor())
        left = from_callable.predict_interval(X_test)
        right = from_estimator.predict_interval(X_test)
        assert left[0] == pytest.approx(right[0])
        assert left[1] == pytest.approx(right[1])

    def test_one_dimensional_features_are_accepted(self):
        X_train, y_train, X_test, y_test = _linear_problem(50, 80, seed=10)
        model = JackknifePlusRegressor(alpha=0.1).fit(X_train[:, 0], y_train, ols_trainer)
        lower, upper = model.predict_interval(X_test[:, 0])
        assert lower.shape == (80,)
        covered = np.mean((y_test >= lower) & (y_test <= upper))
        assert covered >= 0.80

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            JackknifePlusRegressor().predict_interval([1.0])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            JackknifePlusRegressor().fit([[1.0], [2.0]], [1.0], mean_trainer)

    def test_feature_mismatch_at_predict_is_rejected(self):
        X, y, _, _ = _linear_problem(20, 5)
        model = JackknifePlusRegressor(alpha=0.1).fit(X, y, mean_trainer)
        with pytest.raises(ValueError, match="expected 1 features"):
            model.predict_interval(np.zeros((3, 2)))

    def test_too_few_training_points_are_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19"):
            JackknifePlusRegressor(alpha=0.05).fit(np.zeros((5, 1)), np.arange(5.0), mean_trainer)

    def test_n_splits_cannot_exceed_n(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            JackknifePlusRegressor(n_splits=20).fit(
                np.zeros((10, 1)), np.arange(10.0), mean_trainer
            )

    def test_a_callable_that_does_not_return_a_predictor_is_rejected(self):
        def bad(X, y):
            return np.mean(y)

        with pytest.raises(TypeError, match="must return a predictor"):
            JackknifePlusRegressor().fit(np.zeros((20, 1)), np.arange(20.0), bad)

    def test_a_non_model_is_rejected(self):
        with pytest.raises(ValueError, match="fit and predict"):
            JackknifePlusRegressor().fit(np.zeros((20, 1)), np.arange(20.0), object())

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1.0])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            JackknifePlusRegressor(alpha=bad)

    @pytest.mark.parametrize("bad", [1, True, 2.5, "10"])
    def test_n_splits_must_be_an_integer_at_least_two(self, bad):
        with pytest.raises(ValueError, match="n_splits"):
            JackknifePlusRegressor(n_splits=bad)
