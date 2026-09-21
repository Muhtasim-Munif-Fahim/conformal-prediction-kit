"""Tests for split-conformal regression intervals."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.regression import SplitConformalRegressor, conformal_quantile


class TestConformalQuantile:
    def test_uses_the_finite_sample_corrected_rank(self):
        # n=9, alpha=0.1 -> rank ceil(10 * 0.9) = 9, the largest score.
        scores = np.arange(1.0, 10.0)
        assert conformal_quantile(scores, 0.1) == 9.0

    def test_correction_is_above_the_plain_empirical_quantile(self):
        rng = np.random.default_rng(0)
        scores = rng.random(50)
        assert conformal_quantile(scores, 0.1) >= float(np.quantile(scores, 0.9))

    def test_a_smaller_alpha_gives_a_larger_quantile(self):
        scores = np.arange(1.0, 101.0)
        assert conformal_quantile(scores, 0.05) >= conformal_quantile(scores, 0.2)

    def test_too_few_points_for_the_level_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19 calibration points"):
            conformal_quantile(np.arange(5.0), 0.05)

    def test_the_boundary_sample_size_is_accepted(self):
        # alpha=0.1 needs n >= 9.
        assert conformal_quantile(np.arange(9.0), 0.1) == 8.0

    def test_empty_scores_are_rejected(self):
        with pytest.raises(ValueError, match="at least one calibration score"):
            conformal_quantile([], 0.1)

    def test_non_finite_scores_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            conformal_quantile([1.0, np.nan], 0.1)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_must_be_a_strict_fraction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            conformal_quantile([1.0, 2.0], bad)


class TestSplitConformalRegressor:
    def _calibrated(self, alpha=0.1, n=500, seed=0):
        rng = np.random.default_rng(seed)
        truth = rng.normal(0.0, 1.0, n)
        predictions = truth + rng.normal(0.0, 0.5, n)
        return SplitConformalRegressor(alpha=alpha).fit(truth, predictions), rng

    def test_intervals_are_centred_on_the_prediction(self):
        model, _ = self._calibrated()
        lower, upper = model.predict_interval([0.0, 5.0])
        assert (lower + upper) / 2 == pytest.approx([0.0, 5.0])

    def test_every_interval_has_the_same_width_unnormalized(self):
        model, _ = self._calibrated()
        lower, upper = model.predict_interval([0.0, 5.0, -3.0])
        widths = upper - lower
        assert widths == pytest.approx(widths[0])
        assert model.width == pytest.approx(widths[0])

    def test_empirical_coverage_reaches_the_target(self):
        rng = np.random.default_rng(7)
        truth = rng.normal(0.0, 1.0, 2000)
        predictions = truth + rng.normal(0.0, 0.5, 2000)
        model = SplitConformalRegressor(alpha=0.1).fit(truth[:1000], predictions[:1000])
        lower, upper = model.predict_interval(predictions[1000:])
        covered = np.mean((truth[1000:] >= lower) & (truth[1000:] <= upper))
        assert covered >= 0.88

    def test_a_tighter_alpha_widens_the_interval(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(0.0, 1.0, 500)
        predictions = truth + rng.normal(0.0, 0.5, 500)
        wide = SplitConformalRegressor(alpha=0.01).fit(truth, predictions).width
        narrow = SplitConformalRegressor(alpha=0.2).fit(truth, predictions).width
        assert wide > narrow

    def test_a_worse_model_gets_wider_intervals(self):
        rng = np.random.default_rng(2)
        truth = rng.normal(0.0, 1.0, 500)
        good = SplitConformalRegressor(alpha=0.1).fit(truth, truth + rng.normal(0, 0.1, 500))
        bad = SplitConformalRegressor(alpha=0.1).fit(truth, truth + rng.normal(0, 2.0, 500))
        assert bad.width > good.width

    def test_a_perfect_model_gets_a_zero_width_interval(self):
        truth = np.arange(50.0)
        model = SplitConformalRegressor(alpha=0.1).fit(truth, truth)
        assert model.width == 0.0

    def test_fit_returns_self_for_chaining(self):
        truth = np.arange(50.0)
        model = SplitConformalRegressor(alpha=0.1)
        assert model.fit(truth, truth) is model
        assert model.n_calibration_ == 50

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            SplitConformalRegressor().predict_interval([1.0])

    def test_reading_width_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            _ = SplitConformalRegressor().width

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            SplitConformalRegressor().fit([1.0, 2.0], [1.0])

    def test_non_finite_calibration_values_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            SplitConformalRegressor().fit([1.0, np.nan] * 10, [1.0, 1.0] * 10)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1.0])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            SplitConformalRegressor(alpha=bad)


class TestNormalizedIntervals:
    def _heteroskedastic(self, n=1000, seed=3):
        rng = np.random.default_rng(seed)
        sigma = np.linspace(0.1, 3.0, n)
        truth = rng.normal(0.0, sigma)
        return truth, np.zeros(n), sigma

    def test_width_scales_with_the_difficulty_estimate(self):
        truth, predictions, sigma = self._heteroskedastic()
        model = SplitConformalRegressor(alpha=0.1, normalize=True).fit(
            truth, predictions, difficulty=sigma
        )
        lower, upper = model.predict_interval(np.zeros(3), difficulty=[1.0, 2.0, 4.0])
        widths = upper - lower
        assert widths[1] == pytest.approx(2 * widths[0])
        assert widths[2] == pytest.approx(4 * widths[0])

    def test_normalized_coverage_still_reaches_the_target(self):
        truth, predictions, sigma = self._heteroskedastic(n=2000)
        model = SplitConformalRegressor(alpha=0.1, normalize=True).fit(
            truth[:1000], predictions[:1000], difficulty=sigma[:1000]
        )
        lower, upper = model.predict_interval(predictions[1000:], difficulty=sigma[1000:])
        covered = np.mean((truth[1000:] >= lower) & (truth[1000:] <= upper))
        assert covered >= 0.85

    def test_normalize_without_difficulty_is_rejected(self):
        truth, predictions, _ = self._heteroskedastic()
        with pytest.raises(ValueError, match="requires difficulty estimates"):
            SplitConformalRegressor(normalize=True).fit(truth, predictions)

    def test_difficulty_without_normalize_is_rejected(self):
        truth, predictions, sigma = self._heteroskedastic()
        with pytest.raises(ValueError, match="difficulty requires normalize=True"):
            SplitConformalRegressor(normalize=False).fit(truth, predictions, difficulty=sigma)

    def test_a_zero_difficulty_is_rejected_rather_than_dividing_through(self):
        truth, predictions, sigma = self._heteroskedastic()
        sigma = sigma.copy()
        sigma[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            SplitConformalRegressor(normalize=True).fit(truth, predictions, difficulty=sigma)

    def test_mismatched_difficulty_length_is_rejected(self):
        truth, predictions, sigma = self._heteroskedastic()
        with pytest.raises(ValueError, match="same length"):
            SplitConformalRegressor(normalize=True).fit(
                truth, predictions, difficulty=sigma[:10]
            )
