"""Tests for conformalized quantile regression intervals."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.cqr import ConformalizedQuantileRegressor, cqr_scores
from conformal_kit.regression import SplitConformalRegressor


def _heteroskedastic(n=2000, seed=0, z=1.2815515655446004):
    """Noise that grows with x, plus quantile bounds at ``± z * sigma``.

    ``z`` defaults to the standard-normal 90% point, so the raw quantile
    interval covers about 80% and CQR has to expand it to hit 90%.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    sigma = 0.1 + 2.0 * x
    y = rng.normal(0.0, sigma)
    y_lower = -z * sigma
    y_upper = z * sigma
    return y, y_lower, y_upper, sigma


class TestCqrScores:
    def test_a_point_outside_on_the_right_scores_the_overshoot(self):
        assert cqr_scores([5.0], [0.0], [3.0]) == pytest.approx([2.0])

    def test_a_point_outside_on_the_left_scores_the_undershoot(self):
        assert cqr_scores([-2.0], [0.0], [3.0]) == pytest.approx([2.0])

    def test_a_point_inside_gets_a_negative_score(self):
        # Distance to the nearer bound is 2, so the score is -2.
        assert cqr_scores([0.0], [-2.0], [3.0]) == pytest.approx([-2.0])

    def test_identical_bounds_reduce_to_absolute_residuals(self):
        y_true = np.array([1.0, -4.0, 0.5])
        pred = np.zeros(3)
        assert cqr_scores(y_true, pred, pred) == pytest.approx(np.abs(y_true))

    def test_inverted_quantile_predictions_are_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to lower"):
            cqr_scores([0.0], [2.0], [-1.0])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            cqr_scores([0.0, 1.0], [0.0], [1.0])

    def test_empty_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            cqr_scores([], [], [])

    def test_non_finite_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            cqr_scores([0.0, np.nan], [0.0, 0.0], [1.0, 1.0])


class TestConformalizedQuantileRegressor:
    def test_uses_the_finite_sample_corrected_rank(self):
        # Identical bounds -> absolute residuals 1..9. n=9, alpha=0.1
        # -> rank 9, so the expansion is the largest residual.
        y_true = np.arange(1.0, 10.0)
        bounds = np.zeros(9)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(y_true, bounds, bounds)
        assert model.quantile_ == pytest.approx(9.0)
        lower, upper = model.predict_interval([0.0], [0.0])
        assert lower == pytest.approx([-9.0])
        assert upper == pytest.approx([9.0])

    def test_identical_bounds_match_split_conformal(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(size=200)
        y_pred = y_true + rng.normal(scale=0.4, size=200)
        cqr = ConformalizedQuantileRegressor(alpha=0.1).fit(y_true, y_pred, y_pred)
        split = SplitConformalRegressor(alpha=0.1).fit(y_true, y_pred)
        assert cqr.quantile_ == pytest.approx(split.quantile_)
        lower, upper = cqr.predict_interval(y_pred[:5], y_pred[:5])
        split_lo, split_hi = split.predict_interval(y_pred[:5])
        assert lower == pytest.approx(split_lo)
        assert upper == pytest.approx(split_hi)

    def test_expansion_is_applied_to_both_sides(self):
        y_true = np.arange(1.0, 10.0)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y_true, np.zeros(9), np.zeros(9)
        )
        lower, upper = model.predict_interval([-1.0, 2.0], [1.0, 4.0])
        assert lower == pytest.approx([-1.0 - 9.0, 2.0 - 9.0])
        assert upper == pytest.approx([1.0 + 9.0, 4.0 + 9.0])

    def test_an_overcovering_quantile_model_shrinks_the_interval(self):
        # Every label sits well inside a wide interval, so scores are
        # negative and CQR pulls the bounds inward.
        y_true = np.zeros(50)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y_true, np.full(50, -10.0), np.full(50, 10.0)
        )
        assert model.quantile_ < 0
        lower, upper = model.predict_interval([-10.0], [10.0])
        assert lower[0] > -10.0
        assert upper[0] < 10.0

    def test_fit_returns_self_for_chaining(self):
        y_true, y_lower, y_upper, _ = _heteroskedastic(n=80, seed=3)
        model = ConformalizedQuantileRegressor(alpha=0.1)
        assert model.fit(y_true, y_lower, y_upper) is model
        assert model.n_calibration_ == 80

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            ConformalizedQuantileRegressor().predict_interval([0.0], [1.0])

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            ConformalizedQuantileRegressor().fit([1.0, 2.0], [0.0], [1.0])

    def test_inverted_test_bounds_are_rejected(self):
        y_true, y_lower, y_upper, _ = _heteroskedastic(n=80, seed=4)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(y_true, y_lower, y_upper)
        with pytest.raises(ValueError, match="greater than or equal to lower"):
            model.predict_interval([1.0], [0.0])

    def test_too_few_points_for_the_level_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19 calibration points"):
            ConformalizedQuantileRegressor(alpha=0.05).fit(
                np.arange(5.0), np.zeros(5), np.zeros(5)
            )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            ConformalizedQuantileRegressor(alpha=bad)


class TestHeteroskedasticCoverage:
    def test_empirical_coverage_reaches_the_target(self):
        y, y_lower, y_upper, _ = _heteroskedastic(n=4000, seed=7)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y[:2000], y_lower[:2000], y_upper[:2000]
        )
        lower, upper = model.predict_interval(y_lower[2000:], y_upper[2000:])
        covered = np.mean((y[2000:] >= lower) & (y[2000:] <= upper))
        assert covered >= 0.88

    def test_raw_quantile_intervals_undercover_until_conformalized(self):
        y, y_lower, y_upper, _ = _heteroskedastic(n=4000, seed=8)
        raw = np.mean((y[2000:] >= y_lower[2000:]) & (y[2000:] <= y_upper[2000:]))
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y[:2000], y_lower[:2000], y_upper[:2000]
        )
        lower, upper = model.predict_interval(y_lower[2000:], y_upper[2000:])
        covered = np.mean((y[2000:] >= lower) & (y[2000:] <= upper))
        assert raw < 0.85
        assert covered >= 0.88

    def test_widths_scale_with_local_noise(self):
        y, y_lower, y_upper, _ = _heteroskedastic(n=1000, seed=9)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(y, y_lower, y_upper)
        # Three test points whose quantile width is set by sigma.
        test_sigma = np.array([0.5, 1.0, 2.0])
        z = 1.2815515655446004
        lower, upper = model.predict_interval(-z * test_sigma, z * test_sigma)
        widths = upper - lower
        # CQR adds 2 * Q to every interval, so differences in width equal
        # differences in the raw quantile width, which is linear in sigma.
        assert widths[1] - widths[0] == pytest.approx(widths[2] - widths[1])
        assert widths[2] > widths[1] > widths[0]

    def test_coverage_is_even_across_noise_regions(self):
        y, y_lower, y_upper, sigma = _heteroskedastic(n=6000, seed=11)
        cal, test = slice(0, 3000), slice(3000, 6000)
        model = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y[cal], y_lower[cal], y_upper[cal]
        )
        lower, upper = model.predict_interval(y_lower[test], y_upper[test])
        inside = (y[test] >= lower) & (y[test] <= upper)
        quiet = sigma[test] < np.median(sigma[test])
        loud = ~quiet
        assert inside[quiet].mean() >= 0.85
        assert inside[loud].mean() >= 0.85

    def test_a_tighter_alpha_widens_the_interval(self):
        y, y_lower, y_upper, _ = _heteroskedastic(n=800, seed=12)
        wide = ConformalizedQuantileRegressor(alpha=0.05).fit(y, y_lower, y_upper)
        narrow = ConformalizedQuantileRegressor(alpha=0.2).fit(y, y_lower, y_upper)
        wide_lo, wide_hi = wide.predict_interval(y_lower[:20], y_upper[:20])
        narrow_lo, narrow_hi = narrow.predict_interval(y_lower[:20], y_upper[:20])
        assert np.mean(wide_hi - wide_lo) > np.mean(narrow_hi - narrow_lo)

    def test_cqr_is_narrower_in_quiet_regions_than_split_conformal(self):
        y, y_lower, y_upper, sigma = _heteroskedastic(n=4000, seed=13)
        cal, test = slice(0, 2000), slice(2000, 4000)
        # Point predictions at the conditional mean (zero).
        split = SplitConformalRegressor(alpha=0.1).fit(y[cal], np.zeros(2000))
        cqr = ConformalizedQuantileRegressor(alpha=0.1).fit(
            y[cal], y_lower[cal], y_upper[cal]
        )
        split_lo, split_hi = split.predict_interval(np.zeros(2000))
        cqr_lo, cqr_hi = cqr.predict_interval(y_lower[test], y_upper[test])
        quiet = sigma[test] < np.quantile(sigma[test], 0.25)
        assert np.mean((cqr_hi - cqr_lo)[quiet]) < np.mean(
            (split_hi - split_lo)[quiet]
        )
