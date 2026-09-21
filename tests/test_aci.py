"""Tests for adaptive conformal inference (ACI) under streaming residuals."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.aci import (
    AdaptiveConformalClassifier,
    AdaptiveConformalRegressor,
    AdaptiveConformalUpdater,
    aci_update,
    adaptive_conformal_quantile,
)
from conformal_kit.classification import SplitConformalClassifier
from conformal_kit.regression import SplitConformalRegressor, conformal_quantile


class TestAciUpdate:
    def test_a_miss_lowers_the_level(self):
        next_level = aci_update(0.1, err=1, alpha=0.1, gamma=0.05)
        assert next_level == pytest.approx(0.1 + 0.05 * (0.1 - 1.0))
        assert next_level < 0.1

    def test_a_hit_raises_the_level(self):
        next_level = aci_update(0.1, err=0, alpha=0.1, gamma=0.05)
        assert next_level == pytest.approx(0.1 + 0.05 * 0.1)
        assert next_level > 0.1

    def test_the_recursion_telescopes_to_the_mean_error(self):
        alpha, gamma, alpha_t = 0.1, 0.05, 0.1
        errs = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        for err in errs:
            alpha_t = aci_update(alpha_t, err, alpha, gamma)
        mean_err = np.mean(errs)
        assert mean_err == pytest.approx(alpha - (alpha_t - 0.1) / (gamma * len(errs)))

    def test_a_boolean_error_is_accepted(self):
        assert aci_update(0.1, True, 0.1, 0.05) == pytest.approx(aci_update(0.1, 1, 0.1, 0.05))
        assert aci_update(0.1, False, 0.1, 0.05) == pytest.approx(aci_update(0.1, 0, 0.1, 0.05))

    def test_an_error_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="err must be a finite value"):
            aci_update(0.1, 1.5, 0.1, 0.05)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_alpha_is_validated(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            aci_update(0.1, 0, bad, 0.05)

    def test_a_non_positive_gamma_is_rejected(self):
        with pytest.raises(ValueError, match="positive finite step size"):
            aci_update(0.1, 0, 0.1, 0.0)


class TestAdaptiveConformalQuantile:
    def test_matches_conformal_quantile_inside_the_unit_interval(self):
        scores = np.arange(1.0, 10.0)
        assert adaptive_conformal_quantile(scores, 0.1) == pytest.approx(
            conformal_quantile(scores, 0.1)
        )

    def test_a_level_too_tight_for_the_window_is_infinite(self):
        # n=5, alpha=0.05 -> rank ceil(6 * 0.95) = 6 > 5.
        assert np.isposinf(adaptive_conformal_quantile(np.arange(5.0), 0.05))

    def test_a_non_positive_level_is_infinite(self):
        assert np.isposinf(adaptive_conformal_quantile([1.0, 2.0], 0.0))
        assert np.isposinf(adaptive_conformal_quantile([1.0, 2.0], -0.3))

    def test_a_level_at_least_one_is_negative_infinity(self):
        assert np.isneginf(adaptive_conformal_quantile([1.0, 2.0], 1.0))
        assert np.isneginf(adaptive_conformal_quantile([1.0, 2.0], 1.4))

    def test_an_empty_window_is_infinite(self):
        assert np.isposinf(adaptive_conformal_quantile([], 0.1))

    def test_non_finite_scores_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            adaptive_conformal_quantile([1.0, np.nan], 0.5)


class TestAdaptiveConformalUpdater:
    def test_update_returns_self_for_chaining(self):
        updater = AdaptiveConformalUpdater(alpha=0.1, gamma=0.05)
        assert updater.update(0) is updater

    def test_a_batch_of_errors_is_applied_in_order(self):
        updater = AdaptiveConformalUpdater(alpha=0.1, gamma=0.05)
        updater.update([1, 0, 1])
        expected = 0.1
        for err in (1, 0, 1):
            expected = aci_update(expected, err, 0.1, 0.05)
        assert updater.alpha_t == pytest.approx(expected)
        assert updater.n_updates_ == 3
        assert updater.empirical_miscoverage == pytest.approx(2 / 3)

    def test_empirical_coverage_is_none_before_any_update(self):
        updater = AdaptiveConformalUpdater()
        assert updater.empirical_coverage is None
        assert updater.empirical_miscoverage is None

    def test_long_run_coverage_tracks_the_target_on_iid_errors(self):
        rng = np.random.default_rng(0)
        # Independent Bernoulli(alpha) errors: alpha_t is a random walk, but
        # the telescoping identity still pins the average error to alpha.
        errs = rng.binomial(1, 0.1, 4000).astype(float)
        updater = AdaptiveConformalUpdater(alpha=0.1, gamma=0.05)
        updater.update(errs)
        assert updater.empirical_coverage == pytest.approx(0.9, abs=0.03)
        expected = 0.1 - (updater.alpha_t - 0.1) / (0.05 * len(errs))
        assert updater.empirical_miscoverage == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1.0])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            AdaptiveConformalUpdater(alpha=bad)

    def test_a_non_positive_gamma_is_rejected(self):
        with pytest.raises(ValueError, match="positive finite step size"):
            AdaptiveConformalUpdater(gamma=0.0)


def _residual_stream(n_cal, n_pre, n_post, sigma_pre=1.0, sigma_post=2.0, seed=0):
    """Calibration, pre-shift, and post-shift residual streams.

    Predictions are zero, so the labels *are* the residuals. A scale
    change in the noise is a distribution shift the frozen split-conformal
    quantile cannot track.
    """
    rng = np.random.default_rng(seed)
    y_cal = rng.normal(0.0, sigma_pre, n_cal)
    y_pre = rng.normal(0.0, sigma_pre, n_pre)
    y_post = rng.normal(0.0, sigma_post, n_post)
    return (
        y_cal,
        np.zeros(n_cal),
        y_pre,
        np.zeros(n_pre),
        y_post,
        np.zeros(n_post),
    )


class TestAdaptiveConformalRegressor:
    def test_fit_matches_split_conformal_before_any_update(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(size=200)
        y_pred = y_true + rng.normal(scale=0.4, size=200)
        aci = AdaptiveConformalRegressor(alpha=0.1).fit(y_true, y_pred)
        split = SplitConformalRegressor(alpha=0.1).fit(y_true, y_pred)
        assert aci.quantile_ == pytest.approx(split.quantile_)
        assert aci.width == pytest.approx(split.width)
        lower, upper = aci.predict_interval(y_pred[:5])
        split_lo, split_hi = split.predict_interval(y_pred[:5])
        assert lower == pytest.approx(split_lo)
        assert upper == pytest.approx(split_hi)

    def test_fit_returns_self_for_chaining(self):
        y = np.arange(50.0)
        model = AdaptiveConformalRegressor(alpha=0.1)
        assert model.fit(y, y) is model
        assert model.n_calibration_ == 50
        assert model.alpha_t == pytest.approx(0.1)

    def test_a_miss_widens_the_next_interval(self):
        y = np.arange(1.0, 10.0)
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(y, np.zeros(9))
        width_before = model.width
        # Current quantile is 9, so 100 is well outside [-9, 9].
        model.update([100.0], [0.0])
        assert model.alpha_t < 0.1
        assert model.width >= width_before

    def test_a_hit_shrinks_the_level(self):
        y = np.arange(1.0, 10.0)
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(y, np.zeros(9))
        model.update([0.0], [0.0])
        assert model.alpha_t > 0.1

    def test_predict_update_issues_intervals_before_adapting(self):
        y = np.arange(1.0, 10.0)
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(y, np.zeros(9))
        q0 = model.quantile_
        lower, upper = model.predict_update([0.0], [0.0])
        # The issued interval used the pre-update quantile.
        assert lower == pytest.approx([-q0])
        assert upper == pytest.approx([q0])
        # And the state moved afterwards.
        assert model.alpha_t > 0.1

    def test_iid_stream_coverage_reaches_the_target(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0.0, 1.0, 2500)
        pred = np.zeros(2500)
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(y[:500], pred[:500])
        lower, upper = model.predict_update(pred[500:], y[500:])
        covered = np.mean((y[500:] >= lower) & (y[500:] <= upper))
        assert covered >= 0.87
        assert model.updater.empirical_coverage == pytest.approx(covered)

    def test_normalized_width_scales_with_difficulty(self):
        rng = np.random.default_rng(3)
        sigma = np.linspace(0.1, 3.0, 400)
        y = rng.normal(0.0, sigma)
        model = AdaptiveConformalRegressor(alpha=0.1, normalize=True).fit(
            y, np.zeros(400), difficulty=sigma
        )
        lower, upper = model.predict_interval(np.zeros(3), difficulty=[1.0, 2.0, 4.0])
        widths = upper - lower
        assert widths[1] == pytest.approx(2 * widths[0])
        assert widths[2] == pytest.approx(4 * widths[0])

    def test_a_sliding_window_drops_the_oldest_scores(self):
        y = np.arange(1.0, 21.0)
        model = AdaptiveConformalRegressor(alpha=0.1, window_size=9).fit(y, np.zeros(20))
        assert model.n_calibration_ == 9
        # The window kept the last 9 residuals: 12..20. Rank 9 of those is 20.
        assert model.quantile_ == pytest.approx(20.0)

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            AdaptiveConformalRegressor().predict_interval([1.0])

    def test_reading_width_before_fitting_is_an_error(self):
        model = AdaptiveConformalRegressor()
        with pytest.raises(RuntimeError, match="fit must be called"):
            _ = model.width

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            AdaptiveConformalRegressor().fit([1.0, 2.0], [1.0])

    def test_too_few_points_for_the_level_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19 calibration points"):
            AdaptiveConformalRegressor(alpha=0.05).fit(np.arange(5.0), np.zeros(5))

    def test_difficulty_without_normalize_is_rejected(self):
        with pytest.raises(ValueError, match="difficulty requires normalize=True"):
            AdaptiveConformalRegressor().fit(np.arange(20.0), np.zeros(20), difficulty=np.ones(20))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1.0])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            AdaptiveConformalRegressor(alpha=bad)

    @pytest.mark.parametrize("bad", [True, 2.5, "10"])
    def test_window_size_must_be_an_integer(self, bad):
        with pytest.raises(TypeError, match="window_size must be an integer"):
            AdaptiveConformalRegressor(window_size=bad)

    def test_window_size_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            AdaptiveConformalRegressor(window_size=0)


class TestCoverageUnderShift:
    def test_split_conformal_undercovers_after_a_scale_shift(self):
        y_cal, pred_cal, _, _, y_post, pred_post = _residual_stream(400, 0, 2000, seed=4)
        split = SplitConformalRegressor(alpha=0.1).fit(y_cal, pred_cal)
        lower, upper = split.predict_interval(pred_post)
        covered = np.mean((y_post >= lower) & (y_post <= upper))
        assert covered < 0.80

    def test_aci_keeps_long_run_coverage_after_a_mild_scale_shift(self):
        y_cal, pred_cal, y_pre, pred_pre, y_post, pred_post = _residual_stream(
            400, 1500, 1500, sigma_pre=1.0, sigma_post=2.0, seed=5
        )
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(y_cal, pred_cal)
        y_stream = np.concatenate([y_pre, y_post])
        pred_stream = np.concatenate([pred_pre, pred_post])
        lower, upper = model.predict_update(pred_stream, y_stream)
        covered = (y_stream >= lower) & (y_stream <= upper)
        assert covered.mean() >= 0.86
        # Frozen split conformal, same calibration, fails on the shifted half.
        split = SplitConformalRegressor(alpha=0.1).fit(y_cal, pred_cal)
        split_lo, split_hi = split.predict_interval(pred_post)
        split_post = np.mean((y_post >= split_lo) & (y_post <= split_hi))
        assert split_post < 0.80
        # By the end of the shifted half ACI has adapted; the last block
        # should not stay stuck at the frozen-split failure rate.
        late = covered[-800:]
        assert late.mean() >= 0.84

    def test_aci_tracks_a_gradual_scale_drift(self):
        rng = np.random.default_rng(11)
        n_cal, n_stream = 400, 3000
        sigma = np.linspace(1.0, 1.8, n_cal + n_stream)
        y = rng.normal(0.0, sigma)
        pred = np.zeros(n_cal + n_stream)
        model = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(
            y[:n_cal], pred[:n_cal]
        )
        lower, upper = model.predict_update(pred[n_cal:], y[n_cal:])
        covered = np.mean((y[n_cal:] >= lower) & (y[n_cal:] <= upper))
        assert covered >= 0.86

        split = SplitConformalRegressor(alpha=0.1).fit(y[:n_cal], pred[:n_cal])
        split_lo, split_hi = split.predict_interval(pred[n_cal:])
        split_covered = (y[n_cal:] >= split_lo) & (y[n_cal:] <= split_hi)
        # The drift is mild, so the contrast shows up in the noisier tail.
        assert split_covered[-1000:].mean() < covered


def _classification_problem(n=4000, k=4, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    logits = rng.normal(0.0, 1.0, (n, k))
    logits[np.arange(n), labels] += signal
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return labels, probabilities


class TestAdaptiveConformalClassifier:
    def test_fit_matches_split_conformal_before_any_update(self):
        labels, probabilities = _classification_problem(n=800, seed=1)
        aci = AdaptiveConformalClassifier(alpha=0.1, method="lac").fit(labels, probabilities)
        split = SplitConformalClassifier(alpha=0.1, method="lac").fit(labels, probabilities)
        assert aci.quantile_ == pytest.approx(split.quantile_)
        assert np.array_equal(
            aci.predict_set(probabilities[:10]), split.predict_set(probabilities[:10])
        )

    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_iid_stream_coverage_reaches_the_target(self, method):
        labels, probabilities = _classification_problem(n=3000, seed=2)
        model = AdaptiveConformalClassifier(alpha=0.1, gamma=0.05, method=method).fit(
            labels[:800], probabilities[:800]
        )
        mask = model.predict_update(probabilities[800:], labels[800:])
        covered = mask[np.arange(len(labels) - 800), labels[800:]].mean()
        assert covered >= 0.87

    def test_long_run_coverage_holds_after_a_signal_drop(self):
        easy_y, easy_p = _classification_problem(n=2200, signal=2.5, seed=8)
        hard_y, hard_p = _classification_problem(n=1500, signal=0.4, seed=9)
        model = AdaptiveConformalClassifier(alpha=0.1, gamma=0.05, method="lac").fit(
            easy_y[:700], easy_p[:700]
        )
        y_stream = np.concatenate([easy_y[700:], hard_y])
        p_stream = np.concatenate([easy_p[700:], hard_p])
        mask = model.predict_update(p_stream, y_stream)
        covered = mask[np.arange(len(y_stream)), y_stream]
        assert covered.mean() >= 0.86

        frozen = SplitConformalClassifier(alpha=0.1, method="lac").fit(easy_y[:700], easy_p[:700])
        frozen_mask = frozen.predict_set(hard_p)
        frozen_hard = frozen_mask[np.arange(len(hard_y)), hard_y].mean()
        assert frozen_hard < 0.85

    def test_sets_are_never_empty(self):
        labels, probabilities = _classification_problem(n=400, k=3, seed=4)
        model = AdaptiveConformalClassifier(alpha=0.5, method="lac").fit(labels, probabilities)
        assert model.set_sizes(probabilities).min() >= 1

    def test_predict_labels_lists_the_included_indices(self):
        labels, probabilities = _classification_problem(n=300, k=4, seed=5)
        model = AdaptiveConformalClassifier(alpha=0.1).fit(labels, probabilities)
        mask = model.predict_set(probabilities[:5])
        listed = model.predict_labels(probabilities[:5])
        for row, indices in zip(mask, listed):
            assert np.flatnonzero(row).tolist() == indices

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            AdaptiveConformalClassifier().predict_set(np.array([[0.5, 0.5]]))

    def test_a_changed_class_count_is_rejected(self):
        labels, probabilities = _classification_problem(n=200, k=4, seed=6)
        model = AdaptiveConformalClassifier(alpha=0.1).fit(labels, probabilities)
        with pytest.raises(ValueError, match="expected 4 classes, got 3"):
            model.predict_set(np.full((2, 3), 1 / 3))

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method must be 'lac' or 'aps'"):
            AdaptiveConformalClassifier(method="magic")
