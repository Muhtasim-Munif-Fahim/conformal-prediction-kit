"""Tests for adaptive prediction sets and regularized APS."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit import (
    APSClassifier,
    RAPSClassifier,
    SplitConformalClassifier,
    aps_scores,
    raps_scores,
    set_coverage_report,
)


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _calibrated_problem(n=4000, k=5, scale=1.0, seed=0):
    """Labels drawn from the softmax, so the classifier is calibrated."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, scale, size=(n, k))
    probabilities = _softmax(logits)
    # Gumbel-max samples the label from that row's categorical distribution.
    labels = np.argmax(
        np.log(probabilities) + rng.gumbel(size=probabilities.shape), axis=1
    )
    return labels.astype(int), probabilities


def _split(labels, probabilities):
    half = len(labels) // 2
    return (labels[:half], probabilities[:half]), (labels[half:], probabilities[half:])


class TestScores:
    def test_aps_score_is_cumulative_mass_through_the_true_class(self):
        probabilities = np.array([[0.5, 0.3125, 0.1875]])
        # Rank 1 -> 0.5, rank 2 -> 0.8125, rank 3 -> 1.
        assert aps_scores(probabilities, np.array([0])) == pytest.approx(0.5)
        assert aps_scores(probabilities, np.array([1])) == pytest.approx(0.8125)
        assert aps_scores(probabilities, np.array([2])) == pytest.approx(1.0)

    def test_raps_adds_a_penalty_past_k_reg(self):
        probabilities = np.array([[0.5, 0.3125, 0.1875]])
        labels = np.array([2])
        # Rank 3, k_reg 1: two penalized steps.
        score = raps_scores(probabilities, labels, penalty=0.1, k_reg=1)
        assert score == pytest.approx(1.0 + 0.2)
        # The top class is inside k_reg, so it is not penalized.
        top = raps_scores(probabilities, np.array([0]), penalty=0.1, k_reg=1)
        assert top == pytest.approx(0.5)

    def test_zero_penalty_matches_aps(self):
        labels, probabilities = _calibrated_problem(n=50, k=4, seed=1)
        assert np.array_equal(
            raps_scores(probabilities, labels, penalty=0.0, k_reg=1),
            aps_scores(probabilities, labels),
        )

    def test_aps_scores_match_the_split_conformal_rule(self):
        labels, probabilities = _calibrated_problem(n=40, k=4, seed=2)
        expected = SplitConformalClassifier._aps_scores(probabilities, labels)
        assert np.array_equal(aps_scores(probabilities, labels), expected)


class TestCoverage:
    @pytest.mark.parametrize(
        "builder",
        [
            lambda: APSClassifier(alpha=0.1),
            lambda: RAPSClassifier(alpha=0.1, penalty=0.01, k_reg=1),
            lambda: RAPSClassifier(alpha=0.1, penalty=0.1, k_reg=1),
            lambda: RAPSClassifier(alpha=0.1, penalty=0.05, k_reg=2),
        ],
    )
    def test_coverage_is_near_one_minus_alpha(self, builder):
        labels, probabilities = _calibrated_problem()
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
        model = builder().fit(cal_y, cal_p)
        report = set_coverage_report(test_y, model.predict(test_p), alpha=0.1)
        # Finite-sample noise sits a little under 0.9; over-coverage is the
        # deterministic score, not a full-set trivial set.
        assert report["within_tolerance"]
        assert report["coverage"] >= 0.88
        assert report["coverage"] < 0.995
        assert 1.0 < report["mean_set_size"] < labels.max() + 1

    def test_predict_and_predict_set_are_the_same_mask(self):
        labels, probabilities = _calibrated_problem(n=200, k=4, seed=3)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        model = RAPSClassifier(alpha=0.1).fit(cal_y, cal_p)
        assert np.array_equal(model.predict(test_p), model.predict_set(test_p))


class TestRegularization:
    def test_aps_matches_split_conformal_aps(self):
        labels, probabilities = _calibrated_problem(n=300, k=4, seed=4)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        aps = APSClassifier(alpha=0.1).fit(cal_y, cal_p)
        split = SplitConformalClassifier(alpha=0.1, method="aps").fit(cal_y, cal_p)
        assert aps.quantile_ == split.quantile_
        assert np.array_equal(aps.predict_set(test_p), split.predict_set(test_p))

    def test_zero_penalty_and_large_k_reg_match_aps(self):
        labels, probabilities = _calibrated_problem(n=300, k=4, seed=5)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        aps = APSClassifier(alpha=0.1).fit(cal_y, cal_p)
        unpenalized = RAPSClassifier(alpha=0.1, penalty=0.0, k_reg=1).fit(cal_y, cal_p)
        late = RAPSClassifier(alpha=0.1, penalty=0.2, k_reg=4).fit(cal_y, cal_p)
        assert unpenalized.quantile_ == aps.quantile_
        assert late.quantile_ == aps.quantile_
        assert np.array_equal(unpenalized.predict_set(test_p), aps.predict_set(test_p))
        assert np.array_equal(late.predict_set(test_p), aps.predict_set(test_p))

    def test_penalty_drops_a_class_aps_would_keep(self):
        # Every calibration label is rank 1, so the quantile is the top-class
        # mass and does not absorb the penalty. The test row's second class
        # lands exactly on that quantile; the penalty pushes it back out.
        calibration = np.tile([0.75, 0.125, 0.0625, 0.0625], (30, 1))
        labels = np.zeros(30, dtype=int)
        test = np.array([[0.5, 0.25, 0.125, 0.125]])
        aps = APSClassifier(alpha=0.1).fit(labels, calibration)
        raps = RAPSClassifier(alpha=0.1, penalty=0.25, k_reg=1).fit(labels, calibration)
        assert aps.quantile_ == pytest.approx(0.75)
        assert raps.quantile_ == pytest.approx(0.75)
        aps_set = aps.predict_set(test)[0]
        raps_set = raps.predict_set(test)[0]
        assert aps_set.tolist() == [True, True, False, False]
        assert raps_set.tolist() == [True, False, False, False]
        assert raps.set_sizes(test)[0] < aps.set_sizes(test)[0]

    def test_a_harsher_penalty_does_not_grow_sets(self):
        labels, probabilities = _calibrated_problem(n=2000, k=6, scale=0.7, seed=6)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        mild = RAPSClassifier(alpha=0.1, penalty=0.01, k_reg=1).fit(cal_y, cal_p)
        harsh = RAPSClassifier(alpha=0.1, penalty=0.5, k_reg=1).fit(cal_y, cal_p)
        assert harsh.set_sizes(test_p).mean() <= mild.set_sizes(test_p).mean()


class TestSetShape:
    @pytest.mark.parametrize("factory", [APSClassifier, RAPSClassifier])
    def test_sets_are_never_empty(self, factory):
        labels, probabilities = _calibrated_problem(n=400, k=4, seed=7)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        model = factory(alpha=0.5).fit(cal_y, cal_p)
        assert model.set_sizes(test_p).min() >= 1

    def test_sets_are_contiguous_in_probability_rank(self):
        labels, probabilities = _calibrated_problem(n=400, k=6, seed=8)
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        model = RAPSClassifier(alpha=0.1, penalty=0.05, k_reg=1).fit(cal_y, cal_p)
        mask = model.predict_set(test_p[:40])
        for row_mask, row_p in zip(mask, test_p[:40]):
            order = np.argsort(-row_p)
            included = row_mask[order]
            # Once a rank is excluded, no later rank is included.
            assert not np.any(np.diff(included.astype(int)) > 0)

    def test_predict_labels_lists_the_included_indices(self):
        labels, probabilities = _calibrated_problem(n=80, k=4, seed=9)
        model = APSClassifier(alpha=0.1).fit(labels, probabilities)
        mask = model.predict_set(probabilities[:5])
        listed = model.predict_labels(probabilities[:5])
        for row, indices in zip(mask, listed):
            assert np.flatnonzero(row).tolist() == indices

    def test_set_sizes_matches_the_mask(self):
        labels, probabilities = _calibrated_problem(n=80, k=4, seed=10)
        model = RAPSClassifier(alpha=0.1).fit(labels, probabilities)
        assert np.array_equal(
            model.set_sizes(probabilities),
            model.predict_set(probabilities).sum(axis=1),
        )


class TestValidation:
    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            RAPSClassifier().predict(np.array([[0.5, 0.5]]))
        with pytest.raises(RuntimeError, match="fit must be called"):
            APSClassifier().predict_set(np.array([[0.5, 0.5]]))

    def test_a_changed_class_count_is_rejected(self):
        labels, probabilities = _calibrated_problem(n=40, k=4, seed=11)
        model = APSClassifier(alpha=0.1).fit(labels, probabilities)
        with pytest.raises(ValueError, match="expected 4 classes, got 3"):
            model.predict(np.full((2, 3), 1 / 3))

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            APSClassifier().fit(np.array([0, 1, 0]), np.full((2, 2), 0.5))

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            RAPSClassifier().fit(np.array([0.0, 1.0]), np.full((2, 2), 0.5))

    def test_an_out_of_range_label_is_rejected(self):
        with pytest.raises(ValueError, match="outside the probability matrix"):
            APSClassifier().fit(np.array([0, 7]), np.full((2, 2), 0.5))

    def test_a_one_dimensional_probability_array_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            RAPSClassifier().fit(np.array([0, 1]), np.array([0.5, 0.5]))

    def test_probabilities_outside_the_unit_interval_are_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            APSClassifier().fit(np.array([0, 1]), np.array([[1.5, -0.5]] * 2))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            RAPSClassifier(alpha=bad)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_penalty_must_be_a_non_negative_finite_number(self, bad):
        with pytest.raises(ValueError, match="penalty must be"):
            RAPSClassifier(penalty=bad)

    @pytest.mark.parametrize("bad", [1.5, True, "1"])
    def test_k_reg_rejects_a_non_integer_type(self, bad):
        with pytest.raises(TypeError, match="k_reg must be"):
            RAPSClassifier(k_reg=bad)

    def test_a_negative_k_reg_is_rejected(self):
        with pytest.raises(ValueError, match="k_reg must be"):
            RAPSClassifier(k_reg=-1)

    def test_too_few_calibration_points_for_the_level_is_an_error(self):
        labels = np.zeros(3, dtype=int)
        probabilities = np.tile([0.7, 0.3], (3, 1))
        with pytest.raises(ValueError, match="calibration points"):
            APSClassifier(alpha=0.01).fit(labels, probabilities)
