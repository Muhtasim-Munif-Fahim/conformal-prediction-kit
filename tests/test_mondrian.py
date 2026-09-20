"""Tests for Mondrian (class-conditional) conformal prediction sets."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.classification import (
    MondrianConformalClassifier,
    SplitConformalClassifier,
    mondrian_quantiles,
)
from conformal_kit.evaluation import set_coverage_report
from conformal_kit.regression import conformal_quantile


def _problem(n=4000, k=5, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    logits = rng.normal(0.0, 1.0, (n, k))
    logits[np.arange(n), labels] += signal
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return labels, probabilities


def _split(labels, probabilities):
    half = len(labels) // 2
    return (labels[:half], probabilities[:half]), (labels[half:], probabilities[half:])


def _uneven_problem(n=8000, seed=0):
    """Two classes: class 0 is easy, class 1 is hard.

    Split conformal pools their scores into one threshold and under-covers
    the hard class. Mondrian gives the hard class its own, looser quantile.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    logits = rng.normal(0.0, 0.3, (n, 2))
    logits[labels == 0, 0] += 3.0
    logits[labels == 1, 1] += 0.25
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return labels, probabilities


class TestMondrianQuantiles:
    def test_each_class_uses_its_own_conformal_quantile(self):
        # n=9, alpha=0.1 -> rank ceil(10 * 0.9) = 9, so the quantile is the max.
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2)
        labels = np.array([0] * 9 + [1] * 9)
        scores[9:] = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        quantiles = mondrian_quantiles(scores, labels, alpha=0.1)
        assert quantiles == pytest.approx([0.9, 0.09])

    def test_matches_calling_conformal_quantile_per_class(self):
        rng = np.random.default_rng(3)
        scores = rng.uniform(0.0, 1.0, 300)
        labels = rng.integers(0, 3, 300)
        quantiles = mondrian_quantiles(scores, labels, alpha=0.1)
        for cls in range(3):
            assert quantiles[cls] == pytest.approx(
                conformal_quantile(scores[labels == cls], 0.1)
            )

    def test_n_classes_detects_a_trailing_missing_class(self):
        with pytest.raises(ValueError, match="class 2 has no calibration examples"):
            mondrian_quantiles([0.1, 0.2], [0, 1], alpha=0.5, n_classes=3)

    def test_a_class_with_too_few_examples_names_the_class(self):
        scores = np.concatenate([np.linspace(0.1, 0.9, 9), [0.5]])
        labels = np.array([0] * 9 + [1])
        with pytest.raises(ValueError, match="class 1 needs at least 9"):
            mondrian_quantiles(scores, labels, alpha=0.1, n_classes=2)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            mondrian_quantiles([0.1, 0.2], [0], alpha=0.5)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least one calibration score"):
            mondrian_quantiles([], [], alpha=0.5)

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            mondrian_quantiles([0.1, 0.2], [0.0, 1.0], alpha=0.5)


class TestCoverage:
    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_marginal_coverage_meets_the_target(self, method):
        labels, probabilities = _problem()
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
        model = MondrianConformalClassifier(alpha=0.1, method=method).fit(cal_y, cal_p)
        mask = model.predict_set(test_p)
        covered = mask[np.arange(len(test_y)), test_y].mean()
        assert covered >= 0.88

    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_every_class_meets_the_target(self, method):
        # One calibration split can land a class a bit below 1 - alpha; the
        # guarantee is over the draw of the calibration set, so average it.
        per_class = {cls: [] for cls in range(4)}
        for seed in range(6):
            labels, probabilities = _problem(n=4000, k=4, seed=seed)
            (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
            model = MondrianConformalClassifier(alpha=0.1, method=method).fit(cal_y, cal_p)
            mask = model.predict_set(test_p)
            inside = mask[np.arange(len(test_y)), test_y]
            for cls in range(4):
                per_class[cls].append(inside[test_y == cls].mean())
        for values in per_class.values():
            assert np.mean(values) >= 0.88

    def test_set_coverage_report_breaks_coverage_down_by_class(self):
        labels, probabilities = _problem(n=4000, k=3, seed=1)
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
        model = MondrianConformalClassifier(alpha=0.1, method="lac").fit(cal_y, cal_p)
        report = set_coverage_report(
            test_y, model.predict_set(test_p), alpha=0.1, groups=test_y
        )
        assert {row["group"] for row in report["by_group"]} == {0, 1, 2}
        assert report["coverage"] >= 0.88

    def test_split_conformal_can_miss_a_hard_class_that_mondrian_covers(self):
        labels, probabilities = _uneven_problem()
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)

        split = SplitConformalClassifier(alpha=0.1, method="lac").fit(cal_y, cal_p)
        mondrian = MondrianConformalClassifier(alpha=0.1, method="lac").fit(cal_y, cal_p)

        split_mask = split.predict_set(test_p)
        mondrian_mask = mondrian.predict_set(test_p)
        hard = test_y == 1
        split_hard = split_mask[np.arange(len(test_y)), test_y][hard].mean()
        mondrian_hard = mondrian_mask[np.arange(len(test_y)), test_y][hard].mean()

        assert split_hard < 0.85
        assert mondrian_hard >= 0.88

    def test_the_hard_class_gets_a_larger_threshold(self):
        labels, probabilities = _uneven_problem()
        (cal_y, cal_p), _ = _split(labels, probabilities)
        model = MondrianConformalClassifier(alpha=0.1, method="lac").fit(cal_y, cal_p)
        assert model.quantiles_[1] > model.quantiles_[0]

    def test_lac_gives_smaller_sets_than_aps(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        sizes = {}
        for method in ("lac", "aps"):
            model = MondrianConformalClassifier(alpha=0.1, method=method).fit(cal_y, cal_p)
            sizes[method] = model.set_sizes(test_p).mean()
        assert sizes["lac"] < sizes["aps"]

    def test_a_tighter_alpha_gives_larger_sets(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        loose = MondrianConformalClassifier(alpha=0.2).fit(cal_y, cal_p).set_sizes(test_p).mean()
        tight = MondrianConformalClassifier(alpha=0.01).fit(cal_y, cal_p).set_sizes(test_p).mean()
        assert tight > loose


class TestSetShape:
    def test_sets_are_never_empty(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        for method in ("lac", "aps"):
            model = MondrianConformalClassifier(alpha=0.5, method=method).fit(cal_y, cal_p)
            assert model.set_sizes(test_p).min() >= 1

    def test_mask_shape_matches_the_input(self):
        labels, probabilities = _problem(n=200, k=4)
        model = MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)
        assert model.predict_set(probabilities).shape == (200, 4)

    def test_predict_labels_lists_the_included_indices(self):
        labels, probabilities = _problem(n=200, k=4)
        model = MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)
        mask = model.predict_set(probabilities[:5])
        listed = model.predict_labels(probabilities[:5])
        for row, indices in zip(mask, listed):
            assert np.flatnonzero(row).tolist() == indices

    def test_set_sizes_matches_the_mask(self):
        labels, probabilities = _problem(n=300, k=4)
        model = MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)
        assert np.array_equal(
            model.set_sizes(probabilities), model.predict_set(probabilities).sum(axis=1)
        )

    def test_quantiles_and_counts_are_one_per_class(self):
        labels, probabilities = _problem(n=400, k=4)
        model = MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)
        assert model.quantiles_.shape == (4,)
        assert model.n_classes_ == 4
        assert model.n_calibration_ == 400
        assert model.n_calibration_per_class_.tolist() == np.bincount(labels, minlength=4).tolist()

    def test_lac_uses_each_class_threshold_independently(self):
        # n=9 per class, alpha=0.1 -> each quantile is that class's max LAC score.
        scores0 = np.linspace(0.1, 0.9, 9)
        scores1 = np.linspace(0.01, 0.09, 9)
        labels = np.array([0] * 9 + [1] * 9)
        probabilities = np.zeros((18, 2))
        probabilities[:9, 0] = 1.0 - scores0
        probabilities[:9, 1] = scores0
        probabilities[9:, 1] = 1.0 - scores1
        probabilities[9:, 0] = scores1
        model = MondrianConformalClassifier(alpha=0.1, method="lac").fit(labels, probabilities)
        assert model.quantiles_ == pytest.approx([0.9, 0.09])

        # Class 0 is included at p0 >= 0.1; class 1 at p1 >= 0.91.
        test = np.array([[0.15, 0.85], [0.05, 0.95], [0.50, 0.50]])
        mask = model.predict_set(test)
        assert mask[0].tolist() == [True, False]
        assert mask[1].tolist() == [False, True]
        assert mask[2].tolist() == [True, False]


class TestValidation:
    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            MondrianConformalClassifier().predict_set(np.array([[0.5, 0.5]]))

    def test_a_changed_class_count_is_rejected(self):
        labels, probabilities = _problem(n=200, k=4)
        model = MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)
        with pytest.raises(ValueError, match="expected 4 classes, got 3"):
            model.predict_set(np.full((2, 3), 1 / 3))

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            MondrianConformalClassifier().fit(np.array([0, 1, 0]), np.full((2, 2), 0.5))

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            MondrianConformalClassifier().fit(np.array([0.0, 1.0]), np.full((2, 2), 0.5))

    def test_an_out_of_range_label_is_rejected(self):
        with pytest.raises(ValueError, match="outside the probability matrix"):
            MondrianConformalClassifier().fit(np.array([0, 7]), np.full((2, 2), 0.5))

    def test_a_missing_class_is_rejected(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
        probabilities = np.full((9, 3), 1 / 3)
        with pytest.raises(ValueError, match="class 2 has no calibration examples"):
            MondrianConformalClassifier(alpha=0.5).fit(labels, probabilities)

    def test_a_class_with_too_few_examples_is_rejected(self):
        labels = np.array([0] * 20 + [1] * 3)
        probabilities = np.full((23, 2), 0.5)
        with pytest.raises(ValueError, match="class 1 needs at least 9"):
            MondrianConformalClassifier(alpha=0.1).fit(labels, probabilities)

    def test_a_one_dimensional_probability_array_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            MondrianConformalClassifier().fit(np.array([0, 1]), np.array([0.5, 0.5]))

    def test_a_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            MondrianConformalClassifier().fit(np.array([0, 0]), np.ones((2, 1)))

    def test_probabilities_outside_the_unit_interval_are_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            MondrianConformalClassifier().fit(np.array([0, 1]), np.array([[1.5, -0.5]] * 2))

    def test_non_finite_probabilities_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            MondrianConformalClassifier().fit(np.array([0, 1]), np.array([[np.nan, 1.0]] * 2))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            MondrianConformalClassifier(alpha=bad)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method must be 'lac' or 'aps'"):
            MondrianConformalClassifier(method="magic")
