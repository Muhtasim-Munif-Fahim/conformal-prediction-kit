"""Tests for split-conformal classification prediction sets."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.classification import SplitConformalClassifier


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


class TestCoverage:
    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_marginal_coverage_meets_the_target(self, method):
        labels, probabilities = _problem()
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
        model = SplitConformalClassifier(alpha=0.1, method=method).fit(cal_y, cal_p)
        mask = model.predict_set(test_p)
        covered = mask[np.arange(len(test_y)), test_y].mean()
        assert covered >= 0.88

    def test_lac_tracks_the_target_closely(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (test_y, test_p) = _split(labels, probabilities)
        model = SplitConformalClassifier(alpha=0.1, method="lac").fit(cal_y, cal_p)
        mask = model.predict_set(test_p)
        covered = mask[np.arange(len(test_y)), test_y].mean()
        assert 0.88 <= covered <= 0.93

    def test_lac_gives_smaller_sets_than_aps(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        sizes = {}
        for method in ("lac", "aps"):
            model = SplitConformalClassifier(alpha=0.1, method=method).fit(cal_y, cal_p)
            sizes[method] = model.set_sizes(test_p).mean()
        assert sizes["lac"] < sizes["aps"]

    def test_a_tighter_alpha_gives_larger_sets(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        loose = SplitConformalClassifier(alpha=0.2).fit(cal_y, cal_p).set_sizes(test_p).mean()
        tight = SplitConformalClassifier(alpha=0.01).fit(cal_y, cal_p).set_sizes(test_p).mean()
        assert tight > loose

    def test_a_harder_problem_gives_larger_sets(self):
        easy_y, easy_p = _problem(signal=5.0, seed=1)
        hard_y, hard_p = _problem(signal=0.5, seed=1)
        easy = SplitConformalClassifier(alpha=0.1).fit(easy_y, easy_p).set_sizes(easy_p).mean()
        hard = SplitConformalClassifier(alpha=0.1).fit(hard_y, hard_p).set_sizes(hard_p).mean()
        assert hard > easy


class TestSetShape:
    def test_sets_are_never_empty(self):
        labels, probabilities = _problem()
        (cal_y, cal_p), (_, test_p) = _split(labels, probabilities)
        for method in ("lac", "aps"):
            model = SplitConformalClassifier(alpha=0.5, method=method).fit(cal_y, cal_p)
            assert model.set_sizes(test_p).min() >= 1

    def test_mask_shape_matches_the_input(self):
        labels, probabilities = _problem(n=200, k=4)
        model = SplitConformalClassifier(alpha=0.1).fit(labels, probabilities)
        assert model.predict_set(probabilities).shape == (200, 4)

    def test_predict_labels_lists_the_included_indices(self):
        labels, probabilities = _problem(n=200, k=4)
        model = SplitConformalClassifier(alpha=0.1).fit(labels, probabilities)
        mask = model.predict_set(probabilities[:5])
        listed = model.predict_labels(probabilities[:5])
        for row, indices in zip(mask, listed):
            assert np.flatnonzero(row).tolist() == indices

    def test_aps_sets_are_contiguous_in_probability_rank(self):
        # APS sweeps classes in descending probability, so a set must never
        # skip a class more likely than one it contains.
        labels, probabilities = _problem(n=500, k=6)
        model = SplitConformalClassifier(alpha=0.1, method="aps").fit(labels, probabilities)
        mask = model.predict_set(probabilities[:50])
        for row_mask, row_p in zip(mask, probabilities[:50]):
            order = np.argsort(-row_p)
            included = row_mask[order]
            assert not np.any(np.diff(included.astype(int)) > 0)

    def test_set_sizes_matches_the_mask(self):
        labels, probabilities = _problem(n=300, k=4)
        model = SplitConformalClassifier(alpha=0.1).fit(labels, probabilities)
        assert np.array_equal(
            model.set_sizes(probabilities), model.predict_set(probabilities).sum(axis=1)
        )


class TestValidation:
    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            SplitConformalClassifier().predict_set(np.array([[0.5, 0.5]]))

    def test_a_changed_class_count_is_rejected(self):
        labels, probabilities = _problem(n=200, k=4)
        model = SplitConformalClassifier(alpha=0.1).fit(labels, probabilities)
        with pytest.raises(ValueError, match="expected 4 classes, got 3"):
            model.predict_set(np.full((2, 3), 1 / 3))

    def test_mismatched_calibration_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            SplitConformalClassifier().fit(np.array([0, 1, 0]), np.full((2, 2), 0.5))

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            SplitConformalClassifier().fit(np.array([0.0, 1.0]), np.full((2, 2), 0.5))

    def test_an_out_of_range_label_is_rejected(self):
        with pytest.raises(ValueError, match="outside the probability matrix"):
            SplitConformalClassifier().fit(np.array([0, 7]), np.full((2, 2), 0.5))

    def test_a_one_dimensional_probability_array_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            SplitConformalClassifier().fit(np.array([0, 1]), np.array([0.5, 0.5]))

    def test_a_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            SplitConformalClassifier().fit(np.array([0, 0]), np.ones((2, 1)))

    def test_probabilities_outside_the_unit_interval_are_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            SplitConformalClassifier().fit(np.array([0, 1]), np.array([[1.5, -0.5]] * 2))

    def test_non_finite_probabilities_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            SplitConformalClassifier().fit(np.array([0, 1]), np.array([[np.nan, 1.0]] * 2))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            SplitConformalClassifier(alpha=bad)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method must be 'lac' or 'aps'"):
            SplitConformalClassifier(method="magic")
