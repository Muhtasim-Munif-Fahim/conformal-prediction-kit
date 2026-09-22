"""Tests for cross-conformal / CV+ classification prediction sets."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.classification import SplitConformalClassifier
from conformal_kit.cross_conformal import (
    CrossConformalClassifier,
    cross_conformal_p_values,
    cross_conformal_sets,
)
from conformal_kit.regression import conformal_quantile


class ClassMeanClassifier:
    """Nearest class-mean probabilities. ``n_classes`` keeps fold widths aligned."""

    def __init__(self, n_classes):
        self.n_classes = int(n_classes)

    def fit(self, X, y):
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y).ravel()
        self.means_ = np.zeros((self.n_classes, features.shape[1]))
        self.counts_ = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            selected = labels == cls
            self.counts_[cls] = selected.sum()
            if np.any(selected):
                self.means_[cls] = features[selected].mean(axis=0)
        return self

    def predict_proba(self, X):
        features = np.asarray(X, dtype=float)
        diff = features[:, None, :] - self.means_[None, :, :]
        logits = -np.sum(diff * diff, axis=2)
        logits[:, self.counts_ == 0] = -1e6
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)


def class_mean_trainer(n_classes):
    def trainer(X_train, y_train):
        return ClassMeanClassifier(n_classes).fit(X_train, y_train).predict_proba

    return trainer


def _blobs(n, k=3, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    X = np.eye(k)[labels] * sep + rng.normal(scale=1.0, size=(n, k))
    return X, labels


class TestCrossConformalPValues:
    def test_hand_computed_counts(self):
        # Two folds, four calibration scores, one test point, two classes.
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        folds = np.array([0, 0, 1, 1])
        test = np.zeros((2, 1, 2))
        test[0, 0, :] = (0.3, 0.7)
        test[1, 0, :] = (0.5, 0.1)
        p = cross_conformal_p_values(scores, folds, test)
        # Class 0: fold 0 contributes 1, fold 1 contributes 2 -> 4/5.
        # Class 1: fold 0 contributes 0, fold 1 contributes 2 -> 3/5.
        assert p.shape == (1, 2)
        assert p[0, 0] == pytest.approx(4 / 5)
        assert p[0, 1] == pytest.approx(3 / 5)

    def test_a_tie_counts_toward_inclusion(self):
        scores = np.array([0.5, 0.5])
        folds = np.array([0, 1])
        test = np.full((2, 1, 1), 0.5)
        p = cross_conformal_p_values(scores, folds, test)
        assert p[0, 0] == pytest.approx(1.0)

    def test_a_score_worse_than_every_calibration_point_has_the_minimum_p(self):
        scores = np.array([0.1, 0.2, 0.3])
        folds = np.array([0, 1, 2])
        test = np.full((3, 2, 1), 5.0)
        p = cross_conformal_p_values(scores, folds, test)
        assert p == pytest.approx(np.full((2, 1), 1 / 4))

    def test_constant_scores_match_the_split_conformal_quantile(self):
        # One fold, so every comparison uses the same score. p > alpha must
        # agree with score <= conformal_quantile.
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 1.0, 40)
        candidates = np.array([0.0, 0.2, 0.5, 0.9, 1.0, scores.max(), scores.min()])
        folds = np.zeros(40, dtype=int)
        test = np.zeros((1, candidates.size, 1))
        test[0, :, 0] = candidates
        for alpha in (0.1, 0.2, 0.5):
            included = cross_conformal_p_values(scores, folds, test)[:, 0] > alpha
            threshold = conformal_quantile(scores, alpha)
            assert np.array_equal(included, candidates <= threshold)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            cross_conformal_p_values([0.1, 0.2], [0], np.zeros((1, 1, 1)))

    def test_a_missing_fold_is_rejected(self):
        with pytest.raises(ValueError, match="fold 1 has no calibration"):
            cross_conformal_p_values([0.1, 0.2], [0, 2], np.zeros((3, 1, 1)))

    def test_non_finite_scores_are_rejected(self):
        with pytest.raises(ValueError, match="must all be finite"):
            cross_conformal_p_values([0.1, np.inf], [0, 1], np.zeros((2, 1, 1)))

    def test_too_few_points_for_the_level_is_rejected(self):
        scores = np.arange(5.0)
        folds = np.zeros(5, dtype=int)
        test = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="needs at least 19 calibration points"):
            cross_conformal_sets(scores, folds, test, 0.05)


class TestCrossConformalSets:
    def test_p_values_above_alpha_are_included(self):
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        folds = np.array([0, 0, 1, 1])
        test = np.zeros((2, 1, 2))
        test[0, 0, :] = (0.3, 0.7)
        test[1, 0, :] = (0.5, 0.1)
        # p = [0.8, 0.6]. n=4 supports alpha=0.2 (needs 4 points).
        mask = cross_conformal_sets(scores, folds, test, 0.2)
        assert mask.tolist() == [[True, True]]
        mask = cross_conformal_sets(scores, folds, test, 0.7)
        assert mask.tolist() == [[True, False]]

    def test_an_empty_row_keeps_the_preferred_class(self):
        # Every test score is worse than every calibration score, so both
        # p-values equal 1/5 = 0.2, which does not clear alpha=0.2.
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        folds = np.arange(4)
        test = np.full((4, 1, 2), 9.0)
        preference = np.array([[0.1, 0.9]])
        mask = cross_conformal_sets(scores, folds, test, 0.2, preference=preference)
        assert mask.tolist() == [[False, True]]

    def test_without_a_preference_the_highest_p_value_is_kept(self):
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        folds = np.array([0, 0, 1, 1])
        test = np.zeros((2, 1, 2))
        test[0, 0, :] = (0.3, 0.7)
        test[1, 0, :] = (0.5, 0.1)
        # p = [0.8, 0.6], alpha=0.9 excludes both, so class 0 is restored.
        mask = cross_conformal_sets(scores, folds, test, 0.9)
        assert mask.tolist() == [[True, False]]

    def test_a_preference_with_the_wrong_shape_is_rejected(self):
        scores = np.ones(10)
        folds = np.zeros(10, dtype=int)
        test = np.zeros((1, 2, 2))
        with pytest.raises(ValueError, match="preference must have shape"):
            cross_conformal_sets(scores, folds, test, 0.1, preference=np.ones((2, 3)))


class TestCrossConformalClassifier:
    def test_fit_returns_self_for_chaining(self):
        X, y = _blobs(30, seed=1)
        model = CrossConformalClassifier(alpha=0.1, n_splits=5)
        assert model.fit(X, y, class_mean_trainer(3)) is model
        assert model.n_calibration_ == 30
        assert model.n_splits_ == 5
        assert model.n_classes_ == 3

    def test_leave_one_out_scores_never_see_the_held_out_point(self):
        X, y = _blobs(18, k=3, sep=1.0, seed=2)
        model = CrossConformalClassifier(alpha=0.1, method="lac").fit(
            X, y, ClassMeanClassifier(3)
        )
        assert model.n_splits_ == 18
        for index in range(len(y)):
            held = np.ones(len(y), dtype=bool)
            held[index] = False
            proba = ClassMeanClassifier(3).fit(X[held], y[held]).predict_proba(X[index : index + 1])
            expected = 1.0 - proba[0, y[index]]
            assert model.calibration_scores_[index] == pytest.approx(expected)

    @pytest.mark.parametrize("method", ["lac", "aps"])
    @pytest.mark.parametrize("n_splits", [None, 5])
    def test_a_constant_model_matches_split_conformal(self, method, n_splits):
        rng = np.random.default_rng(0)
        # Dyadic probabilities so 1 - (1 - p) lands back on p. Split LAC
        # rebuilds its threshold as 1 - quantile, and a non-dyadic p can
        # fall on the wrong side of that reconstruction.
        proba = np.array([0.5, 0.25, 0.125, 0.125])
        y = rng.integers(0, 4, 40)
        X = rng.normal(size=(40, 2))
        X_test = rng.normal(size=(12, 2))

        def trainer(X_train, y_train):
            def predict_proba(Xnew):
                return np.tile(proba, (len(np.asarray(Xnew)), 1))

            return predict_proba

        cross = CrossConformalClassifier(alpha=0.1, method=method, n_splits=n_splits)
        cross.fit(X, y, trainer)
        split = SplitConformalClassifier(alpha=0.1, method=method).fit(y, np.tile(proba, (40, 1)))
        assert np.array_equal(
            cross.predict_set(X_test),
            split.predict_set(np.tile(proba, (12, 1))),
        )

    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_empirical_coverage_meets_the_cross_conformal_bound(self, method):
        # Theorem: at least 1 - 2*alpha = 80% at alpha=0.1.
        X_train, y_train = _blobs(180, seed=4)
        X_test, y_test = _blobs(400, seed=5)
        model = CrossConformalClassifier(alpha=0.1, method=method, n_splits=6, random_state=0)
        model.fit(X_train, y_train, ClassMeanClassifier(3))
        mask = model.predict_set(X_test)
        covered = mask[np.arange(len(y_test)), y_test].mean()
        assert covered >= 0.80

    def test_empirical_coverage_is_close_to_the_nominal_level(self):
        X_train, y_train = _blobs(240, sep=2.5, seed=6)
        X_test, y_test = _blobs(500, sep=2.5, seed=7)
        model = CrossConformalClassifier(alpha=0.1, method="lac", n_splits=8, random_state=1)
        model.fit(X_train, y_train, ClassMeanClassifier(3))
        mask = model.predict_set(X_test)
        covered = mask[np.arange(len(y_test)), y_test].mean()
        assert covered >= 0.85

    def test_leave_one_out_coverage_meets_the_bound(self):
        X_train, y_train = _blobs(40, sep=3.0, seed=8)
        X_test, y_test = _blobs(200, sep=3.0, seed=9)
        model = CrossConformalClassifier(alpha=0.1, method="aps").fit(
            X_train, y_train, ClassMeanClassifier(3)
        )
        mask = model.predict_set(X_test)
        covered = mask[np.arange(len(y_test)), y_test].mean()
        assert covered >= 0.80
        assert model.n_splits_ == 40

    def test_n_splits_equal_to_n_matches_leave_one_out(self):
        X, y = _blobs(16, seed=10)
        X_test, _ = _blobs(5, seed=11)
        loo = CrossConformalClassifier(alpha=0.2, method="aps").fit(X, y, ClassMeanClassifier(3))
        cvn = CrossConformalClassifier(alpha=0.2, method="aps", n_splits=len(y)).fit(
            X, y, ClassMeanClassifier(3)
        )
        assert np.array_equal(loo.predict_set(X_test), cvn.predict_set(X_test))
        assert loo.predict_p_values(X_test) == pytest.approx(cvn.predict_p_values(X_test))

    def test_a_tighter_alpha_gives_larger_sets(self):
        X, y = _blobs(80, seed=12)
        X_test, _ = _blobs(40, seed=13)
        loose = CrossConformalClassifier(alpha=0.2, n_splits=5).fit(X, y, ClassMeanClassifier(3))
        tight = CrossConformalClassifier(alpha=0.05, n_splits=5).fit(X, y, ClassMeanClassifier(3))
        assert tight.set_sizes(X_test).mean() > loose.set_sizes(X_test).mean()

    def test_a_callable_trainer_and_an_estimator_agree(self):
        X, y = _blobs(24, seed=14)
        X_test, _ = _blobs(6, seed=15)
        from_callable = CrossConformalClassifier(alpha=0.2, n_splits=4).fit(
            X, y, class_mean_trainer(3)
        )
        from_estimator = CrossConformalClassifier(alpha=0.2, n_splits=4).fit(
            X, y, ClassMeanClassifier(3)
        )
        assert np.array_equal(from_callable.predict_set(X_test), from_estimator.predict_set(X_test))

    def test_sets_are_never_empty(self):
        X, y = _blobs(40, seed=16)
        X_test, _ = _blobs(15, seed=17)
        for method in ("lac", "aps"):
            model = CrossConformalClassifier(alpha=0.5, method=method, n_splits=4)
            model.fit(X, y, ClassMeanClassifier(3))
            assert model.set_sizes(X_test).min() >= 1

    def test_predict_labels_lists_the_included_indices(self):
        X, y = _blobs(30, seed=18)
        model = CrossConformalClassifier(alpha=0.2, n_splits=5).fit(X, y, ClassMeanClassifier(3))
        mask = model.predict_set(X[:4])
        listed = model.predict_labels(X[:4])
        for row, indices in zip(mask, listed):
            assert np.flatnonzero(row).tolist() == indices

    def test_set_sizes_matches_the_mask(self):
        X, y = _blobs(30, seed=19)
        model = CrossConformalClassifier(alpha=0.2, n_splits=5).fit(X, y, ClassMeanClassifier(3))
        assert np.array_equal(model.set_sizes(X), model.predict_set(X).sum(axis=1))

    def test_one_dimensional_features_are_accepted(self):
        rng = np.random.default_rng(20)
        labels = rng.integers(0, 2, 40)
        X = labels.astype(float) + rng.normal(scale=0.4, size=40)
        model = CrossConformalClassifier(alpha=0.2, method="lac", n_splits=4)
        model.fit(X, labels, ClassMeanClassifier(2))
        mask = model.predict_set(np.array([-0.2, 1.3, 0.4]))
        assert mask.shape == (3, 2)
        assert mask.sum(axis=1).min() >= 1

    def test_p_values_lie_in_the_unit_interval(self):
        X, y = _blobs(30, seed=21)
        model = CrossConformalClassifier(alpha=0.2, n_splits=5).fit(X, y, ClassMeanClassifier(3))
        p = model.predict_p_values(X[:8])
        assert p.shape == (8, 3)
        assert np.all(p > 0.0) and np.all(p <= 1.0)

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            CrossConformalClassifier().predict_set([[0.0, 1.0]])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            CrossConformalClassifier().fit([[0.0], [1.0]], [0], class_mean_trainer(2))

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            CrossConformalClassifier().fit([[0.0], [1.0]], [0.0, 1.0], class_mean_trainer(2))

    def test_negative_labels_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            CrossConformalClassifier().fit(
                [[0.0], [1.0], [2.0]], [-1, 0, 1], class_mean_trainer(2)
            )

    def test_a_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            CrossConformalClassifier().fit(
                np.zeros((8, 1)), np.zeros(8, dtype=int), class_mean_trainer(1)
            )

    def test_feature_mismatch_at_predict_is_rejected(self):
        X, y = _blobs(20, seed=22)
        model = CrossConformalClassifier(alpha=0.2, n_splits=4).fit(X, y, ClassMeanClassifier(3))
        with pytest.raises(ValueError, match="expected 3 features"):
            model.predict_set(np.zeros((2, 2)))

    def test_too_few_training_points_are_rejected(self):
        with pytest.raises(ValueError, match="needs at least 19"):
            CrossConformalClassifier(alpha=0.05).fit(
                np.zeros((5, 1)), np.array([0, 1, 0, 1, 0]), class_mean_trainer(2)
            )

    def test_n_splits_cannot_exceed_n(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            CrossConformalClassifier(n_splits=12).fit(
                np.zeros((10, 1)), np.array([0, 1] * 5), class_mean_trainer(2)
            )

    def test_a_fold_that_drops_a_class_column_is_rejected(self):
        def trainer(X_train, y_train):
            seen = np.unique(y_train)

            def predict_proba(X):
                return np.full((len(np.asarray(X)), seen.size), 1.0 / seen.size)

            return predict_proba

        # Leave-one-out on the only example of class 1 trains a model that
        # has never seen that class. The column still has to be there.
        X = np.zeros((4, 1))
        y = np.array([0, 0, 0, 1])
        with pytest.raises(ValueError, match="predict_proba must return shape"):
            CrossConformalClassifier(alpha=0.5).fit(X, y, trainer)

    def test_probabilities_outside_the_unit_interval_are_rejected(self):
        def trainer(X_train, y_train):
            def predict_proba(X):
                return np.tile([1.5, -0.5], (len(np.asarray(X)), 1))

            return predict_proba

        with pytest.raises(ValueError, match="between 0 and 1"):
            CrossConformalClassifier(alpha=0.2, n_splits=2).fit(
                np.zeros((8, 1)), np.array([0, 1] * 4), trainer
            )

    def test_a_callable_that_does_not_return_a_predictor_is_rejected(self):
        def bad(X, y):
            return np.mean(y)

        with pytest.raises(TypeError, match="must return a predictor"):
            CrossConformalClassifier().fit(np.zeros((20, 1)), np.array([0, 1] * 10), bad)

    def test_a_non_model_is_rejected(self):
        with pytest.raises(ValueError, match="fit and predict_proba"):
            CrossConformalClassifier().fit(np.zeros((20, 1)), np.array([0, 1] * 10), object())

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_alpha_is_validated_at_construction(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            CrossConformalClassifier(alpha=bad)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method must be 'lac' or 'aps'"):
            CrossConformalClassifier(method="raps")

    @pytest.mark.parametrize("bad", [1, True, 2.5, "10"])
    def test_n_splits_must_be_an_integer_at_least_two(self, bad):
        with pytest.raises(ValueError, match="n_splits"):
            CrossConformalClassifier(n_splits=bad)
