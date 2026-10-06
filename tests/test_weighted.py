"""Tests for weighted split conformal prediction under covariate shift."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conformal_kit import (
    SplitConformalRegressor,
    WeightedConformalRegressor,
    conformal_quantile,
    effective_sample_size,
    likelihood_ratio_from_probabilities,
    weighted_conformal_quantile,
)
from conformal_kit.cli import main


def _shifted_problem(rng, n_cal=500, n_test=500, shift=1.0):
    """X ~ N(0, 1) at calibration, N(shift, 1) at test; noise grows with |x|.

    The likelihood ratio of N(shift, 1) to N(0, 1) is exp(shift * x - shift^2 / 2).
    """

    def draw(n, mu):
        x = rng.normal(mu, 1.0, n)
        y = x + rng.normal(0.0, 0.3 + np.abs(x), n)
        return x, y

    x_cal, y_cal = draw(n_cal, 0.0)
    x_test, y_test = draw(n_test, shift)

    def ratio(x):
        return np.exp(shift * x - shift**2 / 2.0)

    return x_cal, y_cal, x_test, y_test, ratio


class TestWeightedConformalQuantile:
    def test_equal_weights_reduce_to_split_conformal(self):
        rng = np.random.default_rng(0)
        for n in (9, 50, 333):
            scores = rng.random(n)
            for alpha in (0.1, 0.2):
                expected = conformal_quantile(scores, alpha)
                got = weighted_conformal_quantile(scores, np.ones(n), 1.0, alpha)
                assert got == pytest.approx(expected)

    def test_weight_scale_does_not_matter(self):
        rng = np.random.default_rng(1)
        scores = rng.random(40)
        weights = rng.uniform(0.2, 3.0, 40)
        a = weighted_conformal_quantile(scores, weights, 1.5, 0.1)
        b = weighted_conformal_quantile(scores, 7.0 * weights, 7.0 * 1.5, 0.1)
        assert a == pytest.approx(b)

    def test_hand_computed_weighted_quantile(self):
        # Scores 1..4 with weights 1,1,1,5 and test weight 2 -> total 10.
        # Cumulative mass: 0.1, 0.2, 0.3, 0.8 (+inf: 1.0).
        scores = [1.0, 2.0, 3.0, 4.0]
        weights = [1.0, 1.0, 1.0, 5.0]
        assert weighted_conformal_quantile(scores, weights, 2.0, 0.7) == 3.0
        assert weighted_conformal_quantile(scores, weights, 2.0, 0.2) == 4.0
        # 1 - alpha = 0.9 needs mass past 0.8: only the +inf atom gets there.
        assert weighted_conformal_quantile(scores, weights, 2.0, 0.1) == np.inf

    def test_order_of_scores_is_irrelevant(self):
        scores = np.array([4.0, 1.0, 3.0, 2.0])
        weights = np.array([5.0, 1.0, 1.0, 1.0])
        assert weighted_conformal_quantile(scores, weights, 2.0, 0.2) == 4.0

    def test_vectorized_over_test_weights_and_monotone(self):
        rng = np.random.default_rng(2)
        scores = rng.random(200)
        weights = np.ones(200)
        test_w = np.array([0.0, 1.0, 10.0, 50.0])
        quantiles = weighted_conformal_quantile(scores, weights, test_w, 0.1)
        assert quantiles.shape == (4,)
        # A heavier test point moves more mass to +inf, so the quantile grows.
        assert np.all(np.diff(quantiles) >= 0)
        assert quantiles[-1] == np.inf

    def test_too_few_points_give_an_infinite_quantile(self):
        # Split conformal raises here; the weighted version returns +inf.
        assert weighted_conformal_quantile(np.arange(5.0), np.ones(5), 1.0, 0.05) == np.inf

    def test_zero_weight_calibration_points_are_ignored(self):
        scores = [1.0, 2.0, 100.0]
        assert weighted_conformal_quantile(scores, [1.0, 1.0, 0.0], 0.0, 0.4) == 2.0

    @pytest.mark.parametrize(
        "args, match",
        [
            (([], [], 1.0, 0.1), "at least one calibration score"),
            (([1.0, np.nan], [1, 1], 1.0, 0.1), "finite"),
            (([1.0, 2.0], [1.0], 1.0, 0.1), "same length"),
            (([1.0, 2.0], [1.0, -1.0], 1.0, 0.1), "nonnegative"),
            (([1.0, 2.0], [1.0, 1.0], -1.0, 0.1), "nonnegative"),
            (([1.0, 2.0], [1.0, 1.0], np.inf, 0.1), "finite"),
            (([1.0, 2.0], [0.0, 0.0], 0.0, 0.1), "must not all be zero"),
            (([1.0, 2.0], [1.0, 1.0], 1.0, 1.0), "alpha"),
        ],
    )
    def test_validation(self, args, match):
        with pytest.raises(ValueError, match=match):
            weighted_conformal_quantile(*args)


class TestHelpers:
    def test_effective_sample_size(self):
        assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
        assert effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)
        assert effective_sample_size([3.0, 1.0]) == pytest.approx(16.0 / 10.0)
        with pytest.raises(ValueError, match="all be zero"):
            effective_sample_size([0.0, 0.0])

    def test_likelihood_ratio_from_probabilities(self):
        w = likelihood_ratio_from_probabilities([0.5, 0.75, 0.2], 100, 100)
        assert w == pytest.approx([1.0, 3.0, 0.25])
        # Class-imbalance correction: twice as many calibration rows.
        w = likelihood_ratio_from_probabilities([0.5], 200, 100)
        assert w == pytest.approx([2.0])

    def test_likelihood_ratio_recovers_the_true_ratio(self):
        # Exact Bayes posterior for N(1,1) vs N(0,1) with equal pools.
        x = np.linspace(-2, 3, 11)
        true_ratio = np.exp(x - 0.5)
        posterior = true_ratio / (1.0 + true_ratio)
        w = likelihood_ratio_from_probabilities(posterior, 1000, 1000)
        assert w == pytest.approx(true_ratio)

    def test_likelihood_ratio_clip_and_validation(self):
        w = likelihood_ratio_from_probabilities([0.0, 1.0], 10, 10, clip=0.01)
        assert np.all(np.isfinite(w)) and np.all(w > 0)
        with pytest.raises(ValueError, match="pass clip"):
            likelihood_ratio_from_probabilities([1.0], 10, 10)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            likelihood_ratio_from_probabilities([1.2], 10, 10)
        with pytest.raises(ValueError, match="clip"):
            likelihood_ratio_from_probabilities([0.5], 10, 10, clip=0.6)
        with pytest.raises(ValueError, match="n_test must be positive"):
            likelihood_ratio_from_probabilities([0.5], 10, 0)
        with pytest.raises(ValueError, match="not be empty"):
            likelihood_ratio_from_probabilities([], 10, 10)


class TestWeightedConformalRegressor:
    def test_equal_weights_match_split_conformal(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(0, 1, 300)
        pred = truth + rng.normal(0, 0.5, 300)
        new = rng.normal(0, 1, 20)
        weighted = WeightedConformalRegressor(alpha=0.1).fit(truth, pred, np.ones(300))
        split = SplitConformalRegressor(alpha=0.1).fit(truth, pred)
        lo_w, hi_w = weighted.predict_interval(new, np.ones(20))
        lo_s, hi_s = split.predict_interval(new)
        assert lo_w == pytest.approx(lo_s)
        assert hi_w == pytest.approx(hi_s)
        assert weighted.effective_n_ == pytest.approx(300.0)
        assert weighted.n_calibration_ == 300

    def test_restores_coverage_under_covariate_shift(self):
        rng = np.random.default_rng(4)
        weighted_cov, split_cov = [], []
        for _ in range(60):
            x_cal, y_cal, x_test, y_test, ratio = _shifted_problem(rng)
            model = WeightedConformalRegressor(alpha=0.1).fit(y_cal, x_cal, ratio(x_cal))
            lo, hi = model.predict_interval(x_test, ratio(x_test))
            weighted_cov.append(np.mean((y_test >= lo) & (y_test <= hi)))
            base = SplitConformalRegressor(alpha=0.1).fit(y_cal, x_cal)
            lo, hi = base.predict_interval(x_test)
            split_cov.append(np.mean((y_test >= lo) & (y_test <= hi)))
        # Unweighted split conformal under-covers the shifted test set ...
        assert np.mean(split_cov) < 0.86
        # ... the weighted version is back at the 90% target.
        assert np.mean(weighted_cov) == pytest.approx(0.9, abs=0.02)

    def test_widths_vary_with_the_test_weight(self):
        rng = np.random.default_rng(5)
        x_cal, y_cal, _, _, ratio = _shifted_problem(rng)
        model = WeightedConformalRegressor(alpha=0.1).fit(y_cal, x_cal, ratio(x_cal))
        lo, hi = model.predict_interval([0.0, 0.0], [ratio(-1.0), ratio(2.0)])
        assert (hi - lo)[1] > (hi - lo)[0]
        assert model.effective_n_ < 500

    def test_extreme_extrapolation_returns_an_infinite_interval(self):
        model = WeightedConformalRegressor(alpha=0.1).fit(
            np.arange(50.0), np.arange(50.0) + 0.5, np.ones(50)
        )
        lo, hi = model.predict_interval([1.0, 1.0], [1.0, 1000.0])
        assert np.isfinite(hi[0]) and np.isfinite(lo[0])
        assert hi[1] == np.inf and lo[1] == -np.inf

    def test_normalized_scores_scale_with_difficulty(self):
        rng = np.random.default_rng(6)
        sigma = rng.uniform(0.5, 3.0, 400)
        truth = rng.normal(0.0, sigma)
        pred = np.zeros(400)
        model = WeightedConformalRegressor(alpha=0.1, normalize=True).fit(
            truth, pred, np.ones(400), difficulty=sigma
        )
        lo, hi = model.predict_interval([0.0, 0.0], [1.0, 1.0], difficulty=[1.0, 2.0])
        assert (hi - lo)[1] == pytest.approx(2.0 * (hi - lo)[0])

    def test_quantiles_method(self):
        model = WeightedConformalRegressor(alpha=0.2).fit(
            [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 5.0]
        )
        assert model.quantiles([2.0]) == pytest.approx([4.0])

    def test_errors(self):
        model = WeightedConformalRegressor()
        with pytest.raises(RuntimeError, match="fit must be called"):
            model.predict_interval([0.0], [1.0])
        with pytest.raises(RuntimeError, match="fit must be called"):
            model.quantiles([1.0])
        with pytest.raises(ValueError, match="alpha"):
            WeightedConformalRegressor(alpha=0.0)
        with pytest.raises(ValueError, match="same length"):
            model.fit([1.0, 2.0], [1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="calibration_weights must have the same length"):
            model.fit([1.0, 2.0], [1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="must not all be zero"):
            model.fit([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
        with pytest.raises(ValueError, match="empty"):
            model.fit([], [], [])
        with pytest.raises(ValueError, match="difficulty requires normalize"):
            model.fit([1.0], [1.0], [1.0], difficulty=[1.0])
        model.fit(np.arange(20.0), np.arange(20.0), np.ones(20))
        with pytest.raises(ValueError, match="test_weights must have the same length"):
            model.predict_interval([0.0, 1.0], [1.0])
        with pytest.raises(ValueError, match="finite"):
            model.predict_interval([np.nan], [1.0])
        normalized = WeightedConformalRegressor(normalize=True)
        with pytest.raises(ValueError, match="requires difficulty"):
            normalized.fit([1.0], [1.0], [1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            normalized.fit([1.0], [1.0], [1.0], difficulty=[0.0])


class TestWeightedCli:
    def _write(self, path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return str(path)

    def test_weighted_command_writes_intervals_and_summary(self, tmp_path, capsys):
        rng = np.random.default_rng(7)
        x_cal, y_cal, x_test, _, ratio = _shifted_problem(rng, n_cal=300, n_test=40)
        cal = self._write(
            tmp_path / "cal.csv", ["y_true", "y_pred", "weight"], zip(y_cal, x_cal, ratio(x_cal))
        )
        test = self._write(tmp_path / "test.csv", ["y_pred", "weight"], zip(x_test, ratio(x_test)))
        out = tmp_path / "intervals.csv"
        rc = main(["weighted", "--calibration", cal, "--test", test, "--out", str(out), "--json"])
        assert rc == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_calibration"] == 300
        assert summary["n_test"] == 40
        assert 0 < summary["effective_n_calibration"] < 300
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 40
        assert set(rows[0]) == {"y_pred", "weight", "lower", "upper"}

    def test_custom_weight_column_and_text_output(self, tmp_path, capsys):
        cal = self._write(
            tmp_path / "cal.csv", ["y_true", "y_pred", "lr"], [[i, i + 0.5, 1.0] for i in range(30)]
        )
        test = self._write(tmp_path / "test.csv", ["y_pred", "lr"], [[1.0, 1.0], [2.0, 500.0]])
        rc = main(
            ["weighted", "--calibration", cal, "--test", test, "--weight-column", "lr"]
        )
        assert rc == 0
        text = capsys.readouterr().out
        assert "effective n = 30.0" in text
        assert "Infinite intervals  : 1 of 2" in text

    def test_missing_weight_column_is_an_error(self, tmp_path, capsys):
        cal = self._write(tmp_path / "cal.csv", ["y_true", "y_pred"], [[1, 1]])
        test = self._write(tmp_path / "test.csv", ["y_pred"], [[1]])
        assert main(["weighted", "--calibration", cal, "--test", test]) == 2
        assert "missing column(s): weight" in capsys.readouterr().err
