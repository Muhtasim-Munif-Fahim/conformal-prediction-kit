"""Tests for Mondrian (group-conditional) split conformal regression."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.evaluation import interval_coverage_report
from conformal_kit.regression import (
    MondrianSplitConformalRegressor,
    SplitConformalRegressor,
    conformal_quantile,
    mondrian_regression_quantiles,
)


class TestMondrianRegressionQuantiles:
    def test_each_group_uses_its_own_quantile(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2)
        groups = np.array(["easy"] * 9 + ["hard"] * 9)
        scores[9:] = np.linspace(1.0, 5.0, 9)
        quantiles, counts = mondrian_regression_quantiles(scores, groups, alpha=0.1)
        assert counts == {"easy": 9, "hard": 9}
        assert quantiles["easy"] == pytest.approx(0.9)
        assert quantiles["hard"] == pytest.approx(5.0)

    def test_matches_per_group_conformal_quantile(self):
        rng = np.random.default_rng(3)
        scores = rng.uniform(0.0, 2.0, 300)
        groups = rng.integers(0, 3, 300)
        quantiles, _ = mondrian_regression_quantiles(scores, groups, alpha=0.1)
        for g in range(3):
            assert quantiles[g] == pytest.approx(conformal_quantile(scores[groups == g], 0.1))

    def test_too_few_examples_names_the_group(self):
        scores = np.concatenate([np.linspace(0.1, 0.9, 9), [0.5]])
        groups = np.array([0] * 9 + [1])
        with pytest.raises(ValueError, match="group 1 needs at least 9"):
            mondrian_regression_quantiles(scores, groups, alpha=0.1)


class TestMondrianSplitConformalRegressor:
    def test_intervals_use_group_widths(self):
        # Easy group residuals ~0.1, hard ~2.0; enough points for alpha=0.1.
        y_true = np.concatenate([np.zeros(20), np.zeros(20)])
        y_pred = np.concatenate([np.full(20, 0.1), np.full(20, 2.0)])
        groups = np.array(["easy"] * 20 + ["hard"] * 20)
        model = MondrianSplitConformalRegressor(alpha=0.1).fit(y_true, y_pred, groups)
        lower, upper = model.predict_interval([0.0, 0.0], groups=["easy", "hard"])
        easy_width = upper[0] - lower[0]
        hard_width = upper[1] - lower[1]
        assert hard_width > easy_width * 5

    def test_unknown_group_raises(self):
        y_true = np.arange(20, dtype=float)
        y_pred = y_true + 0.1
        groups = np.array([0] * 20)
        model = MondrianSplitConformalRegressor(alpha=0.1).fit(y_true, y_pred, groups)
        with pytest.raises(KeyError, match="unknown group"):
            model.predict_interval([0.0], groups=[1])

    def test_hard_group_is_not_undercovered_vs_pooled(self):
        rng = np.random.default_rng(0)
        n = 2000
        groups = np.array(["easy"] * (n // 2) + ["hard"] * (n // 2))
        truth = rng.normal(0.0, 1.0, n)
        noise = np.where(groups == "easy", 0.2, 2.0)
        pred = truth + rng.normal(0.0, 1.0, n) * noise
        cal, test = slice(None, n // 2), slice(n // 2, None)
        # Rebalance so each half still has both groups.
        order = np.arange(n)
        rng.shuffle(order)
        truth, pred, groups = truth[order], pred[order], groups[order]

        mondrian = MondrianSplitConformalRegressor(alpha=0.1).fit(
            truth[cal], pred[cal], groups[cal]
        )
        pooled = SplitConformalRegressor(alpha=0.1).fit(truth[cal], pred[cal])

        m_lo, m_hi = mondrian.predict_interval(pred[test], groups[test])
        p_lo, p_hi = pooled.predict_interval(pred[test])

        hard = groups[test] == "hard"
        m_cov = ((truth[test][hard] >= m_lo[hard]) & (truth[test][hard] <= m_hi[hard])).mean()
        p_cov = ((truth[test][hard] >= p_lo[hard]) & (truth[test][hard] <= p_hi[hard])).mean()
        # Mondrian should cover the hard group near 90%; pooled often under-covers.
        assert m_cov >= 0.85
        assert m_cov >= p_cov - 0.02

    def test_normalize_scales_widths(self):
        y_true = np.zeros(30)
        y_pred = np.full(30, 1.0)
        groups = np.array([0] * 30)
        difficulty = np.full(30, 2.0)
        model = MondrianSplitConformalRegressor(alpha=0.1, normalize=True).fit(
            y_true, y_pred, groups, difficulty=difficulty
        )
        lower, upper = model.predict_interval([0.0], groups=[0], difficulty=[2.0])
        assert upper[0] - lower[0] == pytest.approx(2.0 * model.quantiles_[0] * 2.0, rel=0, abs=1e-12)
