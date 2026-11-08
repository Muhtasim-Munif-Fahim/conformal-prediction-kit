import numpy as np
import pytest
from scipy import stats

from conformal_kit import (
    DistributionalConformalRegressor,
    quantile_grid_cdf,
    quantile_grid_inverse,
    rearrange_quantiles,
)

LEVELS = np.linspace(0.01, 0.99, 99)


def _skewed(n, seed, misspecify=1.0):
    """Heteroskedastic, right-skewed data with a known (optionally mis-scaled) quantile model."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 3.0, n)
    y = x + x * rng.gamma(2.0, 1.0, n)
    Q = x[:, None] + misspecify * x[:, None] * stats.gamma.ppf(LEVELS, 2.0)[None, :]
    return x, y, Q


def test_cdf_and_inverse_round_trip():
    Q = np.array([[0.0, 1.0, 3.0], [10.0, 10.0, 12.0]])
    taus = [0.1, 0.5, 0.9]
    F = quantile_grid_cdf([0.5, 11.0], Q, taus)
    np.testing.assert_allclose(F, [0.3, 0.7])
    assert quantile_grid_cdf([-1.0], Q[:1], taus)[0] == 0.0
    assert quantile_grid_cdf([3.0], Q[:1], taus)[0] == 1.0
    np.testing.assert_allclose(quantile_grid_inverse(0.3, Q, taus, "lower"), [0.5, 10.0])
    np.testing.assert_allclose(quantile_grid_inverse(0.7, Q, taus, "upper"), [2.0, 11.0])
    assert np.isneginf(quantile_grid_inverse(0.0, Q, taus, "lower")).all()
    assert np.isposinf(quantile_grid_inverse(1.0, Q, taus, "upper")).all()
    np.testing.assert_allclose(quantile_grid_inverse(0.05, Q, taus, "lower"), [0.0, 10.0])


def test_rearrangement_fixes_crossing():
    Q = rearrange_quantiles([[3.0, 1.0, 2.0]])
    np.testing.assert_array_equal(Q, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("score", ["dcp", "dcp-qr"])
@pytest.mark.parametrize("misspecify", [0.6, 1.0, 1.5])
def test_marginal_coverage_even_when_misspecified(score, misspecify):
    covs = []
    for seed in range(20):
        _, y_cal, Q_cal = _skewed(300, seed, misspecify)
        _, y_te, Q_te = _skewed(1000, 100 + seed, misspecify)
        model = DistributionalConformalRegressor(LEVELS, alpha=0.1, score=score).fit(y_cal, Q_cal)
        lo, hi = model.predict_interval(Q_te)
        covs.append(np.mean((y_te >= lo) & (y_te <= hi)))
    assert np.mean(covs) >= 0.89
    assert np.mean(covs) <= 0.94


def test_interval_matches_score_membership():
    _, y_cal, Q_cal = _skewed(200, 1)
    _, y_te, Q_te = _skewed(500, 2)
    model = DistributionalConformalRegressor(LEVELS, alpha=0.2).fit(y_cal, Q_cal)
    lo, hi = model.predict_interval(Q_te)
    np.testing.assert_array_equal((y_te >= lo) & (y_te <= hi), model.contains(y_te, Q_te))


def test_intervals_adapt_to_scale_and_skew():
    _, y_cal, Q_cal = _skewed(500, 3)
    model = DistributionalConformalRegressor(LEVELS, alpha=0.1).fit(y_cal, Q_cal)
    x_te = np.array([0.6, 2.8])
    Q_te = x_te[:, None] + x_te[:, None] * stats.gamma.ppf(LEVELS, 2.0)[None, :]
    lo, hi = model.predict_interval(Q_te)
    width = hi - lo
    assert width[1] > 3 * width[0]
    median = np.median(Q_te, axis=1)
    # right skew: more room above the median than below it
    assert np.all(hi - median > median - lo)


def test_p_values_and_validation():
    _, y_cal, Q_cal = _skewed(99, 4)
    model = DistributionalConformalRegressor(LEVELS, alpha=0.1)
    with pytest.raises(RuntimeError):
        model.predict_interval(Q_cal)
    model.fit(y_cal, Q_cal)
    p = model.p_values(y_cal[:5], Q_cal[:5])
    assert np.all((p > 0) & (p <= 1))
    with pytest.raises(ValueError):
        DistributionalConformalRegressor([0.5])
    with pytest.raises(ValueError):
        DistributionalConformalRegressor([0.2, 0.1])
    with pytest.raises(ValueError):
        DistributionalConformalRegressor(LEVELS, score="nope")
    with pytest.raises(ValueError):
        model.fit(y_cal, Q_cal[:, :5])


def test_coarse_grid_reports_outside_fraction():
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    taus = np.array([0.3, 0.5, 0.7])
    Q = np.tile(stats.norm.ppf(taus), (400, 1))
    model = DistributionalConformalRegressor(taus, alpha=0.1).fit(y, Q)
    assert model.outside_grid_fraction_ == pytest.approx(0.6, abs=0.08)
    lo, hi = model.predict_interval(Q[:3])
    assert np.isinf(lo).all() and np.isinf(hi).all()
