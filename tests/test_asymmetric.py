import numpy as np
import pytest

from conformal_kit import AsymmetricSplitConformalRegressor, SplitConformalRegressor
from conformal_kit.asymmetric import asymmetric_interval


def test_coverage_on_skewed_residuals():
    rng = np.random.default_rng(0)
    alpha = 0.1
    # Skewed residuals: Exp(1) - 1, mean 0, long right tail.
    n_cal, n_test = 800, 5000
    pred_cal = rng.normal(size=n_cal)
    y_cal = pred_cal + rng.exponential(size=n_cal) - 1.0
    pred_test = rng.normal(size=n_test)
    y_test = pred_test + rng.exponential(size=n_test) - 1.0

    asym = AsymmetricSplitConformalRegressor(alpha=alpha).fit(y_cal, pred_cal)
    lo, hi = asym.predict_interval(pred_test)
    cov = float(np.mean((y_test >= lo) & (y_test <= hi)))
    assert cov >= 1 - alpha - 0.02

    # Asymmetric should put more width above the prediction than below.
    assert asym.upper_quantile_ > -asym.lower_quantile_


def test_symmetric_when_residuals_symmetric_matches_split():
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    p = np.zeros(500)
    asym = AsymmetricSplitConformalRegressor(alpha=0.1).fit(y, p)
    sym = SplitConformalRegressor(alpha=0.1).fit(y, p)
    # For symmetric noise the two half-widths are close to the absolute quantile.
    assert abs(asym.upper_quantile_ - (-asym.lower_quantile_)) < 0.15
    assert abs(asym.upper_quantile_ - sym.quantile_) < 0.15


def test_unequal_tails():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    p = np.zeros(400)
    m = AsymmetricSplitConformalRegressor(alpha_lo=0.05, alpha_hi=0.15).fit(y, p)
    assert m.alpha == pytest.approx(0.2)
    lo, hi = m.predict_interval(np.zeros(50))
    assert np.all(lo < 0) and np.all(hi > 0)
    # Heavier upper budget -> smaller upper quantile magnitude than lower (more room above).
    # alpha_hi=0.15 means 85% quantile; alpha_lo=0.05 means 5% quantile of residuals.
    assert m.upper_quantile_ < -m.lower_quantile_


def test_normalize_scales_widths():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=300)
    difficulty = rng.uniform(0.5, 2.0, size=300)
    y = pred + difficulty * rng.normal(size=300)
    m = AsymmetricSplitConformalRegressor(alpha=0.1, normalize=True).fit(y, pred, difficulty)
    d_test = np.array([1.0, 2.0, 0.5])
    lo, hi = m.predict_interval(np.zeros(3), difficulty=d_test)
    widths = hi - lo
    assert widths[1] == pytest.approx(2 * widths[0])
    assert widths[2] == pytest.approx(0.5 * widths[0])


def test_asymmetric_interval_helper():
    lo, hi = asymmetric_interval([0.0, 10.0], -2.0, 3.0)
    np.testing.assert_allclose(lo, [-2.0, 8.0])
    np.testing.assert_allclose(hi, [3.0, 13.0])


def test_rejects_bad_alphas_and_unfitted():
    with pytest.raises(ValueError):
        AsymmetricSplitConformalRegressor(alpha=0.0)
    with pytest.raises(ValueError):
        AsymmetricSplitConformalRegressor(alpha_lo=0.1, alpha_hi=None)
    with pytest.raises(ValueError):
        AsymmetricSplitConformalRegressor(alpha_lo=0.6, alpha_hi=0.5)
    m = AsymmetricSplitConformalRegressor(0.1)
    with pytest.raises(RuntimeError):
        m.predict_interval([0.0])
