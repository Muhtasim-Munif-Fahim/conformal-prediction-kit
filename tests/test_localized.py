import numpy as np
import pytest

from conformal_kit import LocalizedConformalRegressor, SplitConformalRegressor, localizer_weights


def _hetero(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    noise = np.where(x < 0, 0.1, 1.0)  # low noise on the left, high on the right
    y = np.sin(2 * x) + noise * rng.normal(size=n)
    return x, y, np.sin(2 * x)


def test_localizer_weights_shape_and_values():
    K = localizer_weights([[0.0], [1.0]], [[0.0], [2.0], [1.0]], bandwidth=1.0)
    assert K.shape == (2, 3)
    assert K[0, 0] == pytest.approx(1.0) and K[1, 1] == pytest.approx(np.exp(-0.5))
    with pytest.raises(ValueError, match="bandwidth"):
        localizer_weights([[0.0]], [[0.0]], bandwidth=0.0)
    with pytest.raises(ValueError, match="same number of features"):
        localizer_weights([[0.0, 1.0]], [[0.0]], bandwidth=1.0)


def test_adapts_width_and_keeps_marginal_and_local_coverage():
    x_cal, y_cal, f_cal = _hetero(1500, 0)
    x_te, y_te, f_te = _hetero(4000, 1)
    lcp = LocalizedConformalRegressor(alpha=0.1, bandwidth=0.2).fit(x_cal, y_cal, f_cal)
    lo, hi = lcp.predict_interval(x_te, f_te)
    covered = (y_te >= lo) & (y_te <= hi)
    assert covered.mean() >= 0.87
    width = hi - lo
    left, right = x_te < -0.2, x_te > 0.2
    assert width[left].mean() < 0.35 * width[right].mean()
    # each region is covered (split conformal over-covers left, under-covers right)
    assert covered[left].mean() >= 0.85 and covered[right].mean() >= 0.85
    split = SplitConformalRegressor(alpha=0.1).fit(y_cal, f_cal)
    s_lo, s_hi = split.predict_interval(f_te)
    s_cov = (y_te >= s_lo) & (y_te <= s_hi)
    assert s_cov[right].mean() < covered[right].mean()
    assert np.median(width[left]) < np.median(s_hi - s_lo)
    assert 0.0 < lcp.level_ < 1.0 and lcp.loo_levels_.shape == (1500,)


def test_huge_bandwidth_reduces_to_split_conformal():
    x_cal, y_cal, f_cal = _hetero(300, 2)
    lcp = LocalizedConformalRegressor(alpha=0.1, bandwidth=1e6, calibrate=False).fit(x_cal, y_cal, f_cal)
    split = SplitConformalRegressor(alpha=0.1).fit(y_cal, f_cal)
    q = lcp.quantiles(np.array([-0.5, 0.0, 0.7]))
    s_lo, s_hi = split.predict_interval(np.zeros(1))
    np.testing.assert_allclose(q, (s_hi[0] - s_lo[0]) / 2.0)


def test_effective_sample_size_and_far_points():
    x_cal, y_cal, f_cal = _hetero(200, 3)
    lcp = LocalizedConformalRegressor(alpha=0.1, bandwidth=0.05).fit(x_cal, y_cal, f_cal)
    ess = lcp.effective_sample_sizes(np.array([0.0, 50.0]))
    assert ess[0] > 5
    q = lcp.quantiles(np.array([50.0]))  # no calibration mass nearby -> infinite interval
    assert np.isinf(q[0])


def test_multifeature_and_validation():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(size=200)
    lcp = LocalizedConformalRegressor(alpha=0.2).fit(X, y, X[:, 0])
    assert lcp.bandwidth_ == pytest.approx(200 ** (-1 / 7))
    lo, hi = lcp.predict_interval(X[:5], X[:5, 0])
    assert np.all(hi >= lo)
    with pytest.raises(ValueError, match="same number of features"):
        lcp.quantiles(X[:, :2])
    with pytest.raises(ValueError, match="alpha"):
        LocalizedConformalRegressor(alpha=1.5)
    with pytest.raises(ValueError, match="same number of rows"):
        LocalizedConformalRegressor().fit(X, y[:10], X[:, 0])
    with pytest.raises(RuntimeError, match="fit"):
        LocalizedConformalRegressor().quantiles(X)
    tiny = LocalizedConformalRegressor(alpha=0.01).fit(X[:5], y[:5], X[:5, 0])
    assert tiny.level_ == 1.0 and np.all(np.isinf(tiny.quantiles(X[:3])))
