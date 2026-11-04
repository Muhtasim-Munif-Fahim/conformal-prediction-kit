import numpy as np
import pytest

from conformal_kit import MultiOutputConformalRegressor, SplitConformalRegressor, joint_coverage


def _data(rng, n, rho=0.9, scales=(1.0, 3.0)):
    d = len(scales)
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    cov = cov * np.outer(scales, scales)
    pred = rng.normal(size=(n, d))
    y = pred + rng.multivariate_normal(np.zeros(d), cov, size=n)
    return y, pred


@pytest.mark.parametrize("method", ["bonferroni", "max", "mahalanobis"])
def test_joint_coverage_guarantee(method):
    rng = np.random.default_rng(0)
    alpha = 0.1
    covs = []
    for t in range(200):
        y, p = _data(rng, 300)
        yt, pt = _data(rng, 200)
        m = MultiOutputConformalRegressor(alpha, method=method, seed=t).fit(y, p)
        covs.append(m.contains(yt, pt).mean())
    mean = float(np.mean(covs))
    assert mean >= 1 - alpha - 0.01
    if method != "bonferroni":
        assert mean <= 1 - alpha + 0.02  # exact, not conservative


def test_independent_marginal_intervals_undercover_jointly():
    rng = np.random.default_rng(1)
    y, p = _data(rng, 2000, rho=0.0, scales=(1, 1, 1, 1))
    yt, pt = _data(rng, 20000, rho=0.0, scales=(1, 1, 1, 1))
    lo, hi = [], []
    for j in range(4):
        r = SplitConformalRegressor(0.1).fit(y[:, j], p[:, j])
        a, b = r.predict_interval(pt[:, j])
        lo.append(a)
        hi.append(b)
    naive = joint_coverage(yt, np.column_stack(lo), np.column_stack(hi))
    assert naive < 0.7  # ~0.9^4
    m = MultiOutputConformalRegressor(0.1, method="max", seed=0).fit(y, p)
    assert joint_coverage(yt, *m.predict_interval(pt)) >= 0.88


def test_volume_ordering_with_correlated_targets():
    rng = np.random.default_rng(2)
    y, p = _data(rng, 4000, rho=0.95)
    vols = {
        m: MultiOutputConformalRegressor(0.1, method=m, seed=0).fit(y, p).region_volume()
        for m in ("bonferroni", "max", "mahalanobis")
    }
    assert vols["mahalanobis"] < 0.5 * vols["max"]
    assert vols["max"] <= vols["bonferroni"] * 1.02


def test_rectangle_contains_matches_intervals_and_ellipse_inside_box():
    rng = np.random.default_rng(3)
    y, p = _data(rng, 500)
    yt, pt = _data(rng, 500)
    for method in ("bonferroni", "max"):
        m = MultiOutputConformalRegressor(0.2, method=method, seed=0).fit(y, p)
        lo, hi = m.predict_interval(pt)
        box = np.all((yt >= lo) & (yt <= hi), axis=1)
        assert np.array_equal(box, m.contains(yt, pt))
    m = MultiOutputConformalRegressor(0.2, method="mahalanobis", seed=0).fit(y, p)
    lo, hi = m.predict_interval(pt)
    inside = m.contains(yt, pt)
    assert np.all(np.all((yt >= lo) & (yt <= hi), axis=1)[inside])


def test_unit_scale_and_single_target_reduce_to_split_conformal():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    p = np.zeros(200)
    sc = SplitConformalRegressor(0.1).fit(y, p)
    for method in ("bonferroni", "max", "mahalanobis"):
        m = MultiOutputConformalRegressor(0.1, method=method, scale_fraction=0.0).fit(y, p)
        assert m.half_widths()[0] == pytest.approx(sc.quantile_)


def test_validation():
    with pytest.raises(ValueError):
        MultiOutputConformalRegressor(method="copula")
    with pytest.raises(ValueError):
        MultiOutputConformalRegressor(alpha=1.0)
    with pytest.raises(ValueError):
        MultiOutputConformalRegressor(scale_fraction=1.0)
    m = MultiOutputConformalRegressor()
    with pytest.raises(RuntimeError):
        m.predict_interval([[0.0, 0.0]])
    with pytest.raises(ValueError):
        m.fit(np.zeros((10, 2)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        MultiOutputConformalRegressor(scale_fraction=0.1).fit(np.ones((10, 3)), np.zeros((10, 3)))
    rng = np.random.default_rng(5)
    y, p = _data(rng, 100)
    m.fit(y, p)
    with pytest.raises(ValueError):
        m.predict_interval(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        joint_coverage(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 2)))
