"""Tests for exact full conformal ridge regression (RRCM)."""

import csv
import json

import numpy as np
import pytest

from conformal_kit import FullConformalRidgeRegressor, ridge_conformal_set
from conformal_kit.cli import main


def _data(n=40, seed=0, noise="normal"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    eps = rng.normal(size=n) if noise == "normal" else rng.standard_t(2, size=n)
    return X, X @ np.array([1.0, -2.0, 0.5]) + 0.3 + eps


def _brute_p_value(X, y, x, z, ridge, intercept=True):
    Xa = np.vstack([X, x])
    if intercept:
        Xa = np.hstack([Xa, np.ones((Xa.shape[0], 1))])
    ya = np.r_[y, z]
    P = np.eye(Xa.shape[1]) * ridge
    if intercept:
        P[-1, -1] = 0.0
    w = np.linalg.solve(Xa.T @ Xa + P, Xa.T @ ya)
    r = np.abs(ya - Xa @ w)
    return np.mean(r >= r[-1] - 1e-9)


def test_p_values_match_explicit_refits():
    X, y = _data()
    model = FullConformalRidgeRegressor(alpha=0.1, ridge=2.0).fit(X, y)
    x = np.array([0.3, -1.0, 2.0])
    zs = np.array([-5.0, 0.0, 1.7, 4.0, 9.0])
    got = model.p_value(x, zs)
    want = [_brute_p_value(X, y, x, z, 2.0) for z in zs]
    assert np.allclose(got, want)


@pytest.mark.parametrize("seed", range(8))
def test_exact_set_matches_dense_grid(seed):
    X, y = _data(n=25, seed=seed, noise="t")
    rng = np.random.default_rng(100 + seed)
    model = FullConformalRidgeRegressor(alpha=0.2, ridge=0.5, fit_intercept=bool(seed % 2)).fit(X, y)
    x = rng.normal(size=3) * 2
    pieces = model.predict_set(x[None, :])[0]
    grid = np.linspace(-40, 40, 16001)
    keep = model.p_value(x, grid) > 0.2
    inside = np.array([any(lo - 1e-9 <= z <= hi + 1e-9 for lo, hi in pieces) for z in grid])
    assert np.array_equal(inside, keep)


def test_marginal_coverage_and_point_prediction():
    X, y = _data(n=2000, seed=3)
    train, test = slice(0, 60), slice(60, 2000)
    model = FullConformalRidgeRegressor(alpha=0.1, ridge=1.0).fit(X[train], y[train])
    lower, upper = model.predict_interval(X[test])
    covered = (y[test] >= lower) & (y[test] <= upper)
    assert 0.86 <= covered.mean() <= 0.96
    # ridge point prediction lies inside its interval and matches the closed form
    Xd = np.hstack([X[train], np.ones((60, 1))])
    P = np.eye(4)
    P[-1, -1] = 0
    coef = np.linalg.solve(Xd.T @ Xd + P, Xd.T @ y[train])
    assert np.allclose(model.predict(X[test][:5]), np.hstack([X[test][:5], np.ones((5, 1))]) @ coef)
    pred = model.predict(X[test])
    assert np.all((pred >= lower) & (pred <= upper))


def test_smaller_alpha_gives_wider_intervals_and_trivial_case():
    X, y = _data(n=30, seed=4)
    x = X[:3] + 0.1
    lo1, hi1 = FullConformalRidgeRegressor(alpha=0.2).fit(X, y).predict_interval(x)
    lo2, hi2 = FullConformalRidgeRegressor(alpha=0.05).fit(X, y).predict_interval(x)
    assert np.all(hi2 - lo2 >= hi1 - lo1 - 1e-12)
    # (n + 1) * alpha < 1: every label is plausible
    lo, hi = FullConformalRidgeRegressor(alpha=0.02).fit(X[:10], y[:10]).predict_interval(x)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_one_dimensional_input_and_helper():
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, size=50)
    y = 2 * x + rng.normal(scale=0.2, size=50)
    model = FullConformalRidgeRegressor(alpha=0.1, ridge=0.1).fit(x, y)
    lo, hi = model.predict_interval(np.array([0.0, 1.0]))
    assert lo.shape == (2,) and np.all(hi - lo < 1.5)
    assert hi[1] > hi[0]
    pieces = ridge_conformal_set(x[:, None], y, [1.0], alpha=0.1, ridge=0.1)
    assert pieces[0][0] == pytest.approx(lo[1]) and pieces[-1][1] == pytest.approx(hi[1])


def test_validation():
    X, y = _data(n=10)
    with pytest.raises(ValueError, match="alpha"):
        FullConformalRidgeRegressor(alpha=1.0)
    with pytest.raises(ValueError, match="ridge"):
        FullConformalRidgeRegressor(ridge=0.0)
    with pytest.raises(ValueError, match="same number"):
        FullConformalRidgeRegressor().fit(X, y[:5])
    with pytest.raises(ValueError, match="finite"):
        FullConformalRidgeRegressor().fit(X, np.r_[y[:-1], np.nan])
    with pytest.raises(RuntimeError):
        FullConformalRidgeRegressor().predict_interval(X)
    model = FullConformalRidgeRegressor().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.p_value(np.zeros(5), 0.0)


def _write(path, X, y=None):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [f"f{j}" for j in range(X.shape[1])] + (["y_true"] if y is not None else [])
        writer.writerow(header)
        for i in range(X.shape[0]):
            writer.writerow(list(X[i]) + ([y[i]] if y is not None else []))


def test_cli_full_ridge(tmp_path, capsys):
    X, y = _data(n=300, seed=6)
    train, test, out = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "iv.csv"
    _write(train, X[:80], y[:80])
    _write(test, X[80:], y[80:])
    assert main(["full-ridge", "--train", str(train), "--test", str(test), "--out", str(out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["features"] == ["f0", "f1", "f2"] and summary["n_train"] == 80
    assert 0.8 <= summary["coverage"] <= 1.0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 220 and float(rows[0]["lower"]) < float(rows[0]["upper"])
    assert main(["full-ridge", "--train", str(train), "--test", str(test), "--features", "f0,f9"]) == 2
