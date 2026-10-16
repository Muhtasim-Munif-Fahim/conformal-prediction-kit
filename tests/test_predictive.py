"""Tests for split conformal predictive distributions."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conformal_kit import (
    SplitConformalPredictiveSystem,
    SplitConformalRegressor,
    crps_empirical,
    pit_uniformity_report,
)
from conformal_kit.cli import main


def _data(n, seed, hetero=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 4.0, n)
    sigma = 0.2 + x if hetero else np.ones(n)
    y = 2.0 * x + rng.standard_t(5, n) * sigma
    return y, 2.0 * x, sigma


def test_cdf_step_values_and_bounds():
    system = SplitConformalPredictiveSystem().fit([1.0, 2.0, 3.0, 4.0], [0.0] * 4)
    # residuals 1,2,3,4 ; n + 1 = 5
    assert system.cdf(0.5, 0.0, tau=0.0) == pytest.approx(0.0)
    assert system.cdf(0.5, 0.0, tau=1.0) == pytest.approx(0.2)
    assert system.cdf(2.5, 0.0, tau=0.0) == pytest.approx(0.4)
    # exactly at a residual: the tie widens the jump
    assert system.cdf(2.0, 0.0, tau=1.0) == pytest.approx(0.6)
    assert system.cdf(10.0, 0.0, tau=1.0) == pytest.approx(1.0)
    grid = np.linspace(-1, 6, 50)
    values = system.cdf(grid, 0.0, tau=0.3)
    assert np.all(np.diff(values) >= 0)


def test_pit_is_uniform_on_exchangeable_data():
    y_cal, p_cal, _ = _data(800, 0)
    y_test, p_test, _ = _data(4000, 1)
    system = SplitConformalPredictiveSystem().fit(y_cal, p_cal)
    report = pit_uniformity_report(system.pit(y_test, p_test, seed=3))
    assert report["ks_statistic"] < 0.04
    # conditional on one calibration draw, bin shares wobble by ~1/sqrt(n_cal)
    assert report["max_bin_deviation"] < 0.04


def test_quantiles_have_calibrated_frequencies():
    y_cal, p_cal, _ = _data(1000, 2)
    y_test, p_test, _ = _data(5000, 3)
    system = SplitConformalPredictiveSystem().fit(y_cal, p_cal)
    levels = [0.1, 0.5, 0.9]
    q = system.quantile(levels, p_test)
    assert q.shape == (5000, 3)
    freq = np.mean(y_test[:, None] <= q, axis=0)
    assert freq == pytest.approx(levels, abs=0.04)


def test_extreme_upper_quantile_is_infinite_with_small_calibration():
    system = SplitConformalPredictiveSystem().fit(np.arange(5.0), np.zeros(5))
    q = system.quantile([0.01, 0.5, 0.99], [0.0])
    # P(Y <= q_p) >= p: low levels clamp to the smallest residual, high
    # levels beyond rank n are unresolvable and return +inf.
    assert q[0, 0] == 0.0
    assert q[0, 1] == 2.0
    assert q[0, 2] == np.inf


def test_interval_matches_split_conformal_coverage():
    y_cal, p_cal, _ = _data(500, 4)
    y_test, p_test, _ = _data(4000, 5)
    system = SplitConformalPredictiveSystem().fit(y_cal, p_cal)
    lo, hi = system.predict_interval(p_test, alpha=0.1)
    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    assert coverage >= 0.88
    # comparable width to the absolute-residual split conformal interval
    width = np.mean(hi - lo)
    baseline = SplitConformalRegressor(alpha=0.1).fit(y_cal, p_cal).width
    assert width == pytest.approx(baseline, rel=0.25)


def test_normalized_system_adapts_to_heteroskedasticity():
    y_cal, p_cal, s_cal = _data(1000, 6, hetero=True)
    y_test, p_test, s_test = _data(4000, 7, hetero=True)
    plain = SplitConformalPredictiveSystem().fit(y_cal, p_cal)
    norm = SplitConformalPredictiveSystem(normalize=True).fit(y_cal, p_cal, difficulty=s_cal)
    crps_plain = plain.crps(y_test, p_test).mean()
    crps_norm = norm.crps(y_test, p_test, difficulty=s_test).mean()
    assert crps_norm < crps_plain
    pit = norm.pit(y_test, p_test, difficulty=s_test, seed=0)
    assert pit_uniformity_report(pit)["ks_statistic"] < 0.04


def test_crps_empirical_matches_brute_force():
    rng = np.random.default_rng(8)
    samples = rng.normal(size=60)
    y = 0.3
    brute = np.mean(np.abs(samples - y)) - 0.5 * np.mean(
        np.abs(samples[:, None] - samples[None, :])
    )
    assert crps_empirical(samples, y) == pytest.approx(brute)
    assert crps_empirical([2.0], 2.0) == pytest.approx(0.0)


def test_errors():
    system = SplitConformalPredictiveSystem()
    with pytest.raises(RuntimeError):
        system.quantile([0.5], [0.0])
    with pytest.raises(ValueError):
        system.fit([1.0], [1.0, 2.0])
    system.fit([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        system.quantile([1.0], [0.0])
    with pytest.raises(ValueError):
        system.cdf(0.0, 0.0, difficulty=[1.0])
    with pytest.raises(ValueError):
        SplitConformalPredictiveSystem(normalize=True).fit([1.0], [0.0])
    with pytest.raises(ValueError):
        pit_uniformity_report([1.5])


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_cli_predictive(tmp_path, capsys):
    y_cal, p_cal, _ = _data(300, 9)
    y_test, p_test, _ = _data(200, 10)
    cal = _write(tmp_path / "cal.csv", ["y_true", "y_pred"], zip(y_cal, p_cal))
    test = _write(tmp_path / "test.csv", ["y_true", "y_pred"], zip(y_test, p_test))
    out = tmp_path / "q.csv"
    code = main([
        "predictive", "--calibration", cal, "--test", test,
        "--quantiles", "0.1,0.5,0.9", "--out", str(out), "--json", "--seed", "1",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_calibration"] == 300
    assert summary["mean_crps"] > 0
    assert 0.8 <= summary["interval_coverage"] <= 1.0
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["y_pred", "lower", "upper", "q0.1", "q0.5", "q0.9"]
    assert len(rows) == 201


def test_cli_predictive_unlabelled_test(tmp_path, capsys):
    y_cal, p_cal, _ = _data(100, 11)
    cal = _write(tmp_path / "cal.csv", ["y_true", "y_pred"], zip(y_cal, p_cal))
    test = _write(tmp_path / "test.csv", ["y_pred"], [[1.0], [2.0]])
    assert main(["predictive", "--calibration", cal, "--test", test]) == 0
    text = capsys.readouterr().out
    assert "Conformal predictive system on 100" in text
    assert "CRPS" not in text
