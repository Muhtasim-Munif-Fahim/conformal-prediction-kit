"""Tests for conformal PID control."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from conformal_kit import (
    ConformalPIDRegressor,
    conformal_pid,
    default_c_sat,
    ewma_scorecaster,
    pid_saturation,
    quantile_tracker_update,
)
from conformal_kit.cli import main


def _shift_stream(T=3000, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.where(np.arange(T) < T // 2, 1.0, 3.0)
    return np.abs(rng.normal(0.0, scale))


def test_quantile_tracker_update_direction():
    assert quantile_tracker_update(1.0, 1, 0.1, 0.5) == pytest.approx(1.45)
    assert quantile_tracker_update(1.0, 0, 0.1, 0.5) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        quantile_tracker_update(1.0, 0.5, 0.1, 0.5)


def test_saturation_shape():
    assert pid_saturation(5.0, 1, 1.0, 1.0) == 0.0
    assert pid_saturation(0.0, 100, 1.0, 1.0) == 0.0
    val = pid_saturation(3.0, 100, 2.0, 1.0)
    assert val == pytest.approx(2.0 * math.tan(3.0 * math.log(100) / 100))
    assert pid_saturation(-3.0, 100, 2.0, 1.0) == pytest.approx(-val)
    assert pid_saturation(1000.0, 100, 1.0, 1.0) == math.inf
    assert pid_saturation(-1000.0, 100, 1.0, 1.0) == -math.inf
    with pytest.raises(ValueError):
        pid_saturation(1.0, 10, 1.0, None)


def test_default_c_sat_positive():
    assert default_c_sat(0.1, 3000) >= 1.0
    assert default_c_sat(0.1, 1) >= 1.0


def test_long_run_coverage_bound_any_stream():
    rng = np.random.default_rng(1)
    # Adversarial-ish: bounded but wildly non-stationary scores.
    scores = np.concatenate([rng.uniform(0, 1, 1000), rng.uniform(4, 5, 1000), rng.uniform(0, 0.1, 1000)])
    lr = 0.2
    res = conformal_pid(scores, alpha=0.1, lr=lr)
    bound = (scores.max() + lr) / (lr * scores.size)
    assert abs(res.miscoverage - 0.1) <= bound + 1e-12


def test_pid_beats_fixed_quantile_after_shift():
    scores = _shift_stream()
    fixed_q = np.quantile(scores[:300], 0.9)
    fixed_miss = 1 - np.mean(scores[1500:] <= fixed_q)
    res = conformal_pid(scores, alpha=0.1, lr=0.1)
    assert fixed_miss > 0.5
    assert abs(res.miscoverage - 0.1) < 0.03
    assert np.median(res.thresholds[2000:]) > 2 * np.median(res.thresholds[500:1500])


def test_integrator_and_scorecaster_speed_up_recovery():
    scores = _shift_stream()
    post = slice(1500, 1700)
    base = conformal_pid(scores, alpha=0.1, lr=0.1)
    integ = conformal_pid(scores, alpha=0.1, lr=0.1, k_i=50.0, c_sat=0.5)
    cast = conformal_pid(scores, alpha=0.1, lr=0.1, scorecaster=ewma_scorecaster(20))
    base_miss = 1 - base.covered[post].mean()
    assert 1 - integ.covered[post].mean() < base_miss
    assert 1 - cast.covered[post].mean() < base_miss


def test_rolling_miscoverage():
    res = conformal_pid(_shift_stream(500), alpha=0.1, lr=0.1)
    roll = res.rolling_miscoverage(50)
    assert roll.shape == (500,)
    assert roll[-1] == pytest.approx(1 - res.covered[-50:].mean())
    with pytest.raises(ValueError):
        res.rolling_miscoverage(0)


def test_conformal_pid_validation():
    with pytest.raises(ValueError):
        conformal_pid([], 0.1)
    with pytest.raises(ValueError):
        conformal_pid([1.0, np.nan], 0.1)
    with pytest.raises(ValueError):
        conformal_pid([1.0], 1.5)
    with pytest.raises(ValueError):
        conformal_pid([1.0], 0.1, lr=0)
    with pytest.raises(TypeError):
        conformal_pid([1.0], 0.1, scorecaster=3)


def test_ewma_scorecaster():
    f = ewma_scorecaster(1.0)
    assert f([]) == 0.0
    assert f([2.0]) == 2.0
    assert f([0.0, 4.0]) == pytest.approx(4.0 / 1.5)


def test_regressor_fit_seed_and_stream():
    rng = np.random.default_rng(3)
    y_cal = rng.normal(size=300)
    p_cal = y_cal + rng.normal(0, 0.5, 300)
    model = ConformalPIDRegressor(alpha=0.1, lr=0.05).fit(y_cal, p_cal)
    assert model.threshold_ > 0
    assert model.empirical_miscoverage is None
    y = rng.normal(size=2000)
    noise = np.where(np.arange(2000) < 1000, 0.5, 2.0)
    pred = y + rng.normal(0, noise)
    lower, upper = model.predict_update(pred, y)
    assert np.all(lower <= upper)
    covered = (y >= lower) & (y <= upper)
    assert np.array_equal(covered, np.asarray(model.covered_))
    assert abs(model.empirical_miscoverage - 0.1) < 0.03
    with pytest.raises(ValueError):
        model.predict_update([1.0, 2.0], [1.0])


def test_regressor_without_fit_learns_scale():
    rng = np.random.default_rng(4)
    y = rng.normal(size=3000)
    pred = y + rng.normal(0, 1.0, 3000)
    model = ConformalPIDRegressor(alpha=0.2, lr=0.05, k_i=1.0)
    model.predict_update(pred, y)
    assert abs(model.empirical_miscoverage - 0.2) < 0.03


def _stream_csv(tmp_path, n=600):
    rng = np.random.default_rng(5)
    y = rng.normal(size=n)
    pred = y + rng.normal(0, np.where(np.arange(n) < n // 2, 0.5, 1.5))
    path = tmp_path / "stream.csv"
    with open(path, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["y_true", "y_pred"])
        w.writerows(zip(y, pred))
    return str(path)


def test_cli_pid_json_and_out(tmp_path, capsys):
    out = tmp_path / "pid.csv"
    code = main(["pid", "--stream", _stream_csv(tmp_path), "--alpha", "0.1", "--lr", "0.1",
                 "--warmup", "100", "--ewma-halflife", "10", "--out", str(out), "--json"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_stream"] == 500 and summary["n_warmup"] == 100
    assert 0.8 < summary["coverage"] <= 1.0
    with open(out, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 500 and set(rows[0]) == {"y_true", "y_pred", "lower", "upper", "covered"}


def test_cli_pid_rejects_bad_args(tmp_path, capsys):
    path = _stream_csv(tmp_path, 50)
    assert main(["pid", "--stream", path, "--lr", "0"]) == 2
    assert main(["pid", "--stream", path, "--warmup", "50"]) == 2
    assert main(["pid", "--stream", path, "--k-i", "1", "--c-sat", "-1"]) == 2
