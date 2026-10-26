"""Tests for conformal selection (Jin & Candès 2023)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conformal_kit import (
    ConformalSelector,
    benjamini_hochberg,
    conformal_selection,
    selection_p_values,
    selection_report,
)
from conformal_kit.cli import main


def _simulate(rng, n, noise=1.0):
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=noise, size=n)
    mu = 1.5 * x + rng.normal(scale=0.3, size=n)  # imperfect predictor
    return y, mu


def test_p_values_match_the_definition_by_brute_force():
    rng = np.random.default_rng(1)
    y_cal, mu_cal = _simulate(rng, 40)
    _, mu_test = _simulate(rng, 15)
    c = 0.5
    for score in ("clipped", "residual"):
        p = selection_p_values(y_cal, mu_cal, mu_test, c, score=score)
        if score == "residual":
            v_cal = y_cal - mu_cal
        else:
            v_cal = np.where(y_cal > c, np.inf, c - mu_cal)
        v_test = c - mu_test
        expected = [(np.sum(v_cal < v) + 1 + np.sum(v_cal == v)) / 41 for v in v_test]
        np.testing.assert_allclose(p, expected)


def test_clipped_positives_never_count_against_test_units():
    y_cal = np.array([5.0, 5.0, 5.0, -5.0])
    mu_cal = np.array([-10.0, -10.0, -10.0, 0.0])
    p = selection_p_values(y_cal, mu_cal, [100.0], 0.0, score="clipped")
    # only the single null calibration unit could be below; it is not (0 - 0 > 0 - 100)
    assert p[0] == pytest.approx(1 / 5)
    p_res = selection_p_values(y_cal, mu_cal, [100.0], 0.0, score="residual")
    assert p_res[0] == pytest.approx(1 / 5)


def test_p_values_satisfy_the_joint_validity_bound():
    # Jin & Candes: P(p_j <= t, Y_j <= c) <= t for the conformal selection p-value.
    rng = np.random.default_rng(2)
    hits = []
    for _ in range(600):
        y_cal, mu_cal = _simulate(rng, 50)
        y_t, mu_t = _simulate(rng, 1)
        c = 0.0
        p = selection_p_values(y_cal, mu_cal, mu_t, c, randomized=True, random_state=rng)
        hits.append(p[0] <= 0.2 and y_t[0] <= c)
    assert np.mean(hits) <= 0.2 + 0.05


def test_fdr_is_controlled_and_power_is_positive():
    rng = np.random.default_rng(3)
    fdps, powers = [], []
    for _ in range(150):
        y_cal, mu_cal = _simulate(rng, 300, noise=0.7)
        y_t, mu_t = _simulate(rng, 200, noise=0.7)
        sel, _, _ = conformal_selection(y_cal, mu_cal, mu_t, 1.0, q=0.2)
        rep = selection_report(sel, y_t, 1.0)
        fdps.append(rep["fdp"])
        powers.append(rep["power"])
    assert np.mean(fdps) <= 0.2 + 0.03
    assert np.mean(powers) > 0.3


def test_clipped_is_at_least_as_powerful_as_residual_on_average():
    rng = np.random.default_rng(4)
    n_clip = n_res = 0
    for _ in range(60):
        y_cal, mu_cal = _simulate(rng, 200)
        _, mu_t = _simulate(rng, 200)
        n_clip += conformal_selection(y_cal, mu_cal, mu_t, 1.0, q=0.1)[0].sum()
        n_res += conformal_selection(y_cal, mu_cal, mu_t, 1.0, q=0.1, score="residual")[0].sum()
    assert n_clip >= n_res


def test_selection_uses_bh_on_the_p_values():
    rng = np.random.default_rng(5)
    y_cal, mu_cal = _simulate(rng, 100)
    _, mu_t = _simulate(rng, 50)
    sel, p, qv = conformal_selection(y_cal, mu_cal, mu_t, 0.5, q=0.15)
    ref_sel, ref_q = benjamini_hochberg(p, 0.15)
    np.testing.assert_array_equal(sel, ref_sel)
    np.testing.assert_allclose(qv, ref_q)


def test_per_unit_thresholds_and_validation():
    y_cal = np.array([-1.0, -2.0, -3.0])
    mu_cal = np.array([1.0, 2.0, 3.0])
    p = selection_p_values(y_cal, mu_cal, [10.0, -10.0], [0.0, 0.0, 0.0], test_threshold=[0.0, 0.0])
    assert p[0] < p[1]
    with pytest.raises(ValueError, match="test_threshold"):
        selection_p_values(y_cal, mu_cal, [1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="length"):
        selection_p_values(y_cal, mu_cal[:2], [1.0], 0.0)
    with pytest.raises(ValueError, match="score"):
        selection_p_values(y_cal, mu_cal, [1.0], 0.0, score="nope")
    with pytest.raises(ValueError, match="finite"):
        selection_p_values(y_cal, mu_cal, [np.nan], 0.0)


def test_selector_wrapper_and_report():
    rng = np.random.default_rng(6)
    y_cal, mu_cal = _simulate(rng, 200)
    y_t, mu_t = _simulate(rng, 100)
    selector = ConformalSelector(threshold=1.0, q=0.1)
    with pytest.raises(RuntimeError):
        selector.select(mu_t)
    selector.fit(y_cal, mu_cal)
    report = selector.decision_report(mu_t)
    np.testing.assert_array_equal(report["selected"], selector.select(mu_t))
    assert report["n_selected"] == int(report["selected"].sum())
    assert report["n_calibration"] == 200
    with pytest.raises(ValueError):
        ConformalSelector(threshold=0.0, q=1.5)
    rep = selection_report([False, False], [0.0, 2.0], 1.0)
    assert rep["fdp"] == 0.0 and rep["power"] == 0.0
    assert np.isnan(selection_report([True], [0.0], 1.0)["power"])


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_cli_select(tmp_path, capsys):
    rng = np.random.default_rng(7)
    y_cal, mu_cal = _simulate(rng, 300)
    y_t, mu_t = _simulate(rng, 100)
    cal = _write(tmp_path / "cal.csv", ["y_true", "y_pred"], zip(y_cal, mu_cal))
    test = _write(tmp_path / "test.csv", ["y_true", "y_pred"], zip(y_t, mu_t))
    out = tmp_path / "sel.csv"
    code = main([
        "select", "--calibration", cal, "--test", test, "--threshold", "1.0",
        "--alpha", "0.2", "--test-labels", "--out", str(out), "--json",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    expected, _, _ = conformal_selection(y_cal, mu_cal, mu_t, 1.0, q=0.2)
    assert summary["n_selected"] == int(expected.sum())
    assert 0.0 <= summary["fdp"] <= 1.0
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    assert sum(int(r["selected"]) for r in rows) == summary["n_selected"]

    assert main(["select", "--calibration", cal, "--test", test, "--threshold", "1.0"]) == 0
    assert "Selected" in capsys.readouterr().out
