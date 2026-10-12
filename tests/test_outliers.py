"""Tests for conformal outlier detection (conformal p-values + BH)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conformal_kit import ConformalOutlierDetector, benjamini_hochberg, conformal_p_values
from conformal_kit.cli import main


def test_p_value_formula_by_hand():
    cal = [1.0, 2.0, 3.0, 4.0]
    p = conformal_p_values(cal, [0.0, 2.0, 3.5, 10.0])
    # (1 + #{s_i >= s}) / (n + 1)
    np.testing.assert_allclose(p, [5 / 5, 4 / 5, 2 / 5, 1 / 5])


def test_randomized_p_values_bracket_deterministic():
    cal = [1.0, 2.0, 2.0, 2.0, 5.0]
    det = conformal_p_values(cal, [2.0])
    rnd = conformal_p_values(cal, [2.0] * 1000, randomized=True, random_state=0)
    greater = 1  # only 5.0 > 2.0
    assert np.all(rnd >= greater / 6) and np.all(rnd <= det[0])
    assert rnd.mean() == pytest.approx((greater + 0.5 * 4) / 6, abs=0.02)


def test_marginal_validity_under_exchangeability():
    rng = np.random.default_rng(0)
    alpha = 0.1
    hits = 0
    trials = 4000
    for _ in range(trials):
        cal = rng.normal(size=19)
        hits += conformal_p_values(cal, rng.normal(size=1))[0] <= alpha
    # n=19 -> P(p <= 0.1) = floor(0.1 * 20) / 20 = 0.1 exactly.
    assert hits / trials == pytest.approx(0.1, abs=0.015)


def test_randomized_p_values_are_uniform_under_null():
    # Exact uniformity holds jointly over calibration and test draws, even with ties.
    rng = np.random.default_rng(1)
    p = np.array([
        conformal_p_values(
            rng.integers(0, 5, size=30).astype(float),
            rng.integers(0, 5, size=1).astype(float),
            randomized=True,
            random_state=seed,
        )[0]
        for seed in range(6000)
    ])
    for t in (0.1, 0.25, 0.5, 0.9):
        assert np.mean(p <= t) == pytest.approx(t, abs=0.02)


def test_bh_matches_statsmodels():
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=80), rng.uniform(0, 0.01, size=20)])
    rejected, q = benjamini_hochberg(p, 0.1)
    ref_rej, ref_q, _, _ = multipletests(p, alpha=0.1, method="fdr_bh")
    np.testing.assert_array_equal(rejected, ref_rej)
    np.testing.assert_allclose(q, ref_q)


def test_bh_hand_example_and_empty():
    rejected, q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5], 0.05)
    # sorted 0.01, 0.03, 0.04, 0.5 vs 0.0125, 0.025, 0.0375, 0.05 -> only first passes
    assert rejected.tolist() == [True, False, False, False]
    np.testing.assert_allclose(q, [0.04, 0.0533333333, 0.0533333333, 0.5])
    r0, q0 = benjamini_hochberg([], 0.1)
    assert r0.size == 0 and q0.size == 0


def test_detector_controls_fdr_and_has_power():
    rng = np.random.default_rng(4)
    alpha = 0.1
    fdps, powers = [], []
    for _ in range(200):
        cal = rng.normal(size=500)
        inliers = rng.normal(size=180)
        outliers = rng.normal(3.5, 1.0, size=20)
        test = np.concatenate([inliers, outliers])
        truth = np.r_[np.zeros(180), np.ones(20)].astype(bool)
        flags = ConformalOutlierDetector(alpha=alpha).fit(cal).predict(test).astype(bool)
        fdps.append(np.sum(flags & ~truth) / max(1, flags.sum()))
        powers.append(np.mean(flags[truth]))
    assert np.mean(fdps) <= alpha + 0.02
    assert np.mean(powers) > 0.5


def test_marginal_method_flags_more_than_bh():
    rng = np.random.default_rng(5)
    cal = rng.normal(size=300)
    test = rng.normal(size=500)
    marginal = ConformalOutlierDetector(alpha=0.1, method="marginal").fit(cal).predict(test)
    bh = ConformalOutlierDetector(alpha=0.1, method="bh").fit(cal).predict(test)
    assert marginal.sum() >= bh.sum()
    assert 0.05 < marginal.mean() < 0.15
    report = ConformalOutlierDetector(alpha=0.1, method="marginal").fit(cal).decision_report(test)
    assert report["n_flagged"] == marginal.sum()
    assert np.all(report["q_values"] >= report["p_values"] - 1e-12)


def test_validation():
    with pytest.raises(ValueError, match="alpha"):
        ConformalOutlierDetector(alpha=0.0)
    with pytest.raises(ValueError, match="method"):
        ConformalOutlierDetector(method="bonferroni")
    with pytest.raises(ValueError, match="at least 9"):
        ConformalOutlierDetector(alpha=0.1).fit(np.arange(5.0))
    with pytest.raises(RuntimeError, match="fit"):
        ConformalOutlierDetector().predict([1.0])
    with pytest.raises(ValueError, match="finite"):
        conformal_p_values([1.0, np.nan], [0.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([1.5], 0.1)


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["score"])
        writer.writerows([[r] for r in rows])
    return str(path)


def test_cli_outliers(tmp_path, capsys):
    rng = np.random.default_rng(6)
    cal = _write(tmp_path / "cal.csv", rng.normal(size=400))
    test = _write(tmp_path / "test.csv", np.r_[rng.normal(size=90), rng.normal(6.0, 0.5, size=10)])
    out = tmp_path / "flags.csv"
    code = main(["outliers", "--calibration", cal, "--test", test, "--out", str(out), "--json"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["method"] == "bh" and summary["n_test"] == 100
    assert 10 <= summary["n_flagged"] <= 13
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == {"score", "p_value", "q_value", "outlier"}
    assert sum(int(r["outlier"]) for r in rows[-10:]) == 10

    code = main(["outliers", "--calibration", cal, "--test", test, "--method", "marginal"])
    assert code == 0
    assert "per-point false-alarm rate" in capsys.readouterr().out
