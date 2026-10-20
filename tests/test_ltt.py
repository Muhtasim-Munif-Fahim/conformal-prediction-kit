"""Tests for Learn then Test high-probability risk control."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from conformal_kit import (
    LTTMultilabelController,
    MultilabelRiskController,
    bentkus_p_value,
    binomial_p_value,
    false_negative_rate,
    fwer_reject,
    hoeffding_bentkus_p_value,
    hoeffding_p_value,
    learn_then_test,
    ltt_p_values,
)
from conformal_kit.cli import main


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _multilabel(rng, n, n_labels=8):
    """Calibrated sigmoid probabilities and labels drawn from them."""
    logits = rng.normal(-1.0, 1.5, size=(n, n_labels))
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.random((n, n_labels)) < p
    return y.astype(int), p


def test_p_values_match_closed_forms():
    n, alpha = 100, 0.1
    assert binomial_p_value(0.05, n, alpha) == pytest.approx(_binom_cdf(5, n, alpha), rel=1e-10)
    assert bentkus_p_value(0.05, n, alpha) == pytest.approx(
        math.e * _binom_cdf(5, n, alpha), rel=1e-10
    )
    # Zero empirical risk: Hoeffding KL form reduces to (1 - alpha)^n.
    assert hoeffding_p_value(0.0, n, alpha) == pytest.approx(0.9**100, rel=1e-10)
    r = 0.04
    kl = r * math.log(r / alpha) + (1 - r) * math.log((1 - r) / (1 - alpha))
    assert hoeffding_p_value(r, n, alpha) == pytest.approx(math.exp(-n * kl), rel=1e-10)
    grid = np.linspace(0, 0.3, 31)
    hb = hoeffding_bentkus_p_value(grid, n, alpha)
    assert np.allclose(hb, np.minimum(hoeffding_p_value(grid, n, alpha), bentkus_p_value(grid, n, alpha)))
    assert np.all(np.diff(hb) >= -1e-12)  # larger empirical risk, weaker evidence
    assert np.all(hb[grid >= alpha] == 1.0)
    assert np.all((hb >= 0) & (hb <= 1))


def test_p_values_are_super_uniform_at_the_null_boundary():
    rng = np.random.default_rng(0)
    n, alpha, delta, trials = 200, 0.1, 0.1, 4000
    risk = rng.binomial(n, alpha, size=trials) / n
    for fn in (binomial_p_value, hoeffding_bentkus_p_value):
        p = np.array([fn(r, n, alpha) for r in risk])
        assert np.mean(p <= delta) <= delta + 0.01
    # The exact binomial p-value is much less conservative than HB.
    p_bin = np.array([binomial_p_value(r, n, alpha) for r in risk])
    p_hb = np.array([hoeffding_bentkus_p_value(r, n, alpha) for r in risk])
    assert np.all(p_bin <= p_hb + 1e-12)
    with pytest.raises(ValueError, match="0/1"):
        binomial_p_value(0.0525, n, alpha)


def test_fwer_procedures():
    p = np.array([0.001, 0.02, 0.04, 0.2, 0.011])
    bonf = fwer_reject(p, 0.05, "bonferroni")
    holm = fwer_reject(p, 0.05, "holm")
    assert bonf.tolist() == [True, False, False, False, False]
    # Holm: 0.001<=0.01, 0.011<=0.0125, 0.02<=0.0167? no -> stop.
    assert holm.tolist() == [True, False, False, False, True]
    assert np.all(holm >= bonf)
    seq = fwer_reject(p, 0.05, "fixed_sequence", order=[4, 0, 1, 3, 2])
    assert seq.tolist() == [True, True, False, False, True]  # stops at index 3
    with pytest.raises(ValueError, match="order"):
        fwer_reject(p, 0.05, "fixed_sequence", order=[0, 0, 1, 2, 3])
    with pytest.raises(ValueError, match="method"):
        fwer_reject(p, 0.05, "bh")


def test_non_monotone_loss_certifies_only_safe_region():
    # Loss is low at both ends and bad in the middle; CRC/RCPS would need
    # monotonicity, LTT with Bonferroni handles it.
    rng = np.random.default_rng(1)
    lambdas = np.linspace(0, 1, 21)
    true_risk = 0.02 + 0.3 * np.exp(-((lambdas - 0.5) ** 2) / 0.02)
    losses = (rng.random((2000, lambdas.size)) < true_risk).astype(float)
    res = learn_then_test(losses, lambdas, alpha=0.1, delta=0.1, method="bonferroni")
    assert res.n_valid > 0
    assert np.all(true_risk[res.valid] <= 0.1)
    assert res.valid[0] and res.valid[-1]
    assert not res.valid[10]
    holm = learn_then_test(losses, lambdas, alpha=0.1, delta=0.1, method="holm", p_value="binomial")
    assert np.all(holm.valid >= res.valid)
    assert res.to_dict()["n_valid"] == res.n_valid


def test_ltt_high_probability_guarantee_vs_crc():
    rng = np.random.default_rng(2)
    alpha, delta = 0.15, 0.1
    y_test, p_test = _multilabel(rng, 20000)
    ltt_fail = crc_fail = 0
    trials = 80
    for _ in range(trials):
        y, p = _multilabel(rng, 400)
        ltt = LTTMultilabelController(alpha=alpha, delta=delta).fit(y, p)
        crc = MultilabelRiskController(alpha=alpha).fit(y, p)
        ltt_fail += false_negative_rate(y_test, ltt.predict_set(p_test)) > alpha
        crc_fail += false_negative_rate(y_test, crc.predict_set(p_test)) > alpha
        assert ltt.lambda_ >= crc.lambda_ - 1e-2  # LTT is the more conservative rule
    assert ltt_fail / trials <= delta
    # CRC controls only the mean, so it exceeds alpha far more often.
    assert crc_fail / trials > 0.25
    assert ltt.calibration_risk_ <= alpha
    labels = ltt.predict_labels(p_test[:3])
    assert len(labels) == 3


def test_scaled_losses_and_selection():
    rng = np.random.default_rng(3)
    lambdas = np.arange(10, dtype=float)
    base = np.clip(1.0 - lambdas / 9.0, 0, 1)
    losses = 5.0 * np.clip(base + rng.normal(0, 0.05, size=(500, 10)), 0, 1)
    res = learn_then_test(losses, lambdas, alpha=1.5, delta=0.1, loss_bound=5.0)
    risk, p = ltt_p_values(losses, 1.5, loss_bound=5.0)
    assert np.allclose(res.risk_hat, risk) and np.allclose(res.p_values, p)
    assert res.lambda_hat == lambdas[res.valid].min()
    assert np.all(risk[res.valid] < 1.5)
    hi = learn_then_test(losses, lambdas, alpha=1.5, delta=0.1, loss_bound=5.0, select="max")
    assert hi.lambda_hat == 9.0


def test_nothing_certified_and_validation():
    losses = np.full((20, 3), 0.5)
    res = learn_then_test(losses, [0.1, 0.2, 0.3], alpha=0.1, delta=0.1)
    assert res.lambda_hat is None and res.n_valid == 0
    with pytest.raises(ValueError, match="no threshold"):
        y = np.ones((5, 3), dtype=int)
        LTTMultilabelController(alpha=0.05, delta=0.05).fit(y, np.full((5, 3), 0.5))
    with pytest.raises(ValueError, match="shape"):
        learn_then_test(np.zeros((5, 2)), [0.1, 0.2, 0.3], 0.1)
    with pytest.raises(ValueError, match="distinct"):
        learn_then_test(np.zeros((5, 2)), [0.1, 0.1], 0.1)
    with pytest.raises(ValueError, match="loss_bound"):
        ltt_p_values(np.full((5, 2), 2.0), 0.1)
    with pytest.raises(ValueError, match="delta"):
        learn_then_test(np.zeros((5, 2)), [0.1, 0.2], 0.1, delta=1.0)
    with pytest.raises(ValueError, match="p_value"):
        LTTMultilabelController(p_value="binomial")
    with pytest.raises(RuntimeError):
        LTTMultilabelController().predict_set(np.zeros((1, 2)))


def test_cli_ltt(tmp_path, capsys):
    rng = np.random.default_rng(4)
    lambdas = np.linspace(0, 1, 11)
    losses = (rng.random((800, 1)) < (0.4 * (1 - lambdas))).astype(int)  # monotone rows
    path = tmp_path / "losses.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{v:g}" for v in lambdas])
        writer.writerows(losses.tolist())
    out = tmp_path / "ltt.csv"
    code = main(["ltt", "--losses", str(path), "--alpha", "0.1", "--delta", "0.1",
                 "--p-value", "binomial", "--json", "--out", str(out)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["lambda_hat"] is not None
    assert summary["crc_lambda"] <= summary["lambda_hat"]
    assert summary["n_candidates"] == 11
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11 and set(rows[0]) == {"lambda", "risk_hat", "p_value", "valid"}
    assert main(["ltt", "--losses", str(path), "--alpha", "0.001"]) == 1
    assert "Certified" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    assert main(["ltt", "--losses", str(bad)]) == 2
