import json

import numpy as np
import pytest

from conformal_kit import (
    conformal_test_martingale,
    exchangeability_test,
    online_conformal_p_values,
)
from conformal_kit.cli import main


def _shift(n=400, at=200, delta=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x[at:] += delta
    return x


def test_p_values_uniform_under_iid():
    rng = np.random.default_rng(1)
    p = online_conformal_p_values(rng.normal(size=2000), random_state=2)
    assert p.min() >= 0 and p.max() <= 1
    # KS distance to U(0,1)
    s = np.sort(p)
    grid = np.arange(1, s.size + 1) / s.size
    assert np.max(np.abs(s - grid)) < 0.04


def test_conservative_p_values_dominate_randomized():
    x = np.random.default_rng(3).normal(size=50)
    pr = online_conformal_p_values(x, random_state=0)
    pc = online_conformal_p_values(x, randomized=False)
    assert np.all(pc >= pr - 1e-12)
    assert pc[0] == 1.0


def test_identity_and_callable_scores():
    s = np.array([0.1, 0.5, 0.2, 0.9])
    p = online_conformal_p_values(s, score="identity", randomized=False)
    assert np.allclose(p, [1.0, 0.5, 2 / 3, 0.25])
    p2 = online_conformal_p_values(s, score=lambda bag: bag, randomized=False)
    assert np.allclose(p, p2)
    with pytest.raises(ValueError):
        online_conformal_p_values(s, score=lambda bag: bag[:-1])
    with pytest.raises(ValueError):
        online_conformal_p_values(s, score="nope")


def test_knn_and_multivariate():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    p = online_conformal_p_values(X, score="knn", random_state=0)
    assert p.shape == (60,)
    p2 = online_conformal_p_values(X, random_state=0)
    assert p2.shape == (60,)


def test_power_martingale_closed_form():
    p = np.array([0.5, 0.25, 0.8])
    path = conformal_test_martingale(p, "power", epsilon=0.5)
    expected = np.cumsum(np.log10(0.5 * p ** -0.5))
    assert np.allclose(path, expected)


def test_mixture_matches_numeric_integral():
    p = np.array([0.1, 0.3, 0.05])
    path = conformal_test_martingale(p, "mixture", grid_size=20000)
    eps = np.linspace(0, 1, 200001)[1:]
    vals = np.prod(eps[:, None] * p[None, :] ** (eps[:, None] - 1), axis=1)
    assert path[-1] == pytest.approx(np.log10(vals.mean()), rel=1e-3)


def test_jumper_with_zero_jump_and_uniform_p_is_one():
    # f_eps(1/2) = 1 for every eps, so capital stays at 1.
    path = conformal_test_martingale(np.full(10, 0.5), "jumper")
    assert np.allclose(path, 0.0)
    # with jump=0 the jumper is an average of three fixed product martingales
    p = np.array([0.1, 0.9, 0.3])
    path = conformal_test_martingale(p, "jumper", jump=0.0)
    expected = np.mean([np.prod(1 + e * (p - 0.5)) for e in (-1, 0, 1)])
    assert path[-1] == pytest.approx(np.log10(expected))


def test_martingale_expectation_is_one_under_null():
    rng = np.random.default_rng(5)
    finals = [
        10 ** conformal_test_martingale(rng.uniform(size=30), m)[-1]
        for m in ("jumper", "power")
        for _ in range(2000)
    ]
    # (the mixture is also a martingale, but too heavy-tailed for a cheap
    # Monte Carlo mean; it is checked against the integral above instead)
    assert np.mean(finals) == pytest.approx(1.0, abs=0.1)


def test_detects_change_point_and_controls_false_alarms():
    res = exchangeability_test(_shift(), alpha=0.01, random_state=0)
    assert res.rejected
    assert 200 <= res.alarm_index < 260
    false_alarms = sum(
        exchangeability_test(np.random.default_rng(s).normal(size=200), alpha=0.05, random_state=s).rejected
        for s in range(60)
    )
    assert false_alarms <= 8  # Ville: rate <= 5%
    d = res.to_dict()
    assert d["n"] == 400 and d["rejected"] is True


def test_validation():
    with pytest.raises(ValueError):
        conformal_test_martingale([0.5, 1.5])
    with pytest.raises(ValueError):
        conformal_test_martingale([0.5], "bogus")
    with pytest.raises(ValueError):
        conformal_test_martingale([0.5], "power", epsilon=1.0)
    with pytest.raises(ValueError):
        exchangeability_test([1.0, 2.0], alpha=0)
    with pytest.raises(ValueError):
        online_conformal_p_values([])


def test_cli(tmp_path, capsys):
    path = tmp_path / "seq.csv"
    path.write_text("value\n" + "\n".join(f"{v:.6f}" for v in _shift()) + "\n")
    out = tmp_path / "mart.csv"
    assert main(["martingale", "--data", str(path), "--seed", "0", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "REJECTED" in text
    assert out.read_text().splitlines()[0] == "step,value,p_value,log10_martingale"
    assert main(["martingale", "--data", str(path), "--seed", "0", "--json", "--method", "mixture"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "mixture" and payload["n"] == 400
    assert main(["martingale", "--data", str(path), "--seed", "0", "--fail-on-alarm"]) == 1
