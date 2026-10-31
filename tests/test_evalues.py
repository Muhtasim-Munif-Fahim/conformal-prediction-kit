"""Tests for conformal e-values and e-BH."""

import numpy as np
import pytest

from conformal_kit import (
    benjamini_hochberg,
    conformal_e_values,
    conformal_p_values,
    derandomized_e_values,
    e_bh,
    e_to_p,
    p_to_e,
)


def _scores(seed, n=300, m=200, n_out=40, shift=3.0):
    rng = np.random.default_rng(seed)
    cal = rng.normal(size=n)
    test = rng.normal(size=m)
    test[:n_out] += shift
    is_out = np.zeros(m, dtype=bool)
    is_out[:n_out] = True
    return cal, test, is_out


def test_e_bh_matches_bh_on_conformal_p_values():
    for seed in range(10):
        cal, test, _ = _scores(seed)
        e, _ = conformal_e_values(cal, test, alpha=0.1)
        p = conformal_p_values(cal, test)
        bh, _ = benjamini_hochberg(p, 0.1)
        np.testing.assert_array_equal(e_bh(e, 0.1), bh)


def test_fixed_threshold_e_values_have_mean_at_most_one():
    rng = np.random.default_rng(0)
    means = []
    for _ in range(20000):
        cal = rng.normal(size=50)
        test = rng.normal(size=1)
        e, _ = conformal_e_values(cal, test, threshold=1.0)
        means.append(e[0])
    # exact expectation is 1 - (1 - q)^(n + 1) ~= 0.9997 for q = P(Z >= 1)
    se = np.std(means) / np.sqrt(len(means))
    assert np.mean(means) <= 1.0 + 3 * se


def test_e_bh_basic_and_validation():
    rej = e_bh([100.0, 1.0, 50.0, 0.0], alpha=0.1)
    assert rej.tolist() == [True, False, True, False]
    assert e_bh([], 0.1).size == 0
    with pytest.raises(ValueError):
        e_bh([-1.0], 0.1)
    with pytest.raises(ValueError):
        e_bh([1.0], 1.5)


def test_derandomized_fdr_and_power():
    fdps, powers = [], []
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        pool = rng.normal(size=600)
        test = rng.normal(size=200)
        test[:40] += 3.5
        runs = []
        for _ in range(5):
            idx = rng.permutation(pool.size)[:300]
            runs.append(conformal_e_values(pool[idx], test, alpha=0.1)[0])
        e = derandomized_e_values(runs)
        rej = e_bh(e, 0.1)
        fdps.append(rej[40:].sum() / max(1, rej.sum()))
        powers.append(rej[:40].mean())
    assert np.mean(fdps) <= 0.15
    assert np.mean(powers) > 0.5


def test_no_threshold_gives_zero_e_values():
    cal = np.arange(10.0)
    e, T = conformal_e_values(cal, [0.0, 1.0], alpha=0.01)
    assert T == float("inf")
    assert np.all(e == 0)


def test_derandomized_weights_and_calibrators():
    E = np.array([[1.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(derandomized_e_values(E), [2.0, 1.0])
    np.testing.assert_allclose(derandomized_e_values(E, weights=[3, 1]), [1.5, 1.5])
    with pytest.raises(ValueError):
        derandomized_e_values(E, weights=[1.0])
    np.testing.assert_allclose(p_to_e([0.25, 1.0], kappa=0.5), [1.0, 0.5])
    np.testing.assert_allclose(e_to_p([0.5, 4.0, 0.0]), [1.0, 0.25, 1.0])
    with pytest.raises(ValueError):
        p_to_e([0.5], kappa=1.0)
