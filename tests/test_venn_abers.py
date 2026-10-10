"""Tests for inductive Venn-Abers predictors."""

import numpy as np
import pytest

from conformal_kit import VennAbersCalibrator, venn_abers_probabilities


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _miscalibrated(n, seed):
    """True P(y=1|x) = sigmoid(3x), but the model reports sigmoid(x)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < _sigmoid(3 * x)).astype(int)
    return _sigmoid(x), y, _sigmoid(3 * x)


def _isotonic_reference(scores, labels, s, label):
    iso = pytest.importorskip("sklearn.isotonic")
    model = iso.IsotonicRegression(out_of_bounds="clip")
    model.fit(np.append(scores, s), np.append(labels, label))
    return float(model.predict([s])[0])


def test_matches_isotonic_regression_reference():
    scores, labels, _ = _miscalibrated(80, seed=0)
    test = np.array([0.05, 0.3, 0.5, scores[3], 0.97])
    p0, p1 = venn_abers_probabilities(scores, labels, test)
    for i, s in enumerate(test):
        assert p0[i] == pytest.approx(_isotonic_reference(scores, labels, s, 0))
        assert p1[i] == pytest.approx(_isotonic_reference(scores, labels, s, 1))


def test_p0_le_p1_and_merge_inside_interval():
    scores, labels, _ = _miscalibrated(300, seed=1)
    test = np.linspace(0.01, 0.99, 40)
    cal = VennAbersCalibrator().fit(scores, labels)
    p0, p1 = cal.predict_interval(test)
    assert np.all(p0 <= p1 + 1e-12)
    assert np.all((p0 >= 0) & (p1 <= 1))
    merged = cal.predict_proba(test)
    assert merged.shape == (40, 2)
    np.testing.assert_allclose(merged.sum(axis=1), 1.0)
    assert np.all(merged[:, 1] >= p0 - 1e-12) and np.all(merged[:, 1] <= p1 + 1e-12)
    # monotone scores -> (weakly) monotone calibrated probabilities
    assert np.all(np.diff(merged[:, 1]) >= -1e-9)


def test_interval_narrows_with_more_calibration_data():
    test = np.linspace(0.1, 0.9, 25)
    widths = []
    for n in (50, 500, 3000):
        scores, labels, _ = _miscalibrated(n, seed=2)
        widths.append(VennAbersCalibrator().fit(scores, labels).interval_width(test).mean())
    assert widths[0] > widths[1] > widths[2]
    assert widths[2] < 0.02


def test_improves_calibration_of_miscalibrated_model():
    scores, labels, _ = _miscalibrated(2000, seed=3)
    test_scores, test_labels, truth = _miscalibrated(2000, seed=4)
    probs = VennAbersCalibrator().fit(scores, labels).predict_proba(test_scores)[:, 1]
    raw_err = np.mean((test_scores - truth) ** 2)
    va_err = np.mean((probs - truth) ** 2)
    assert va_err < 0.3 * raw_err
    brier_raw = np.mean((test_scores - test_labels) ** 2)
    brier_va = np.mean((probs - test_labels) ** 2)
    assert brier_va < brier_raw


def test_binary_matrix_matches_1d():
    scores, labels, _ = _miscalibrated(200, seed=5)
    matrix = np.column_stack([1 - scores, scores])
    test = np.linspace(0.05, 0.95, 10)
    a = VennAbersCalibrator().fit(scores, labels).predict_proba(test)
    b = VennAbersCalibrator().fit(matrix, labels).predict_proba(np.column_stack([1 - test, test]))
    np.testing.assert_allclose(a, b)


def test_multiclass_one_vs_rest():
    rng = np.random.default_rng(6)
    n, k = 600, 3
    logits = rng.normal(0, 1.5, (n, k))
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in true_p])
    reported = np.exp(logits / 3) / np.exp(logits / 3).sum(axis=1, keepdims=True)
    cal = VennAbersCalibrator().fit(reported[:400], y[:400])
    probs = cal.predict_proba(reported[400:])
    assert probs.shape == (200, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    p0, p1 = cal.predict_interval(reported[400:])
    assert p0.shape == (200, 3) and np.all(p0 <= p1 + 1e-12)
    assert cal.interval_width(reported[400:]).shape == (200,)
    nll_raw = -np.mean(np.log(reported[400:][np.arange(200), y[400:]]))
    nll_va = -np.mean(np.log(np.clip(probs[np.arange(200), y[400:]], 1e-12, 1)))
    assert nll_va < nll_raw


def test_ties_with_calibration_scores():
    scores = np.array([0.2, 0.2, 0.2, 0.8, 0.8])
    labels = np.array([0, 1, 0, 1, 1])
    p0, p1 = venn_abers_probabilities(scores, labels, [0.2, 0.8])
    assert p0[0] == pytest.approx(1 / 4) and p1[0] == pytest.approx(2 / 4)
    assert p0[1] == pytest.approx(2 / 3) and p1[1] == pytest.approx(1.0)


def test_errors():
    with pytest.raises(ValueError):
        venn_abers_probabilities([], [], [0.5])
    with pytest.raises(ValueError):
        venn_abers_probabilities([0.1, 0.2], [0, 2], [0.5])
    with pytest.raises(ValueError):
        venn_abers_probabilities([0.1, 0.2], [0], [0.5])
    with pytest.raises(ValueError):
        venn_abers_probabilities([0.1, np.nan], [0, 1], [0.5])
    with pytest.raises(RuntimeError):
        VennAbersCalibrator().predict_proba([0.5])
    cal = VennAbersCalibrator().fit([0.1, 0.9], [0, 1])
    with pytest.raises(ValueError):
        cal.predict_proba(np.ones((2, 2)))
    with pytest.raises(ValueError):
        VennAbersCalibrator().fit(np.ones((3, 3)), np.array([0, 1, 3]))
    with pytest.raises(ValueError):
        VennAbersCalibrator().fit(np.ones((3, 1)), np.array([0, 0, 0]))
