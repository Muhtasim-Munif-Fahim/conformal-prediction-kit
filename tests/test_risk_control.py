"""Tests for conformal risk control."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit import (
    MultilabelRiskController,
    conformal_quantile,
    conformal_risk_threshold,
    false_negative_rate,
)


def _multilabel(n=6000, k=8, seed=0):
    """Calibrated independent-sigmoid multilabel data."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(-0.5, 1.5, size=(n, k))
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random((n, k)) < probabilities).astype(int)
    return labels, probabilities


def test_fnr_helper():
    y = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1]])
    sets = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 1]])
    # row FNRs: 0.5, 0 (no positives), 0
    assert false_negative_rate(y, sets) == pytest.approx(0.5 / 3)
    with pytest.raises(ValueError):
        false_negative_rate(y, sets[:, :2])


def test_threshold_rule_hand_computed():
    losses = np.array([[1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    lambdas = [0.0, 0.5, 1.0]
    # corrected risks: (3*1+1)/4=1, (3*0.5+1)/4=0.625, (0+1)/4=0.25
    assert conformal_risk_threshold(losses, lambdas, alpha=0.7) == 0.5
    assert conformal_risk_threshold(losses, lambdas, alpha=0.3) == 1.0
    with pytest.raises(ValueError, match="no lambda"):
        conformal_risk_threshold(losses, lambdas, alpha=0.2)


def test_miscoverage_loss_recovers_split_conformal():
    rng = np.random.default_rng(3)
    scores = rng.exponential(size=199)
    lambdas = np.sort(scores)
    losses = (scores[:, None] > lambdas[None, :]).astype(float)
    assert conformal_risk_threshold(losses, lambdas, alpha=0.1) == pytest.approx(
        conformal_quantile(scores, 0.1)
    )


def test_threshold_validation():
    with pytest.raises(ValueError, match="increasing"):
        conformal_risk_threshold([[1.0, 0.0]], [1.0, 0.0], 0.5)
    with pytest.raises(ValueError, match="non-increasing"):
        conformal_risk_threshold([[0.0, 1.0]], [0.0, 1.0], 0.5)
    with pytest.raises(ValueError, match="loss_bound"):
        conformal_risk_threshold([[2.0, 0.0]], [0.0, 1.0], 0.5)
    with pytest.raises(ValueError, match="shape"):
        conformal_risk_threshold([[1.0, 0.0]], [0.0, 0.5, 1.0], 0.5)
    with pytest.raises(ValueError, match="alpha"):
        conformal_risk_threshold([[1.0, 0.0]], [0.0, 1.0], 1.5)


def test_controller_matches_generic_rule_on_grid():
    y, p = _multilabel(n=300, k=5, seed=1)
    ctrl = MultilabelRiskController(alpha=0.15).fit(y, p)
    rows, cols = np.nonzero(y)
    lambdas = np.unique(np.concatenate([[0.0], 1.0 - p[rows, cols]]))
    losses = np.empty((y.shape[0], lambdas.size))
    for j, lam in enumerate(lambdas):
        sets = p >= 1.0 - lam - 1e-12
        pos = y.sum(axis=1)
        hits = (y.astype(bool) & sets).sum(axis=1)
        losses[:, j] = np.where(pos > 0, 1.0 - hits / np.maximum(pos, 1), 0.0)
    assert ctrl.lambda_ == pytest.approx(conformal_risk_threshold(losses, lambdas, 0.15))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_controller_bounds_test_fnr(alpha):
    y, p = _multilabel(n=12000, seed=2)
    half = len(y) // 2
    ctrl = MultilabelRiskController(alpha=alpha).fit(y[:half], p[:half])
    fnr = false_negative_rate(y[half:], ctrl.predict_set(p[half:]))
    assert fnr <= alpha + 0.02
    assert fnr >= alpha - 0.03  # not wildly conservative
    assert ctrl.calibration_risk_ <= alpha
    assert 0.0 <= ctrl.threshold_ <= 1.0


def test_smaller_alpha_gives_larger_sets():
    y, p = _multilabel(n=3000, seed=4)
    loose = MultilabelRiskController(alpha=0.3).fit(y, p)
    tight = MultilabelRiskController(alpha=0.05).fit(y, p)
    assert tight.set_sizes(p).mean() > loose.set_sizes(p).mean()
    labels = tight.predict_labels(p[:3])
    assert len(labels) == 3 and all(isinstance(row, np.ndarray) for row in labels)


def test_controller_validation():
    y, p = _multilabel(n=50, k=3)
    with pytest.raises(ValueError, match="alpha"):
        MultilabelRiskController(alpha=0.0)
    with pytest.raises(RuntimeError, match="fit"):
        MultilabelRiskController().predict_set(p)
    with pytest.raises(ValueError, match="same shape"):
        MultilabelRiskController().fit(y[:, :2], p)
    with pytest.raises(ValueError, match="0/1"):
        MultilabelRiskController().fit(y * 2, p)
    with pytest.raises(ValueError, match="calibration rows"):
        MultilabelRiskController(alpha=0.01).fit(y[:5], p[:5])
    ctrl = MultilabelRiskController(alpha=0.2).fit(y, p)
    with pytest.raises(ValueError, match="label columns"):
        ctrl.predict_set(p[:, :2])
