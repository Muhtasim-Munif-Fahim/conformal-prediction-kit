"""Tests for clustered conformal prediction."""

import json

import numpy as np
import pytest

from conformal_kit import (
    ClusteredConformalClassifier,
    MondrianConformalClassifier,
    SplitConformalClassifier,
    class_coverage_gap,
    clustering_parameters,
    weighted_kmeans,
)
from conformal_kit.cli import main


def _problem(rng, n, n_classes=60, hard=None):
    if hard is None:
        hard = np.arange(n_classes) % 2 == 0
    y = rng.integers(0, n_classes, n)
    logits = rng.normal(0.0, 1.0, (n, n_classes))
    logits[np.arange(n), y] += np.where(hard[y], 1.0, 4.0)
    p = np.exp(logits)
    return y, p / p.sum(axis=1, keepdims=True)


def test_clustering_parameters_heuristic():
    frac, k = clustering_parameters(50, 100)
    n_clust = int(50 * 100 / 175)
    assert k == n_clust // 2 and frac == pytest.approx(n_clust / 50)
    assert clustering_parameters(2, 10) == (0.0, 1)
    with pytest.raises(ValueError):
        clustering_parameters(0, 10)


def test_weighted_kmeans_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2)), rng.normal([0, 5], 0.1, (20, 2))])
    labels, centers, inertia = weighted_kmeans(pts, 3, seed=1)
    for block in range(3):
        assert len(set(labels[block * 20:(block + 1) * 20])) == 1
    assert len(set(labels)) == 3 and inertia < 5
    # Weights pull a centre towards the heavy point.
    _, c1, _ = weighted_kmeans(np.array([[0.0], [1.0]]), 1, weights=[3.0, 1.0])
    assert c1[0, 0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        weighted_kmeans(pts, 0)
    with pytest.raises(ValueError):
        weighted_kmeans(pts, 2, weights=np.zeros(60))


def test_clusters_group_classes_by_difficulty():
    rng = np.random.default_rng(1)
    y, p = _problem(rng, 60 * 60)
    cc = ClusteredConformalClassifier(alpha=0.1, method="lac", n_clusters=2, seed=0).fit(y, p)
    hard = np.arange(60) % 2 == 0
    assign = cc.class_to_cluster_
    assert np.all(assign >= 0)
    assert len(set(assign[hard])) == 1 and len(set(assign[~hard])) == 1
    assert assign[hard][0] != assign[~hard][0]
    # Hard classes need a larger LAC score threshold.
    assert cc.cluster_quantiles_[assign[hard][0]] > cc.cluster_quantiles_[assign[~hard][0]]


def test_marginal_coverage_is_valid_on_average():
    rng = np.random.default_rng(2)
    covs = []
    for _ in range(15):
        y, p = _problem(rng, 1500, n_classes=30)
        yt, pt = _problem(rng, 3000, n_classes=30)
        sets = ClusteredConformalClassifier(alpha=0.1, method="aps", seed=0).fit(y, p).predict_set(pt)
        covs.append(sets[np.arange(yt.size), yt].mean())
    assert np.mean(covs) >= 0.89


def test_beats_split_on_class_conditional_gap():
    rng = np.random.default_rng(3)
    y, p = _problem(rng, 60 * 40)
    yt, pt = _problem(rng, 20000)
    split = SplitConformalClassifier(0.1, "lac").fit(y, p).predict_set(pt)
    clus = ClusteredConformalClassifier(0.1, "lac", seed=0).fit(y, p).predict_set(pt)
    mond = MondrianConformalClassifier(0.1, "lac").fit(y, p).predict_set(pt)
    g_split = class_coverage_gap(yt, split, 0.1)["cov_gap"]
    g_clus = class_coverage_gap(yt, clus, 0.1)["cov_gap"]
    assert g_clus < 0.6 * g_split
    assert clus.sum(axis=1).mean() <= mond.sum(axis=1).mean() * 1.05


def test_rare_classes_use_marginal_threshold():
    rng = np.random.default_rng(4)
    y, p = _problem(rng, 2000, n_classes=10)
    y[y == 9] = 8  # make class 9 rare: only three examples
    y[:3] = 9
    cc = ClusteredConformalClassifier(alpha=0.1, method="lac", seed=0).fit(y, p)
    assert cc.class_to_cluster_[9] == -1
    assert cc.quantiles_[9] == cc.marginal_quantile_
    assert 0 < cc.marginal_quantile_ <= 1
    assert cc.n_clustering_points_ + cc.n_calibration_ == y.size
    assert cc.quantile_ == cc.marginal_quantile_


def test_all_null_reduces_to_split_quantile():
    rng = np.random.default_rng(5)
    y, p = _problem(rng, 400, n_classes=40)  # ~10 per class, too few to cluster
    cc = ClusteredConformalClassifier(alpha=0.1, method="lac", frac_clustering=0.3, n_min=50, seed=0).fit(y, p)
    assert cc.n_clusters_ == 0
    assert np.all(cc.class_to_cluster_ == -1)
    assert np.allclose(cc.quantiles_, cc.marginal_quantile_)


def test_class_coverage_gap_and_validation():
    y = np.array([0, 0, 1, 1])
    sets = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=bool)
    g = class_coverage_gap(y, sets, alpha=0.1)
    assert g["per_class_coverage"] == {0: 1.0, 1: 0.5}
    assert g["cov_gap"] == pytest.approx((0.1 + 0.4) / 2)
    assert g["frac_undercovered"] == 0.5
    with pytest.raises(ValueError):
        class_coverage_gap(y, sets[:2], 0.1)
    with pytest.raises(ValueError):
        ClusteredConformalClassifier(alpha=1.5)
    with pytest.raises(ValueError):
        ClusteredConformalClassifier(method="raps")
    with pytest.raises(ValueError):
        ClusteredConformalClassifier(frac_clustering=1.0)
    with pytest.raises(RuntimeError):
        ClusteredConformalClassifier().predict_set(np.full((1, 2), 0.5))
    cc = ClusteredConformalClassifier(seed=0).fit(*_problem(np.random.default_rng(6), 600, n_classes=5))
    with pytest.raises(ValueError):
        cc.predict_set(np.full((1, 3), 1 / 3))
    assert len(cc.predict_labels(np.full((2, 5), 0.2))) == 2


def test_cli_classify_clustered(tmp_path, capsys):
    rng = np.random.default_rng(7)
    y, p = _problem(rng, 1200, n_classes=12)
    yt, pt = _problem(rng, 600, n_classes=12)
    cal = tmp_path / "cal.csv"
    test = tmp_path / "test.csv"
    header = ",".join(f"p{i}" for i in range(12))
    cal.write_text("y_true," + header + "\n" + "\n".join(
        str(a) + "," + ",".join(f"{v:.6f}" for v in row) for a, row in zip(y, p)) + "\n")
    test.write_text("y_true," + header + "\n" + "\n".join(
        str(a) + "," + ",".join(f"{v:.6f}" for v in row) for a, row in zip(yt, pt)) + "\n")
    code = main(["classify", "--calibration", str(cal), "--test", str(test), "--method", "clustered", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code in (0, 1)
    assert out["method"] == "clustered" and out["n_clusters"] >= 1
    assert "class_coverage_gap" in out
