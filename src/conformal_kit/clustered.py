"""Clustered conformal prediction (Ding, Fannjiang, Jordan & Tibshirani, NeurIPS 2023).

Split conformal uses one threshold for all labels, which over-covers easy
classes and under-covers hard ones. Mondrian (class-conditional) conformal
fixes that with one threshold per class, but it needs ``ceil(1/alpha) - 1``
calibration points *per class*. With hundreds of classes and a few examples
each, the per-class quantiles are noisy and often infinite.

Clustered conformal prediction sits in between. Classes whose score
distributions look alike are pooled, and each pool gets a threshold:

1. A fraction of the calibration data is held out for clustering. Each class
   with at least ``n_min`` such points is embedded as the vector of its score
   quantiles at levels ``(0.5, 0.6, 0.7, 0.8, 0.9)`` plus ``1 - alpha``.
2. Weighted k-means (weights ``sqrt(n_class)``) groups those embeddings into
   ``n_clusters`` clusters. Rare classes go to a *null* cluster.
3. On the remaining calibration points, each cluster gets the finite-sample
   conformal quantile of its pooled scores. The null cluster, and any
   cluster too small to support ``alpha``, uses the marginal quantile.

Coverage is exact *cluster-conditionally* (and so marginally). When the
clustering groups classes with similar score distributions, class-conditional
coverage is approximately ``1 - alpha``. The class-conditional coverage gap
``CovGap = mean_k |coverage_k - (1 - alpha)|`` (see
:func:`class_coverage_gap`) measures how close it gets.
"""

from __future__ import annotations

import math

import numpy as np

from .classification import SplitConformalClassifier, _aps_score_matrix
from .regression import conformal_quantile

__all__ = [
    "ClusteredConformalClassifier",
    "clustering_parameters",
    "class_coverage_gap",
    "weighted_kmeans",
]

QUANTILE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


def clustering_parameters(n_per_class, n_classes):
    """Default ``(frac_clustering, n_clusters)`` from Ding et al.'s heuristic.

    With ``n`` calibration points in a typical class and ``K`` classes, use
    ``n_clustering = floor(n K / (75 + K))`` points per class for clustering
    and ``floor(n_clustering / 2)`` clusters (at least 1).
    """
    n = float(n_per_class)
    k = int(n_classes)
    if n <= 0 or k < 1:
        raise ValueError("n_per_class and n_classes must be positive")
    n_clustering = int(math.floor(n * k / (75.0 + k)))
    n_clusters = max(1, n_clustering // 2)
    frac = min(max(n_clustering / n, 0.0), 0.9)
    return frac, n_clusters


def weighted_kmeans(points, n_clusters, weights=None, n_init=10, max_iter=300, seed=None):
    """Weighted k-means with k-means++ initialisation (Lloyd iterations).

    Returns ``(labels, centers, inertia)`` for the best of ``n_init`` runs.
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("points must be a non-empty 2-D array")
    n = X.shape[0]
    k = int(n_clusters)
    if not 1 <= k <= n:
        raise ValueError("n_clusters must be between 1 and the number of points")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per point, and not all zero")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, int(n_init))):
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.choice(n, p=w / w.sum())]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            probs = w * d2
            if probs.sum() <= 0:
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=probs / probs.sum())
            centers[c] = X[idx]
            d2 = np.minimum(d2, ((X - centers[c]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=int)
        for it in range(int(max_iter)):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            if it > 0 and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                members = labels == c
                if members.any() and w[members].sum() > 0:
                    centers[c] = np.average(X[members], axis=0, weights=w[members])
        inertia = float((w * ((X - centers[labels]) ** 2).sum(axis=1)).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)
    return best


def class_coverage_gap(y_true, prediction_sets, alpha=0.1):
    """Class-conditional coverage gap ``mean_k |cov_k - (1 - alpha)|`` (in coverage units).

    Also returns per-class coverage and the share of classes under-covered
    by more than 10 percentage points. Classes absent from ``y_true`` are
    skipped.
    """
    labels = np.asarray(y_true).ravel()
    mask = np.asarray(prediction_sets, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != labels.size:
        raise ValueError("prediction_sets must be a (n_samples, n_classes) mask aligned with y_true")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    inside = mask[np.arange(labels.size), labels]
    present = np.unique(labels)
    cov = {int(c): float(inside[labels == c].mean()) for c in present}
    gaps = np.array([abs(v - (1.0 - alpha)) for v in cov.values()])
    under = np.array([v < (1.0 - alpha) - 0.1 for v in cov.values()])
    return {"cov_gap": float(gaps.mean()), "per_class_coverage": cov,
            "frac_undercovered": float(under.mean()), "n_classes": int(present.size)}


class ClusteredConformalClassifier:
    """Clustered conformal prediction sets with cluster-wise thresholds.

    ``fit(y, probabilities)`` uses part of the calibration data
    (``frac_clustering``) to cluster classes by their score quantiles and
    the rest to calibrate one threshold per cluster. ``n_clusters`` and
    ``frac_clustering`` default to :func:`clustering_parameters` evaluated
    at the median class count. ``n_min`` (default ``ceil(1/alpha) - 1``,
    at least 1) is the number of clustering points a class needs to be
    clustered; rarer classes use the marginal threshold. ``method`` is ``"lac"`` or
    ``"aps"`` as for :class:`~conformal_kit.classification.SplitConformalClassifier`.

    Fitted attributes: ``class_to_cluster_`` (``-1`` = null cluster),
    ``cluster_quantiles_``, ``marginal_quantile_`` and ``quantiles_`` (the
    threshold used for each class).
    """

    def __init__(self, alpha=0.1, method="lac", n_clusters=None, frac_clustering=None,
                 n_min=None, seed=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if method not in {"lac", "aps"}:
            raise ValueError("method must be 'lac' or 'aps'")
        if frac_clustering is not None and not 0.0 < frac_clustering < 1.0:
            raise ValueError("frac_clustering must be in (0, 1)")
        if n_clusters is not None and int(n_clusters) < 1:
            raise ValueError("n_clusters must be at least 1")
        self.alpha = float(alpha)
        self.method = method
        self.n_clusters = n_clusters
        self.frac_clustering = frac_clustering
        self.n_min = n_min
        self.seed = seed
        self.quantiles_ = None

    def _scores(self, matrix, labels):
        rows = np.arange(labels.size)
        if self.method == "lac":
            return 1.0 - matrix[rows, labels]
        return SplitConformalClassifier._aps_scores(matrix, labels)

    def fit(self, y_true, probabilities):
        labels = np.asarray(y_true).ravel()
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if labels.size != matrix.shape[0]:
            raise ValueError("y_true and probabilities must have the same length")
        if labels.size == 0:
            raise ValueError("calibration set must not be empty")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y_true must contain integer class indices")
        if labels.min() < 0 or labels.max() >= matrix.shape[1]:
            raise ValueError("y_true contains a class index outside the probability matrix")
        K = matrix.shape[1]
        scores = self._scores(matrix, labels)
        counts = np.bincount(labels, minlength=K)
        observed = counts[counts > 0]
        frac_default, k_default = clustering_parameters(float(np.median(observed)), K)
        frac = self.frac_clustering if self.frac_clustering is not None else frac_default
        n_min = self.n_min if self.n_min is not None else max(int(math.ceil(1.0 / self.alpha)) - 1, 1)
        rng = np.random.default_rng(self.seed)
        # Stratified split: within each class, frac of the points go to clustering.
        is_clust = np.zeros(labels.size, dtype=bool)
        for c in np.flatnonzero(counts):
            idx = np.flatnonzero(labels == c)
            m = int(math.floor(frac * idx.size))
            if m > 0:
                is_clust[rng.choice(idx, size=m, replace=False)] = True
        clust_labels, clust_scores = labels[is_clust], scores[is_clust]
        cal_labels, cal_scores = labels[~is_clust], scores[~is_clust]
        if cal_scores.size == 0:
            raise ValueError("no calibration points left after the clustering split")
        clust_counts = np.bincount(clust_labels, minlength=K) if clust_labels.size else np.zeros(K, int)
        eligible = np.flatnonzero(clust_counts >= n_min)
        levels = sorted(set(QUANTILE_LEVELS) | {1.0 - self.alpha})
        class_to_cluster = np.full(K, -1, dtype=int)
        n_clusters = 0
        if eligible.size >= 1:
            target = int(self.n_clusters) if self.n_clusters is not None else k_default
            n_clusters = min(max(target, 1), eligible.size)
            emb = np.array([np.quantile(clust_scores[clust_labels == c], levels) for c in eligible])
            w = np.sqrt(clust_counts[eligible].astype(float))
            assign, _, _ = weighted_kmeans(emb, n_clusters, weights=w, seed=self.seed)
            class_to_cluster[eligible] = assign
        self.marginal_quantile_ = float(conformal_quantile(cal_scores, self.alpha))
        cluster_q = np.full(n_clusters, self.marginal_quantile_)
        self.cluster_fallback_ = np.zeros(n_clusters, dtype=bool)
        cal_cluster = class_to_cluster[cal_labels]
        for c in range(n_clusters):
            pooled = cal_scores[cal_cluster == c]
            try:
                cluster_q[c] = conformal_quantile(pooled, self.alpha)
            except ValueError:
                self.cluster_fallback_[c] = True
        self.class_to_cluster_ = class_to_cluster
        self.cluster_quantiles_ = cluster_q
        self.quantiles_ = np.where(class_to_cluster >= 0, cluster_q[np.maximum(class_to_cluster, 0)]
                                   if n_clusters else self.marginal_quantile_, self.marginal_quantile_)
        self.n_clusters_ = int(n_clusters)
        self.n_classes_ = int(K)
        self.frac_clustering_ = float(frac)
        self.n_min_ = int(n_min)
        self.n_clustering_points_ = int(is_clust.sum())
        self.n_calibration_ = int(cal_scores.size)
        return self

    def predict_set(self, probabilities):
        if self.quantiles_ is None:
            raise RuntimeError("fit must be called before predict_set")
        matrix = SplitConformalClassifier._checked_probabilities(probabilities)
        if matrix.shape[1] != self.n_classes_:
            raise ValueError(f"expected {self.n_classes_} classes, got {matrix.shape[1]}")
        if self.method == "lac":
            mask = matrix >= 1.0 - self.quantiles_
        else:
            mask = _aps_score_matrix(matrix) <= self.quantiles_
        SplitConformalClassifier._ensure_non_empty(mask, matrix)
        return mask

    predict = predict_set

    @property
    def quantile_(self):
        """The marginal (null-cluster) threshold, for API parity with split conformal."""
        return self.marginal_quantile_

    def predict_labels(self, probabilities):
        mask = self.predict_set(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def set_sizes(self, probabilities):
        return self.predict_set(probabilities).sum(axis=1)
