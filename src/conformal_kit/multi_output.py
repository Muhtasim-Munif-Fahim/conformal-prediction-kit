"""Joint split-conformal prediction regions for multi-output regression.

When a model predicts ``d`` targets at once (the x/y of a location, several
sensor readings, a multi-horizon forecast), running ``d`` separate conformal
intervals at level ``1 - alpha`` only covers each target marginally. The
chance that *all* targets are covered at once can be as low as
``1 - d * alpha``. :class:`MultiOutputConformalRegressor` calibrates one
**joint** region instead, with ``P(y in region) >= 1 - alpha``:

* ``method="bonferroni"`` -- per-target intervals at level ``alpha / d``.
  Valid under any dependence (union bound) but conservative, especially when
  targets are correlated.
* ``method="max"`` -- a hyper-rectangle from the single score
  ``max_j |r_j| / s_j`` (Messoudi et al., 2021, "Copula-based conformal
  prediction for multi-target regression", max-norm variant). The rectangle
  is exact, not conservative.
* ``method="mahalanobis"`` -- an ellipsoid from ``sqrt(r' S^-1 r)``
  (Johnstone & Cox, 2021, "Conformal uncertainty sets for robust
  optimization"). It follows the correlation between targets, which can
  shrink the region volume a lot.

The scales ``s_j`` and the covariance ``S`` are estimated on a fraction
``scale_fraction`` of the calibration residuals, and the conformal quantile
is taken on the *remaining* points. Because the scaling never sees those
points, the finite-sample guarantee holds exactly. With
``scale_fraction=0`` the scales are 1 and ``S`` is the identity.
"""

from __future__ import annotations

import math

import numpy as np

from .regression import conformal_quantile

__all__ = ["MultiOutputConformalRegressor", "joint_coverage"]

_METHODS = ("bonferroni", "max", "mahalanobis")


def _as_2d(a, name):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def joint_coverage(y_true, lower, upper):
    """Fraction of rows where every target lies inside its interval."""
    y = _as_2d(y_true, "y_true")
    lo, hi = _as_2d(lower, "lower"), _as_2d(upper, "upper")
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same shape")
    return float(np.mean(np.all((y >= lo) & (y <= hi), axis=1)))


class MultiOutputConformalRegressor:
    """Joint ``1 - alpha`` prediction regions for ``d``-dimensional targets."""

    def __init__(self, alpha=0.1, method="max", scale_fraction=0.3, seed=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if not 0.0 <= scale_fraction < 1.0:
            raise ValueError("scale_fraction must be in [0, 1)")
        self.alpha = float(alpha)
        self.method = method
        self.scale_fraction = float(scale_fraction)
        self.seed = seed
        self.quantile_ = None
        self.quantiles_ = None
        self.scale_ = None
        self.precision_ = None
        self.n_targets_ = 0
        self.n_calibration_ = 0

    def fit(self, y_true, y_pred):
        y = _as_2d(y_true, "y_true")
        p = _as_2d(y_pred, "y_pred")
        if y.shape != p.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        r = y - p
        n, d = r.shape
        self.n_targets_ = d
        if self.method == "bonferroni":
            a = self.alpha / d
            self.quantiles_ = np.array([conformal_quantile(np.abs(r[:, j]), a) for j in range(d)])
            self.n_calibration_ = n
            return self
        n_scale = int(math.floor(self.scale_fraction * n))
        if self.scale_fraction > 0 and n_scale < d + 1:
            raise ValueError("too few calibration points for the scale split")
        idx = np.random.default_rng(self.seed).permutation(n)
        fit_part, cal_part = r[idx[:n_scale]], r[idx[n_scale:]]
        if self.method == "max":
            scale = fit_part.std(axis=0, ddof=1) if n_scale else np.ones(d)
            if np.any(scale <= 0):
                raise ValueError("a target has zero residual spread on the scale split")
            self.scale_ = scale
            scores = np.max(np.abs(cal_part) / scale, axis=1)
        else:
            if n_scale:
                cov = np.atleast_2d(np.cov(fit_part, rowvar=False))
                cov = cov + 1e-12 * np.trace(cov) / d * np.eye(d)
            else:
                cov = np.eye(d)
            self.precision_ = np.linalg.inv(cov)
            scores = np.sqrt(np.einsum("ij,jk,ik->i", cal_part, self.precision_, cal_part))
        self.quantile_ = conformal_quantile(scores, self.alpha)
        self.n_calibration_ = int(cal_part.shape[0])
        return self

    def _check_fitted(self):
        if self.quantile_ is None and self.quantiles_ is None:
            raise RuntimeError("fit must be called first")

    def _pred(self, y_pred):
        p = _as_2d(y_pred, "y_pred")
        if p.shape[1] != self.n_targets_:
            raise ValueError(f"expected {self.n_targets_} targets, got {p.shape[1]}")
        return p

    def half_widths(self):
        """Per-target half-widths of the region's bounding box."""
        self._check_fitted()
        if self.method == "bonferroni":
            return self.quantiles_.copy()
        if self.method == "max":
            return self.quantile_ * self.scale_
        cov = np.linalg.inv(self.precision_)
        return self.quantile_ * np.sqrt(np.diag(cov))

    def predict_interval(self, y_pred):
        """``(lower, upper)`` boxes. Exact for rectangles; bounding box for ellipsoids."""
        self._check_fitted()
        p = self._pred(y_pred)
        h = self.half_widths()
        return p - h, p + h

    def contains(self, y_true, y_pred):
        """Boolean array: is each ``y_true`` row inside its joint region?"""
        self._check_fitted()
        p = self._pred(y_pred)
        y = _as_2d(y_true, "y_true")
        if y.shape != p.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        r = y - p
        if self.method == "mahalanobis":
            s = np.sqrt(np.einsum("ij,jk,ik->i", r, self.precision_, r))
            return s <= self.quantile_ + 1e-12
        return np.all(np.abs(r) <= self.half_widths() + 1e-12, axis=1)

    def region_volume(self):
        """Lebesgue volume of one region (the same for every test point)."""
        self._check_fitted()
        d = self.n_targets_
        if self.method != "mahalanobis":
            return float(np.prod(2.0 * self.half_widths()))
        unit_ball = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
        det_cov = 1.0 / np.linalg.det(self.precision_)
        return float(unit_ball * self.quantile_ ** d * math.sqrt(det_cov))
