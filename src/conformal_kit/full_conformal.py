"""Exact full conformal prediction for ridge regression (RRCM).

Split conformal sacrifices a calibration set. *Full* (transductive) conformal
prediction uses every labelled point for both fitting and calibration: for a
test input ``x`` and a candidate label ``z`` it refits the model on the
augmented data ``(X, y) + (x, z)``, scores every point by its absolute
residual, and keeps ``z`` when the test point's residual is not among the
``alpha`` most extreme. Done naively that is one refit per candidate label.

For ridge regression the refit is linear in ``z``: with the augmented design
``Xa`` and hat matrix ``H = Xa (Xa'Xa + P)^{-1} Xa'``, the residual vector is
``(I - H)(y, z) = a + b z``. Each training point ``i`` is "at least as
strange" as the test point exactly when ``|a_i + b_i z| >= |a_t + b_t z|``,
a condition whose solution set in ``z`` is bounded by at most two roots.
Sweeping the sorted roots gives the exact prediction set -- the Ridge
Regression Confidence Machine of Nouretdinov, Melluish & Vovk (ICML 2001);
see also Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*
(2005), Sec. 2.3.

The set ``{z : p(z) > alpha}`` with ``p(z) = #{i : |r_i(z)| >= |r_t(z)|} / (n + 1)``
(the test point counts itself) has finite-sample coverage ``>= 1 - alpha``
under exchangeability. It is usually a single interval; the rare
disconnected case is available from :meth:`FullConformalRidgeRegressor.predict_set`.
``Xa'Xa + P`` is updated from ``X'X + P`` with Sherman-Morrison, so each test
point costs ``O(n p)`` after an ``O(n p^2)`` fit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FullConformalRidgeRegressor", "ridge_conformal_set"]


def _design(X, fit_intercept):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a 1-D or 2-D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    if fit_intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    return X


def _union_from_counts(a, b, alpha):
    """Exact ``{z : p(z) > alpha}`` from residual coefficients ``a + b z``.

    ``a`` and ``b`` have length ``n + 1`` with the test point last. ``p(z)``
    is piecewise constant between the roots, so probing every root and one
    point per gap is exact. Returns the sorted, closed ``(low, high)``
    intervals of the set's closure (ends may be infinite).
    """
    n1 = a.size
    at, bt = a[-1], b[-1]
    ai, bi = a[:-1], b[:-1]
    roots = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for num, den in ((at - ai, bi - bt), (-(ai + at), bi + bt)):
            r = num / den
            roots.append(r[np.isfinite(r)])
    pts = np.unique(np.concatenate(roots)) if roots else np.empty(0)

    def p_value(z):
        z = np.atleast_1d(z)
        rt = np.abs(at + bt * z)
        ri = np.abs(ai[:, None] + bi[:, None] * z[None, :])
        # small relative tolerance so ties at the roots count as ">="
        tol = 1e-12 * (1.0 + np.maximum(ri, rt[None, :]))
        return (1 + np.sum(ri >= rt[None, :] - tol, axis=0)) / n1

    if pts.size == 0:
        keep = p_value(np.array([0.0]))[0] > alpha
        return [(-np.inf, np.inf)] if keep else []
    # probe each root and one point in each gap (including the two rays)
    span = max(1.0, float(np.max(np.abs(pts))))
    mids = np.concatenate([[pts[0] - span], (pts[:-1] + pts[1:]) / 2.0, [pts[-1] + span]])
    p_pts = p_value(pts) > alpha
    p_mid = p_value(mids) > alpha
    # walk ray, root, gap, root, ..., root, ray; merge consecutive kept pieces
    pieces = [(-np.inf, pts[0], p_mid[0])]
    for k in range(pts.size):
        pieces.append((pts[k], pts[k], p_pts[k]))
        right = pts[k + 1] if k + 1 < pts.size else np.inf
        pieces.append((pts[k], right, p_mid[k + 1]))
    intervals = []
    run_start = None
    for low, high, kept in pieces:
        if kept and run_start is None:
            run_start = low
        if kept:
            run_end = high
        elif run_start is not None:
            intervals.append((float(run_start), float(run_end)))
            run_start = None
    if run_start is not None:
        intervals.append((float(run_start), float(run_end)))
    return intervals


def ridge_conformal_set(X_train, y_train, x_test, alpha=0.1, ridge=1.0, fit_intercept=True):
    """Exact full-conformal ridge prediction set for one test input.

    Convenience wrapper around :class:`FullConformalRidgeRegressor`; returns
    the list of ``(low, high)`` intervals.
    """
    model = FullConformalRidgeRegressor(alpha=alpha, ridge=ridge, fit_intercept=fit_intercept)
    return model.fit(X_train, y_train).predict_set(np.atleast_2d(x_test))[0]


class FullConformalRidgeRegressor:
    """Ridge Regression Confidence Machine: exact full conformal intervals.

    Parameters
    ----------
    alpha:
        Target miscoverage; intervals cover with probability ``>= 1 - alpha``.
    ridge:
        L2 penalty ``> 0`` on the coefficients (the intercept is not
        penalized).
    fit_intercept:
        Add an unpenalized intercept column.
    """

    def __init__(self, alpha=0.1, ridge=1.0, fit_intercept=True):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if not ridge > 0:
            raise ValueError("ridge must be positive")
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.fit_intercept = bool(fit_intercept)

    def fit(self, X, y):
        """Store the training data and the inverse ``(X'X + P)^{-1}``."""
        Xd = _design(X, self.fit_intercept)
        y = np.asarray(y, dtype=float).ravel()
        if y.size != Xd.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if y.size < 1:
            raise ValueError("at least one training point is required")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        if (y.size + 1) * self.alpha < 1.0:
            # p-values are multiples of 1/(n+1) >= 1/(n+1) > alpha: every z is kept
            self.trivial_ = True
        else:
            self.trivial_ = False
        penalty = np.full(Xd.shape[1], self.ridge)
        if self.fit_intercept:
            penalty[-1] = 0.0
        gram = Xd.T @ Xd + np.diag(penalty)
        try:
            self.gram_inv_ = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            raise ValueError("X'X + ridge*I is singular; increase ridge") from None
        self.X_ = Xd
        self.y_ = y
        self.Xty_ = Xd.T @ y
        self.coef_ = self.gram_inv_ @ self.Xty_
        self.n_features_in_ = Xd.shape[1] - int(self.fit_intercept)
        return self

    def _check_fitted(self):
        if not hasattr(self, "X_"):
            raise RuntimeError("FullConformalRidgeRegressor is not fitted; call fit() first")

    def residual_coefficients(self, x):
        """``(a, b)`` with augmented residuals ``a + b z`` (test point last)."""
        self._check_fitted()
        xt = _design(np.atleast_2d(x), self.fit_intercept)[0]
        if xt.size != self.X_.shape[1]:
            raise ValueError("test input has the wrong number of features")
        G = self.gram_inv_
        Gx = G @ xt
        M_inv = G - np.outer(Gx, Gx) / (1.0 + xt @ Gx)  # Sherman-Morrison
        Xa = np.vstack([self.X_, xt])
        a = np.concatenate([self.y_, [0.0]]) - Xa @ (M_inv @ self.Xty_)
        b = -Xa @ (M_inv @ xt)
        b[-1] += 1.0
        return a, b

    def p_value(self, x, z):
        """Full-conformal p-value of candidate label(s) ``z`` at input ``x``."""
        a, b = self.residual_coefficients(x)
        z = np.atleast_1d(np.asarray(z, dtype=float))
        rt = np.abs(a[-1] + b[-1] * z)
        ri = np.abs(a[:-1, None] + b[:-1, None] * z[None, :])
        tol = 1e-12 * (1.0 + np.maximum(ri, rt[None, :]))
        return (1 + np.sum(ri >= rt[None, :] - tol, axis=0)) / a.size

    def predict(self, X):
        """Ridge point predictions from the training fit."""
        self._check_fitted()
        return _design(X, self.fit_intercept) @ self.coef_

    def predict_set(self, X):
        """Exact prediction set per row: a list of ``(low, high)`` intervals."""
        self._check_fitted()
        Xd = np.asarray(X, dtype=float)
        if Xd.ndim == 1:
            Xd = Xd[:, None] if self.n_features_in_ == 1 else Xd[None, :]
        out = []
        for row in Xd:
            if self.trivial_:
                out.append([(-np.inf, np.inf)])
                continue
            a, b = self.residual_coefficients(row)
            out.append(_union_from_counts(a, b, self.alpha))
        return out

    def predict_interval(self, X):
        """Lower / upper bounds (convex hull of each prediction set).

        Returns ``(lower, upper)`` arrays; an empty set gives ``NaN`` bounds
        and an unbounded side gives ``-inf`` / ``inf``.
        """
        sets = self.predict_set(X)
        lower = np.array([s[0][0] if s else np.nan for s in sets])
        upper = np.array([s[-1][1] if s else np.nan for s in sets])
        return lower, upper
