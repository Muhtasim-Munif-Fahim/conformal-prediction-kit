"""Learn then Test: high-probability risk control (Angelopoulos, Bates, Candès,
Jordan & Lei, 2021) and RCPS (Bates, Angelopoulos, Lei, Malik & Jordan, 2021).

Conformal risk control (:mod:`conformal_kit.risk_control`) bounds the
*expected* test loss. That is an average over calibration draws, and roughly
half of all calibration sets can still give a threshold whose true risk is
above ``alpha``. Learn then Test (LTT) gives the stronger guarantee

    P( R(lambda_hat) <= alpha ) >= 1 - delta

over the draw of the calibration set. It treats every candidate ``lambda``
as a hypothesis test of ``H_j : R(lambda_j) > alpha``. A valid p-value for
each null comes from a concentration inequality on the empirical risk. A
family-wise error rate (FWER) procedure then returns the set of
``lambda`` values whose null is rejected. Any ``lambda`` from that set
controls the risk with probability at least ``1 - delta``, so the loss does
**not** need to be monotone in ``lambda``. That is the main advantage over
conformal risk control and RCPS.

p-values (losses rescaled to ``[0, 1]``):

* ``"hoeffding"``: ``exp(-n * h(min(R_hat, alpha), alpha))`` with
  ``h(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b))`` (the tight KL form).
* ``"bentkus"``: ``e * P(Binomial(n, alpha) <= ceil(n R_hat))``.
* ``"hb"`` (default): the minimum of the two, the Hoeffding-Bentkus p-value
  recommended by the LTT paper.
* ``"binomial"``: the exact ``P(Binomial(n, alpha) <= n R_hat)``. It is valid
  only for 0/1 losses (miscoverage, top-k error, ...).

FWER procedures:

* ``"bonferroni"``: reject when ``p_j <= delta / m``.
* ``"holm"``: step-down Bonferroni. It always rejects at least as many
  hypotheses as Bonferroni.
* ``"fixed_sequence"``: test in a fixed order at level ``delta`` each and stop
  at the first failure. With a loss that is monotone in ``lambda``, testing
  from the safest ``lambda`` down gives RCPS-style control without paying for
  multiplicity. This is the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .risk_control import _checked_alpha, _checked_multilabel

__all__ = [
    "LTTMultilabelController",
    "LTTResult",
    "bentkus_p_value",
    "binomial_p_value",
    "fwer_reject",
    "hoeffding_bentkus_p_value",
    "hoeffding_p_value",
    "learn_then_test",
    "ltt_p_values",
]

_P_VALUES = ("hb", "hoeffding", "bentkus", "binomial")
_FWER = ("fixed_sequence", "bonferroni", "holm")


def _checked_delta(delta):
    try:
        value = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be strictly between 0 and 1") from None
    if not 0.0 < value < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    return value


def _checked_n(n):
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError("n must be a positive integer")
    return int(n)


def _checked_risk(risk_hat):
    r = np.asarray(risk_hat, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("risk_hat must lie in [0, 1] (rescale losses by their bound)")
    return r


def _binom_cdf(k, n, p):
    """``P(Binomial(n, p) <= k)`` for integer ``k`` (log-space, numpy only)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    i = np.arange(k + 1, dtype=float)
    log_terms = (
        math.lgamma(n + 1)
        - np.array([math.lgamma(v + 1) for v in i])
        - np.array([math.lgamma(n - v + 1) for v in i])
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
    )
    top = log_terms.max()
    return float(min(1.0, math.exp(top) * np.exp(log_terms - top).sum()))


def hoeffding_p_value(risk_hat, n, alpha):
    """Hoeffding (KL form) p-value for ``H: R > alpha`` given empirical risk ``risk_hat``."""
    alpha = _checked_alpha(alpha)
    n = _checked_n(n)
    r = np.minimum(_checked_risk(risk_hat), alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(r > 0.0, r * np.log(np.where(r > 0, r, 1.0) / alpha), 0.0)
        second = np.where(
            r < 1.0, (1.0 - r) * np.log(np.where(r < 1, 1.0 - r, 1.0) / (1.0 - alpha)), 0.0
        )
    p = np.exp(-n * (first + second))
    return float(p) if p.ndim == 0 else p


def bentkus_p_value(risk_hat, n, alpha):
    """Bentkus p-value ``e * P(Binomial(n, alpha) <= ceil(n * risk_hat))``, capped at 1."""
    alpha = _checked_alpha(alpha)
    n = _checked_n(n)
    r = _checked_risk(risk_hat)
    flat = np.atleast_1d(r).ravel()
    out = np.array(
        [min(1.0, math.e * _binom_cdf(math.ceil(n * v - 1e-9), n, alpha)) for v in flat]
    )
    return float(out[0]) if r.ndim == 0 else out.reshape(r.shape)


def hoeffding_bentkus_p_value(risk_hat, n, alpha):
    """Hoeffding-Bentkus p-value: the minimum of the two (both are valid)."""
    return np.minimum(hoeffding_p_value(risk_hat, n, alpha), bentkus_p_value(risk_hat, n, alpha))


def binomial_p_value(risk_hat, n, alpha):
    """Exact binomial p-value ``P(Binomial(n, alpha) <= n * risk_hat)``.

    Valid only when every loss is 0 or 1, so ``n * risk_hat`` is an integer count.
    """
    alpha = _checked_alpha(alpha)
    n = _checked_n(n)
    r = _checked_risk(risk_hat)
    flat = np.atleast_1d(r).ravel()
    counts = np.rint(flat * n)
    if np.any(np.abs(counts - flat * n) > 1e-6):
        raise ValueError("binomial p-values need 0/1 losses (n * risk_hat must be an integer)")
    out = np.array([_binom_cdf(int(c), n, alpha) for c in counts])
    return float(out[0]) if r.ndim == 0 else out.reshape(r.shape)


_P_FUNCS = {
    "hb": hoeffding_bentkus_p_value,
    "hoeffding": hoeffding_p_value,
    "bentkus": bentkus_p_value,
    "binomial": binomial_p_value,
}


def ltt_p_values(losses, alpha, p_value="hb", loss_bound=1.0):
    """Per-candidate empirical risks and p-values from an ``(n, m)`` loss matrix.

    Returns ``(risk_hat, p_values)``, with ``risk_hat`` on the original loss scale.
    """
    if p_value not in _P_FUNCS:
        raise ValueError(f"p_value must be one of {_P_VALUES}")
    alpha = _checked_alpha(alpha) if loss_bound == 1.0 else float(alpha)
    bound = float(loss_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("loss_bound must be a positive finite number")
    if not 0.0 < alpha < bound:
        raise ValueError("alpha must be strictly between 0 and loss_bound")
    matrix = np.asarray(losses, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("losses must be a non-empty 2D array (n_calibration, n_lambdas)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("losses must all be finite")
    if np.any(matrix < 0.0) or np.any(matrix > bound + 1e-12):
        raise ValueError("losses must lie in [0, loss_bound]")
    n = matrix.shape[0]
    risk = matrix.mean(axis=0)
    scaled = np.clip(risk / bound, 0.0, 1.0)
    p = np.atleast_1d(_P_FUNCS[p_value](scaled, n, alpha / bound))
    return risk, np.asarray(p, dtype=float)


def fwer_reject(p_values, delta, method="fixed_sequence", order=None):
    """Boolean mask of rejected nulls with family-wise error rate at most ``delta``.

    ``order`` is the testing order for ``"fixed_sequence"``: indices of
    ``p_values``, tested first to last. The default is index order.
    """
    delta = _checked_delta(delta)
    if method not in _FWER:
        raise ValueError(f"method must be one of {_FWER}")
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0 + 1e-12):
        raise ValueError("p_values must be a non-empty array in [0, 1]")
    m = p.size
    reject = np.zeros(m, dtype=bool)
    if method == "bonferroni":
        return p <= delta / m
    if method == "holm":
        for step, idx in enumerate(np.argsort(p, kind="stable")):
            if p[idx] > delta / (m - step):
                break
            reject[idx] = True
        return reject
    seq = np.arange(m) if order is None else np.asarray(order, dtype=int).ravel()
    if seq.size == 0 or np.unique(seq).size != seq.size or seq.min() < 0 or seq.max() >= m:
        raise ValueError("order must be distinct indices into p_values")
    for idx in seq:
        if p[idx] > delta:
            break
        reject[idx] = True
    return reject


@dataclass(frozen=True)
class LTTResult:
    """Outcome of :func:`learn_then_test`.

    ``valid`` marks every ``lambda`` certified to have risk at most ``alpha``
    (jointly, with probability at least ``1 - delta``). ``lambda_hat`` is the
    selected one, or ``None`` when nothing was certified.
    """

    lambdas: np.ndarray
    risk_hat: np.ndarray
    p_values: np.ndarray
    valid: np.ndarray
    lambda_hat: float | None
    alpha: float
    delta: float
    n_calibration: int
    p_value: str
    method: str

    @property
    def n_valid(self):
        return int(self.valid.sum())

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "n_calibration": self.n_calibration,
            "p_value": self.p_value,
            "method": self.method,
            "lambda_hat": self.lambda_hat,
            "n_valid": self.n_valid,
            "n_candidates": int(self.lambdas.size),
            "valid_lambdas": [float(v) for v in self.lambdas[self.valid]],
        }


def learn_then_test(
    losses,
    lambdas,
    alpha,
    delta=0.1,
    p_value="hb",
    method="fixed_sequence",
    loss_bound=1.0,
    order=None,
    select="min",
):
    """Certify the ``lambda`` values with risk ``<= alpha`` with probability ``>= 1 - delta``.

    Parameters
    ----------
    losses : array of shape (n, m)
        ``losses[i, j]`` is calibration point ``i``'s loss at ``lambdas[j]``,
        in ``[0, loss_bound]``. Monotonicity is **not** required.
    lambdas : array of shape (m,)
        Candidate parameters (distinct).
    alpha, delta : float
        Risk level and allowed failure probability.
    p_value : {"hb", "hoeffding", "bentkus", "binomial"}
    method : {"fixed_sequence", "bonferroni", "holm"}
    order : sequence of int, optional
        Testing order for ``fixed_sequence``. The default tests from the
        largest ``lambda`` down, which is the right order when larger
        ``lambda`` means a safer, larger set (as in conformal risk control).
    select : {"min", "max"}
        Which certified ``lambda`` to return: the smallest (default, the
        tightest set when losses fall with ``lambda``) or the largest.
    """
    delta = _checked_delta(delta)
    if select not in ("min", "max"):
        raise ValueError("select must be 'min' or 'max'")
    grid = np.asarray(lambdas, dtype=float).ravel()
    matrix = np.asarray(losses, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != grid.size or grid.size == 0:
        raise ValueError("losses must have shape (n_calibration, len(lambdas))")
    if not np.all(np.isfinite(grid)) or np.unique(grid).size != grid.size:
        raise ValueError("lambdas must be finite and distinct")
    risk, p = ltt_p_values(matrix, alpha, p_value=p_value, loss_bound=loss_bound)
    if method == "fixed_sequence" and order is None:
        order = np.argsort(-grid, kind="stable")
    valid = fwer_reject(p, delta, method=method, order=order)
    if valid.any():
        chosen = grid[valid].min() if select == "min" else grid[valid].max()
        lambda_hat = float(chosen)
    else:
        lambda_hat = None
    return LTTResult(
        lambdas=grid,
        risk_hat=risk,
        p_values=p,
        valid=valid,
        lambda_hat=lambda_hat,
        alpha=float(alpha),
        delta=delta,
        n_calibration=int(matrix.shape[0]),
        p_value=p_value,
        method=method,
    )


class LTTMultilabelController:
    """Multilabel sets whose false negative rate is ``<= alpha`` with probability ``>= 1 - delta``.

    This is the high-probability counterpart of
    :class:`~conformal_kit.risk_control.MultilabelRiskController`. The set
    for a row is ``{k : p_k >= 1 - lambda_hat}``. ``lambda_hat`` is chosen by
    Learn then Test (by default, Hoeffding-Bentkus p-values tested in fixed
    sequence from ``lambda = 1`` down) on a grid of ``n_lambdas`` thresholds.

    Attributes
    ----------
    lambda_ : float
    threshold_ : float
        Probability cut-off ``1 - lambda_``.
    calibration_risk_ : float
        Empirical calibration FNR at ``lambda_``.
    result_ : LTTResult
    """

    def __init__(self, alpha=0.1, delta=0.1, p_value="hb", method="fixed_sequence", n_lambdas=201):
        self.alpha = _checked_alpha(alpha)
        self.delta = _checked_delta(delta)
        if p_value not in _P_VALUES or p_value == "binomial":
            raise ValueError("p_value must be 'hb', 'hoeffding' or 'bentkus' (FNR is not 0/1)")
        if method not in _FWER:
            raise ValueError(f"method must be one of {_FWER}")
        if int(n_lambdas) < 2:
            raise ValueError("n_lambdas must be at least 2")
        self.p_value = p_value
        self.method = method
        self.n_lambdas = int(n_lambdas)
        self.lambda_ = None
        self.threshold_ = None
        self.calibration_risk_ = None
        self.result_ = None
        self.n_labels_ = None

    @staticmethod
    def _fnr_losses(y, p, grid):
        positives = np.maximum(y.sum(axis=1), 1)
        included = p[:, :, None] >= (1.0 - grid)[None, None, :] - 1e-12
        hits = (y[:, :, None] & included).sum(axis=1)
        return np.where(y.any(axis=1)[:, None], 1.0 - hits / positives[:, None], 0.0)

    def fit(self, y_true, probabilities):
        """Calibrate on held-out multi-hot labels and per-label probabilities."""
        y, p = _checked_multilabel(y_true, probabilities)
        grid = np.linspace(0.0, 1.0, self.n_lambdas)
        losses = self._fnr_losses(y, p, grid)
        result = learn_then_test(
            losses, grid, self.alpha, self.delta, p_value=self.p_value, method=self.method
        )
        if result.lambda_hat is None:
            raise ValueError(
                f"no threshold certified at alpha={self.alpha}, delta={self.delta} "
                f"with n={y.shape[0]}; use more calibration rows or a larger alpha/delta"
            )
        self.result_ = result
        self.lambda_ = result.lambda_hat
        self.threshold_ = 1.0 - self.lambda_
        self.calibration_risk_ = float(result.risk_hat[np.flatnonzero(grid == self.lambda_)[0]])
        self.n_labels_ = p.shape[1]
        return self

    def predict_set(self, probabilities):
        """Boolean ``(n, n_labels)`` mask of the labels in each set."""
        if self.lambda_ is None:
            raise RuntimeError("call fit before predicting")
        _, p = _checked_multilabel(None, probabilities)
        if p.shape[1] != self.n_labels_:
            raise ValueError(f"expected {self.n_labels_} label columns, got {p.shape[1]}")
        return p >= self.threshold_ - 1e-12

    def predict_labels(self, probabilities):
        """List of label-index arrays, one per row."""
        return [np.flatnonzero(row) for row in self.predict_set(probabilities)]
