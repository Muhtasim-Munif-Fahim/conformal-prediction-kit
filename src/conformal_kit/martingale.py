"""Conformal test martingales: online testing of exchangeability.

Every split / full conformal guarantee assumes the data are exchangeable.
Conformal test martingales (Vovk, Nouretdinov & Gammerman 2003; Vovk 2021)
turn that assumption into something you can *monitor*: process the
observations one by one, compute the smoothed conformal p-value of each
new point relative to all points seen so far, and bet on those p-values
not being uniform. Under exchangeability the smoothed p-values are i.i.d.
``U(0, 1)``, so the capital ``S_n = prod_i f_i(p_i)`` (with
``integral_0^1 f_i = 1``) is a non-negative martingale with ``S_0 = 1``;
by Ville's inequality ``P(sup_n S_n >= 1 / alpha) <= alpha``. Raising an
alarm the first time ``S_n >= 1 / alpha`` is therefore an anytime-valid
test with false-alarm probability at most ``alpha`` *over the entire,
unbounded sequence* -- no multiple-testing correction needed.

Betting strategies:

* ``"power"``: ``f(p) = eps * p ** (eps - 1)`` for a fixed ``eps`` in (0, 1).
* ``"mixture"``: the power martingale averaged uniformly over ``eps``
  (computed on a fine grid in log space); no tuning parameter.
* ``"jumper"``: the Simple Jumper of Vovk et al. ("Retrain or not
  retrain?", 2021) mixing ``f_eps(p) = 1 + eps (p - 1/2)`` for
  ``eps in {-1, 0, 1}`` with jump rate ``J``; it recovers quickly after
  long stretches of exchangeability, so it is the one to use for
  change-point detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ExchangeabilityTestResult",
    "conformal_test_martingale",
    "exchangeability_test",
    "online_conformal_p_values",
]

_METHODS = ("jumper", "mixture", "power")


def _as_sequence(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim not in (1, 2) or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _mean_distance_scores(bag):
    """Non-conformity of every point in the bag: distance to the bag mean."""
    centre = bag.mean(axis=0)
    diff = bag - centre
    return np.abs(diff) if diff.ndim == 1 else np.sqrt((diff**2).sum(axis=1))


def _knn_scores(bag, k=1):
    """Distance to the k-th nearest other point of the bag (inf if none)."""
    pts = bag[:, None] if bag.ndim == 1 else bag
    n = pts.shape[0]
    if n <= k:
        return np.zeros(n)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def online_conformal_p_values(data, *, score="mean_distance", randomized=True, random_state=None):
    """Smoothed online conformal p-values ``p_1, ..., p_n``.

    At step ``n`` every point of the bag ``z_1..z_n`` is scored relative to
    the bag (so scores are recomputed, keeping them exchangeable) and::

        p_n = (#{i : a_i > a_n} + theta_n * #{i : a_i = a_n}) / n

    with ``theta_n ~ U(0, 1)`` (or ``1`` when ``randomized=False``, which
    gives conservative p-values). Under exchangeability the randomized
    p-values are exactly i.i.d. uniform.

    ``score`` is ``"mean_distance"`` (distance to the bag mean),
    ``"knn"`` (distance to the nearest other point), ``"identity"`` (the
    values themselves are already non-conformity scores computed by a
    fixed, pre-trained model) or a callable ``bag -> scores``.
    Cost is ``O(n^2)`` score evaluations for bag-dependent scores.
    """
    arr = _as_sequence(data, "data")
    rng = np.random.default_rng(random_state)
    n = arr.shape[0]
    p = np.empty(n)
    if score == "identity":
        if arr.ndim != 1:
            raise ValueError("score='identity' needs a 1-D array of scores")
        fn = None
    elif score == "mean_distance":
        fn = _mean_distance_scores
    elif score == "knn":
        fn = _knn_scores
    elif callable(score):
        fn = score
    else:
        raise ValueError("score must be 'mean_distance', 'knn', 'identity' or a callable")
    thetas = rng.uniform(size=n) if randomized else np.ones(n)
    for t in range(n):
        if fn is None:
            a = arr[: t + 1]
        else:
            a = np.asarray(fn(arr[: t + 1]), dtype=float)
            if a.shape != (t + 1,):
                raise ValueError("score callable must return one score per bag point")
        current = a[-1]
        greater = np.count_nonzero(a > current)
        equal = np.count_nonzero(a == current)
        p[t] = (greater + thetas[t] * equal) / (t + 1)
    return p


def _checked_p(p_values):
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p_values must be non-empty")
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("p_values must lie in [0, 1]")
    return p


def conformal_test_martingale(p_values, method="mixture", *, epsilon=0.92, jump=0.01, grid_size=1000):
    """Return the log10 path ``log10 S_1, ..., log10 S_n`` of a test martingale.

    ``method`` is ``"power"`` (uses ``epsilon``), ``"mixture"`` (power
    martingale mixed over ``epsilon ~ U(0, 1)``, ``grid_size`` midpoints) or
    ``"jumper"`` (Simple Jumper with jump rate ``jump``).
    """
    p = _checked_p(p_values)
    if method == "power":
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        # Guard p = 0 (only possible for non-randomized p-values of 0 count).
        logf = np.log(epsilon) + (epsilon - 1.0) * np.log(np.maximum(p, 1e-300))
        return np.cumsum(logf) / np.log(10.0)
    if method == "mixture":
        if not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError("grid_size must be an integer >= 2")
        eps = (np.arange(grid_size) + 0.5) / grid_size
        logp = np.log(np.maximum(p, 1e-300))
        # cumulative log capital for each eps: n log eps + (eps - 1) sum log p
        cum_logp = np.cumsum(logp)
        steps = np.arange(1, p.size + 1)
        logs = steps[:, None] * np.log(eps)[None, :] + (eps[None, :] - 1.0) * cum_logp[:, None]
        top = logs.max(axis=1, keepdims=True)
        mix = top[:, 0] + np.log(np.exp(logs - top).mean(axis=1))
        return mix / np.log(10.0)
    if method == "jumper":
        if not 0.0 <= jump <= 1.0:
            raise ValueError("jump must lie in [0, 1]")
        eps = np.array([-1.0, 0.0, 1.0])
        capital = np.full(3, 1.0 / 3.0)
        log_scale = 0.0  # keep capital normalised to avoid under/overflow
        out = np.empty(p.size)
        for i, pi in enumerate(p):
            total = capital.sum()
            capital = (1.0 - jump) * capital + jump * total / 3.0
            capital = capital * (1.0 + eps * (pi - 0.5))
            total = capital.sum()
            log_scale += np.log(total)
            capital = capital / total
            out[i] = log_scale / np.log(10.0)
        return out
    raise ValueError(f"method must be one of {_METHODS}")


@dataclass(frozen=True)
class ExchangeabilityTestResult:
    """Outcome of :func:`exchangeability_test`."""

    method: str
    alpha: float
    p_values: np.ndarray
    log10_martingale: np.ndarray
    alarm_index: int | None
    rejected: bool
    max_log10_martingale: float

    @property
    def final_log10_martingale(self) -> float:
        return float(self.log10_martingale[-1])

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n": int(self.p_values.size),
            "rejected": self.rejected,
            "alarm_index": self.alarm_index,
            "max_log10_martingale": self.max_log10_martingale,
            "final_log10_martingale": self.final_log10_martingale,
        }


def exchangeability_test(
    data,
    alpha=0.01,
    *,
    method="jumper",
    score="mean_distance",
    randomized=True,
    random_state=None,
    epsilon=0.92,
    jump=0.01,
):
    """Anytime-valid online test of exchangeability.

    Raises an alarm at the first (0-based) index where the martingale reaches
    ``1 / alpha``; under exchangeability this happens with probability at
    most ``alpha`` however long the sequence runs.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    p = online_conformal_p_values(data, score=score, randomized=randomized, random_state=random_state)
    path = conformal_test_martingale(p, method, epsilon=epsilon, jump=jump)
    level = -np.log10(alpha)
    hits = np.flatnonzero(path >= level)
    alarm = int(hits[0]) if hits.size else None
    return ExchangeabilityTestResult(
        method=method,
        alpha=float(alpha),
        p_values=p,
        log10_martingale=path,
        alarm_index=alarm,
        rejected=alarm is not None,
        max_log10_martingale=float(path.max()),
    )
