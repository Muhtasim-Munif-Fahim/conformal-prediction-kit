"""Check that conformal outputs actually deliver the coverage they promise.

The guarantee is a theorem about exchangeable data, not an observation about
your data. Verifying it on a test set is what catches the assumption breaking:
a distribution shift, a calibration set that leaked into training, a subgroup
the marginal average is hiding.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "coverage_by_group",
    "interval_coverage_report",
    "set_coverage_report",
]


def _coverage_tolerance(covered, n, alpha, z=1.96):
    """Half-width of a normal-approximation band around the target coverage."""
    target = 1.0 - alpha
    if n == 0:
        return 0.0
    return z * math.sqrt(max(target * (1.0 - target), 1e-12) / n)


def interval_coverage_report(y_true, lower, upper, alpha=0.1, groups=None):
    """Measure empirical coverage and width of prediction intervals.

    Returns ``coverage`` (share of points inside their interval),
    ``target_coverage``, ``n``, ``mean_width``, ``median_width``,
    ``tolerance`` and ``within_tolerance``.

    ``within_tolerance`` compares the observed coverage against a normal
    approximation band around the target, so ordinary sampling noise on a
    small test set does not read as a broken guarantee. Coverage *above* the
    target is never flagged: conformal prediction guarantees at least
    ``1 - alpha``, and over-covering is conservative, not wrong.

    Passing ``groups`` adds a ``by_group`` entry. The conformal guarantee is
    marginal -- averaged over test points -- so it can hold overall while
    failing badly within a subgroup. Checking that is the caller's job, and
    this is the hook for it.
    """
    observed = np.asarray(y_true, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    if not observed.size:
        raise ValueError("y_true must not be empty")
    if not (observed.size == low.size == high.size):
        raise ValueError("y_true, lower and upper must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if np.any(high < low):
        raise ValueError("upper must be greater than or equal to lower")

    inside = (observed >= low) & (observed <= high)
    widths = high - low
    coverage = float(inside.mean())
    tolerance = _coverage_tolerance(coverage, observed.size, alpha)

    report = {
        "n": int(observed.size),
        "coverage": coverage,
        "target_coverage": 1.0 - alpha,
        "tolerance": tolerance,
        # Over-covering is conservative, so only a shortfall counts as a miss.
        "within_tolerance": bool(coverage >= (1.0 - alpha) - tolerance),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
    }
    if groups is not None:
        report["by_group"] = coverage_by_group(inside, groups, alpha=alpha, widths=widths)
    return report


def set_coverage_report(y_true, prediction_sets, alpha=0.1, groups=None):
    """Measure empirical coverage and size of classification prediction sets.

    ``prediction_sets`` is the boolean ``(n_samples, n_classes)`` mask from
    :meth:`~conformal_kit.classification.SplitConformalClassifier.predict_set`
    or :meth:`~conformal_kit.classification.MondrianConformalClassifier.predict_set`.
    Returns the same coverage fields as :func:`interval_coverage_report`
    plus ``mean_set_size``, ``median_set_size`` and ``singleton_rate`` --
    the share of points the model resolved to exactly one class, which is
    the practical measure of how useful the sets are.

    Pass ``groups=y_true`` to read coverage per class. That is the check
    Mondrian is designed to pass and split conformal is not.
    """
    labels = np.asarray(y_true).ravel()
    mask = np.asarray(prediction_sets, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("prediction_sets must be a 2-D boolean mask")
    if labels.size != mask.shape[0]:
        raise ValueError("y_true and prediction_sets must have the same length")
    if not labels.size:
        raise ValueError("y_true must not be empty")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true must contain integer class indices")
    if labels.min() < 0 or labels.max() >= mask.shape[1]:
        raise ValueError("y_true contains a class index outside the prediction sets")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    inside = mask[np.arange(labels.size), labels]
    sizes = mask.sum(axis=1)
    coverage = float(inside.mean())
    tolerance = _coverage_tolerance(coverage, labels.size, alpha)

    report = {
        "n": int(labels.size),
        "coverage": coverage,
        "target_coverage": 1.0 - alpha,
        "tolerance": tolerance,
        "within_tolerance": bool(coverage >= (1.0 - alpha) - tolerance),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
    }
    if groups is not None:
        report["by_group"] = coverage_by_group(inside, groups, alpha=alpha, widths=sizes)
    return report


def coverage_by_group(covered, groups, alpha=0.1, widths=None):
    """Break a boolean covered-mask down by group label.

    Returns one entry per group -- ``group``, ``n``, ``coverage``,
    ``tolerance``, ``within_tolerance`` and, when ``widths`` is given,
    ``mean_width`` -- sorted by coverage ascending so the worst-served group
    is first. That ordering is the point: a marginal guarantee can be met
    while one subgroup is badly under-covered, and this puts that group at
    the top of the list rather than leaving it to be searched for.
    """
    inside = np.asarray(covered, dtype=bool).ravel()
    labels = np.asarray(groups).ravel()
    if inside.size != labels.size:
        raise ValueError("covered and groups must have the same length")
    if widths is not None:
        measures = np.asarray(widths, dtype=float).ravel()
        if measures.size != inside.size:
            raise ValueError("widths must have the same length as covered")
    else:
        measures = None

    rows = []
    for name in dict.fromkeys(labels.tolist()):
        selected = labels == name
        subset = inside[selected]
        coverage = float(subset.mean())
        tolerance = _coverage_tolerance(coverage, subset.size, alpha)
        entry = {
            "group": name,
            "n": int(subset.size),
            "coverage": coverage,
            "tolerance": tolerance,
            "within_tolerance": bool(coverage >= (1.0 - alpha) - tolerance),
        }
        if measures is not None:
            entry["mean_width"] = float(measures[selected].mean())
        rows.append(entry)

    rows.sort(key=lambda row: (row["coverage"], str(row["group"])))
    return rows
