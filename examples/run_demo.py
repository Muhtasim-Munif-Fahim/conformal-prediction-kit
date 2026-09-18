"""Compare standard, normalized, and CQR intervals on heteroskedastic data.

The point of the demo: on data whose noise grows across the input space, all
three methods hit the same marginal coverage, but only the adaptive ones put
their width where the uncertainty actually is. Marginal coverage alone cannot
tell them apart -- the per-region breakdown can.

Run with:  python examples/run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conformal_kit import (
    ConformalizedQuantileRegressor,
    SplitConformalRegressor,
    interval_coverage_report,
)

OUTPUT = Path(__file__).parent / "output"
SEED = 11
N = 4000


def make_data(rng):
    """A model with no bias but noise that grows tenfold across the range."""
    x = rng.uniform(0.0, 10.0, N)
    sigma = 0.2 + 0.5 * x
    y = np.sin(x) + rng.normal(0.0, sigma)
    # A model that has learned the mean but not the noise.
    prediction = np.sin(x)
    return x, y, prediction, sigma


def region_labels(x):
    """Split the input range into thirds so coverage can be read per region."""
    return np.where(x < 10 / 3, "low-noise", np.where(x < 20 / 3, "mid-noise", "high-noise"))


def main():
    rng = np.random.default_rng(SEED)
    x, y, prediction, sigma = make_data(rng)
    half = N // 2
    cal, test = slice(0, half), slice(half, N)
    groups = region_labels(x[test])

    standard = SplitConformalRegressor(alpha=0.1).fit(y[cal], prediction[cal])
    lower, upper = standard.predict_interval(prediction[test])
    standard_report = interval_coverage_report(
        y[test], lower, upper, alpha=0.1, groups=groups
    )

    normalized = SplitConformalRegressor(alpha=0.1, normalize=True).fit(
        y[cal], prediction[cal], difficulty=sigma[cal]
    )
    n_lower, n_upper = normalized.predict_interval(
        prediction[test], difficulty=sigma[test]
    )
    normalized_report = interval_coverage_report(
        y[test], n_lower, n_upper, alpha=0.1, groups=groups
    )

    # A quantile model aimed at 80% (normal 10%/90% points). CQR expands it
    # to the 90% target without flattening the width vs. x.
    z_80 = 1.2815515655446004
    q_lower = prediction - z_80 * sigma
    q_upper = prediction + z_80 * sigma
    cqr = ConformalizedQuantileRegressor(alpha=0.1).fit(
        y[cal], q_lower[cal], q_upper[cal]
    )
    c_lower, c_upper = cqr.predict_interval(q_lower[test], q_upper[test])
    cqr_report = interval_coverage_report(
        y[test], c_lower, c_upper, alpha=0.1, groups=groups
    )

    reports = (
        ("standard", standard_report),
        ("normalized", normalized_report),
        ("cqr", cqr_report),
    )

    lines = [
        "# Conformal intervals on heteroskedastic data",
        "",
        (
            f"{N} points, noise sigma rising from 0.2 to 5.2 across the range. "
            "Target coverage 90%."
        ),
        "",
        "## Marginal coverage",
        "",
        "| method | coverage | mean width |",
        "| --- | --- | --- |",
    ]
    for name, report in reports:
        lines.append(
            f"| {name} | {report['coverage']:.1%} | {report['mean_width']:.3f} |"
        )
    lines += [
        "",
        "All three meet the target. On this summary alone they look equivalent.",
        "",
        "## Coverage by noise region",
        "",
        "| method | region | coverage | mean width |",
        "| --- | --- | --- | --- |",
    ]
    for name, report in reports:
        for row in sorted(report["by_group"], key=lambda entry: str(entry["group"])):
            lines.append(
                f"| {name} | {row['group']} | {row['coverage']:.1%} | "
                f"{row['mean_width']:.3f} |"
            )

    def _spread(report):
        coverages = [row["coverage"] for row in report["by_group"]]
        return max(coverages) - min(coverages)

    standard_spread = _spread(standard_report)
    normalized_spread = _spread(normalized_report)
    cqr_spread = _spread(cqr_report)

    lines += [
        "",
        "## Reading it",
        "",
        (
            f"Coverage spread across regions: standard {standard_spread:.1%}, "
            f"normalized {normalized_spread:.1%}, CQR {cqr_spread:.1%}."
        ),
        "",
        (
            "The standard interval is one fixed width, so it over-covers the "
            "quiet region and under-covers the noisy one while averaging out to "
            "the target. Normalized conformal multiplies width by a difficulty "
            "estimate, so each region is served at close to the rate asked for. "
            "CQR expands a quantile model's lower and upper bounds by one shared "
            "amount: widths still track local noise, and coverage is more even "
            "than the constant-width interval, without needing a separate "
            "difficulty estimate."
        ),
        "",
        (
            "This is what marginal coverage hides, and why "
            "`interval_coverage_report` takes a `groups` argument."
        ),
        "",
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "heteroskedastic_report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")

    print(f"standard   coverage={standard_report['coverage']:.1%} "
          f"width={standard_report['mean_width']:.3f} spread={standard_spread:.1%}")
    print(f"normalized coverage={normalized_report['coverage']:.1%} "
          f"width={normalized_report['mean_width']:.3f} spread={normalized_spread:.1%}")
    print(f"cqr        coverage={cqr_report['coverage']:.1%} "
          f"width={cqr_report['mean_width']:.3f} spread={cqr_spread:.1%}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
