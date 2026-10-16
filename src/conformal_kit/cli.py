"""Command-line interface for conformal-prediction-kit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque

import numpy as np

from . import __version__
from .aps import APSClassifier, RAPSClassifier
from .enbpi import enbpi_interval
from .evaluation import interval_coverage_report, set_coverage_report
from .outliers import ConformalOutlierDetector
from .predictive import SplitConformalPredictiveSystem, pit_uniformity_report
from .regression import SplitConformalRegressor
from .weighted import WeightedConformalRegressor

__all__ = ["build_parser", "main"]


def _read_columns(path, required):
    """Read a CSV into a dict of float columns, checking the header first."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [name for name in required if name not in fieldnames]
            if missing:
                raise ValueError(
                    f"{path} is missing column(s): {', '.join(missing)}"
                )
            columns = {name: [] for name in required}
            for line, row in enumerate(reader, start=2):
                for name in required:
                    raw = row.get(name)
                    try:
                        columns[name].append(float(raw))
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"{path} line {line}: column '{name}' is not a number: {raw!r}"
                        ) from None
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from None
    if not columns[required[0]]:
        raise ValueError(f"{path} contains no data rows")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _cmd_calibrate(args):
    """Calibrate on one CSV and write intervals for another."""
    calibration = _read_columns(args.calibration, ["y_true", "y_pred"])
    test = _read_columns(args.test, ["y_pred"])

    model = SplitConformalRegressor(alpha=args.alpha).fit(
        calibration["y_true"], calibration["y_pred"]
    )
    lower, upper = model.predict_interval(test["y_pred"])

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y_pred", "lower", "upper"])
            writer.writerows(zip(test["y_pred"], lower, upper))

    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_calibration": model.n_calibration_,
        "n_test": len(lower),
        "interval_width": model.width,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Calibrated on {summary['n_calibration']} points at alpha={args.alpha}")
        print(f"Target coverage : {summary['target_coverage']:.1%}")
        print(f"Interval width  : {summary['interval_width']:.6g}")
        print(f"Intervals written for {summary['n_test']} test predictions")
        if args.out:
            print(f"Wrote {args.out}")
    return 0


def _parse_levels(raw):
    try:
        levels = [float(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"--quantiles must be comma-separated numbers, got {raw!r}") from None
    if not levels:
        raise ValueError("--quantiles must list at least one level")
    return levels


def _cmd_predictive(args):
    """Conformal predictive distributions: quantiles, intervals, PIT and CRPS."""
    calibration = _read_columns(args.calibration, ["y_true", "y_pred"])
    test = _read_columns(args.test, ["y_pred"])
    levels = _parse_levels(args.quantiles)
    system = SplitConformalPredictiveSystem().fit(
        calibration["y_true"], calibration["y_pred"]
    )
    quantiles = system.quantile(levels, test["y_pred"])
    lower, upper = system.predict_interval(test["y_pred"], alpha=args.alpha)

    summary = {
        "alpha": args.alpha,
        "n_calibration": system.n_calibration_,
        "n_test": int(test["y_pred"].size),
        "quantile_levels": levels,
    }
    try:
        labelled = _read_columns(args.test, ["y_pred", "y_true"])
    except ValueError:
        labelled = None
    if labelled is not None:
        y_true = labelled["y_true"]
        pit = system.pit(y_true, labelled["y_pred"], seed=args.seed)
        report = pit_uniformity_report(pit)
        covered = (y_true >= lower) & (y_true <= upper)
        summary.update(
            {
                "mean_crps": float(np.mean(system.crps(y_true, labelled["y_pred"]))),
                "pit_ks_statistic": report["ks_statistic"],
                "pit_max_bin_deviation": report["max_bin_deviation"],
                "interval_coverage": float(np.mean(covered)),
            }
        )

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["y_pred", "lower", "upper"] + [f"q{level:g}" for level in levels]
            )
            for row in zip(test["y_pred"], lower, upper, quantiles):
                writer.writerow([row[0], row[1], row[2], *row[3]])

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Conformal predictive system on {summary['n_calibration']} calibration points")
        print(f"Quantile levels : {', '.join(f'{lv:g}' for lv in levels)}")
        print(f"Central interval: {1 - args.alpha:.1%}")
        if "mean_crps" in summary:
            print(f"Mean CRPS       : {summary['mean_crps']:.6g}")
            print(f"PIT KS distance : {summary['pit_ks_statistic']:.4f}")
            print(f"Coverage        : {summary['interval_coverage']:.1%}")
        if args.out:
            print(f"Wrote {args.out}")
    return 0


def _cmd_outliers(args):
    """Flag outliers in a test CSV of anomaly scores against inlier calibration scores."""
    column = args.score_column
    calibration = _read_columns(args.calibration, [column])
    test = _read_columns(args.test, [column])
    detector = ConformalOutlierDetector(
        alpha=args.alpha,
        method=args.method,
        randomized=args.randomized,
        random_state=args.seed,
    ).fit(calibration[column])
    report = detector.decision_report(test[column])

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([column, "p_value", "q_value", "outlier"])
            writer.writerows(
                zip(test[column], report["p_values"], report["q_values"], report["flagged"])
            )

    summary = {
        key: report[key]
        for key in ("method", "alpha", "n_calibration", "n_test", "n_flagged")
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        control = "FDR" if args.method == "bh" else "per-point false-alarm rate"
        print(f"Calibrated on {summary['n_calibration']} inlier scores")
        print(f"Method          : {args.method} ({control} <= {args.alpha:g})")
        print(f"Flagged         : {summary['n_flagged']} of {summary['n_test']} test points")
        if args.out:
            print(f"Wrote {args.out}")
    return 0


def _cmd_weighted(args):
    """Weighted split conformal under covariate shift from two CSVs."""
    weight = args.weight_column
    calibration = _read_columns(args.calibration, ["y_true", "y_pred", weight])
    test = _read_columns(args.test, ["y_pred", weight])

    model = WeightedConformalRegressor(alpha=args.alpha).fit(
        calibration["y_true"], calibration["y_pred"], calibration[weight]
    )
    lower, upper = model.predict_interval(test["y_pred"], test[weight])
    widths = upper - lower
    finite = np.isfinite(widths)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y_pred", "weight", "lower", "upper"])
            writer.writerows(zip(test["y_pred"], test[weight], lower, upper))

    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_calibration": model.n_calibration_,
        "effective_n_calibration": model.effective_n_,
        "n_test": int(lower.size),
        "n_infinite": int((~finite).sum()),
        "median_finite_width": float(np.median(widths[finite])) if finite.any() else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Calibrated on {summary['n_calibration']} weighted points at alpha={args.alpha} "
            f"(effective n = {summary['effective_n_calibration']:.1f})"
        )
        print(f"Target coverage     : {summary['target_coverage']:.1%}")
        if summary["median_finite_width"] is not None:
            print(f"Median finite width : {summary['median_finite_width']:.6g}")
        print(f"Infinite intervals  : {summary['n_infinite']} of {summary['n_test']}")
        if args.out:
            print(f"Wrote {args.out}")
    return 0


def _probability_column_names(fieldnames, path):
    """Return ``p0, p1, ...`` in order, or raise if the block has a hole."""
    present = set()
    for name in fieldnames:
        if len(name) > 1 and name[0] == "p" and name[1:].isdigit():
            present.add(int(name[1:]))
    if not present or 0 not in present:
        raise ValueError(f"{path} is missing probability columns p0,p1,...")
    last = max(present)
    missing = [f"p{index}" for index in range(last + 1) if index not in present]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
    if last < 1:
        raise ValueError(f"{path} is missing probability columns p0,p1,...")
    return [f"p{index}" for index in range(last + 1)]


def _read_class_csv(path):
    """Read ``y_true`` (optional) and ``p0,p1,...`` probability columns.

    ``y_true`` is required for calibration and optional for a test file.
    Probability columns must be a contiguous block starting at ``p0``.
    """
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            prob_names = _probability_column_names(fieldnames, path)
            has_labels = "y_true" in fieldnames
            wanted = (["y_true"] if has_labels else []) + prob_names
            columns = {name: [] for name in wanted}
            for line, row in enumerate(reader, start=2):
                for name in wanted:
                    raw = row.get(name)
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"{path} line {line}: column '{name}' is not a number: {raw!r}"
                        ) from None
                    if name == "y_true" and abs(value - round(value)) > 1e-8:
                        raise ValueError(
                            f"{path} line {line}: column 'y_true' is not an "
                            f"integer class index: {raw!r}"
                        )
                    columns[name].append(value)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from None
    if not columns[prob_names[0]]:
        raise ValueError(f"{path} contains no data rows")
    probabilities = np.column_stack([columns[name] for name in prob_names])
    labels = None
    if has_labels:
        labels = np.rint(columns["y_true"]).astype(int)
    return labels, probabilities


def _cmd_classify(args):
    """Calibrate APS or RAPS on one CSV and write prediction sets for another."""
    if args.method == "aps" and (args.penalty is not None or args.k_reg is not None):
        raise ValueError("--penalty and --k-reg apply only to --method raps")

    cal_labels, cal_probabilities = _read_class_csv(args.calibration)
    if cal_labels is None:
        raise ValueError(f"{args.calibration} is missing column(s): y_true")
    test_labels, test_probabilities = _read_class_csv(args.test)

    if args.method == "aps":
        model = APSClassifier(alpha=args.alpha).fit(cal_labels, cal_probabilities)
    else:
        penalty = 0.01 if args.penalty is None else args.penalty
        k_reg = 1 if args.k_reg is None else args.k_reg
        model = RAPSClassifier(alpha=args.alpha, penalty=penalty, k_reg=k_reg).fit(
            cal_labels, cal_probabilities
        )

    mask = model.predict(test_probabilities)
    sizes = mask.sum(axis=1)
    label_lists = [np.flatnonzero(row).tolist() for row in mask]

    if args.out:
        n_classes = mask.shape[1]
        header = [f"in_{index}" for index in range(n_classes)] + ["set", "size"]
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row, included, size in zip(mask, label_lists, sizes):
                membership = [int(flag) for flag in row]
                rendered = " ".join(str(label) for label in included)
                writer.writerow(membership + [rendered, int(size)])

    summary = {
        "alpha": args.alpha,
        "method": args.method,
        "target_coverage": 1.0 - args.alpha,
        "n_calibration": model.n_calibration_,
        "n_test": int(mask.shape[0]),
        "n_classes": int(mask.shape[1]),
        "quantile": model.quantile_,
        "mean_set_size": float(sizes.mean()),
    }
    if args.method == "raps":
        summary["penalty"] = model.penalty
        summary["k_reg"] = model.k_reg

    report = None
    if test_labels is not None:
        report = set_coverage_report(test_labels, mask, alpha=args.alpha)
        summary["coverage"] = report["coverage"]
        summary["within_tolerance"] = report["within_tolerance"]
        summary["median_set_size"] = report["median_set_size"]
        summary["singleton_rate"] = report["singleton_rate"]

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Calibrated on {summary['n_calibration']} points at "
            f"alpha={args.alpha} ({args.method})"
        )
        print(f"Target coverage : {summary['target_coverage']:.1%}")
        print(f"Quantile        : {summary['quantile']:.6g}")
        print(f"Mean set size   : {summary['mean_set_size']:.6g}")
        if report is not None:
            print(f"Coverage        : {report['coverage']:.3%}")
            verdict = "meets target" if report["within_tolerance"] else "BELOW TARGET"
            print(f"Verdict         : {verdict}")
        print(f"Sets written for {summary['n_test']} test predictions")
        if args.out:
            print(f"Wrote {args.out}")
    if report is not None and not report["within_tolerance"]:
        return 1
    return 0


def _cmd_evaluate(args):
    """Report empirical coverage for a CSV of intervals."""
    data = _read_columns(args.predictions, ["y_true", "lower", "upper"])
    report = interval_coverage_report(
        data["y_true"], data["lower"], data["upper"], alpha=args.alpha
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Points          : {report['n']}")
        print(f"Coverage        : {report['coverage']:.3%}")
        print(f"Target coverage : {report['target_coverage']:.3%}")
        print(f"Mean width      : {report['mean_width']:.6g}")
        print(f"Median width    : {report['median_width']:.6g}")
        verdict = "meets target" if report["within_tolerance"] else "BELOW TARGET"
        print(f"Verdict         : {verdict}")
    # A coverage shortfall is a real finding, so make it visible to a shell.
    return 0 if report["within_tolerance"] else 1



def _cmd_enbpi(args):
    """Build EnbPI intervals from sequential y_true/y_pred residuals."""
    data = _read_columns(args.data, ["y_true", "y_pred"])
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    n = y_true.size
    if n < 2:
        raise ValueError("enbpi needs at least two rows")
    split = args.train_size
    if not 0.0 < split < 1.0:
        raise ValueError("--train-size must be strictly between 0 and 1")
    n_train = max(1, min(n - 1, round(split * n)))
    residuals = np.abs(y_true[:n_train] - y_pred[:n_train])
    lowers = []
    uppers = []

    pool = deque(residuals.tolist(), maxlen=args.max_resid)
    for i in range(n_train, n):
        lower, upper = enbpi_interval(y_pred[i], np.asarray(pool, dtype=float), args.alpha)
        lowers.append(lower)
        uppers.append(upper)
        pool.append(abs(float(y_true[i]) - float(y_pred[i])))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y_true", "y_pred", "lower", "upper"])
            for i, (lo, hi) in enumerate(zip(lowers, uppers)):
                idx = n_train + i
                writer.writerow([y_true[idx], y_pred[idx], lo, hi])

    covered = [
        float(lo <= y_true[n_train + i] <= hi)
        for i, (lo, hi) in enumerate(zip(lowers, uppers))
    ]
    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_train": n_train,
        "n_test": len(lowers),
        "empirical_coverage": float(np.mean(covered)) if covered else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"EnbPI on {summary['n_test']} steps after {n_train} burn-in rows")
        print(f"Target coverage : {summary['target_coverage']:.1%}")
        if summary["empirical_coverage"] is not None:
            print(f"Empirical cover : {summary['empirical_coverage']:.1%}")
        if args.out:
            print(f"Wrote {args.out}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="conformal-kit",
        description=(
            "Distribution-free prediction intervals and prediction sets "
            "with coverage guarantees"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    calibrate = sub.add_parser(
        "calibrate", help="Calibrate on held-out predictions and emit test intervals"
    )
    calibrate.add_argument("--calibration", required=True, help="CSV with y_true,y_pred")
    calibrate.add_argument("--test", required=True, help="CSV with y_pred")
    calibrate.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    calibrate.add_argument("--out", default=None, help="Write intervals to this CSV")
    calibrate.add_argument("--json", action="store_true", help="Emit JSON")

    classify = sub.add_parser(
        "classify",
        help="Calibrate APS or RAPS and emit test prediction sets",
    )
    classify.add_argument(
        "--calibration",
        required=True,
        help="CSV with y_true,p0,p1,... class probabilities",
    )
    classify.add_argument(
        "--test",
        required=True,
        help="CSV with p0,p1,... and optional y_true",
    )
    classify.add_argument(
        "--method",
        choices=("aps", "raps"),
        default="aps",
        help="aps is cumulative probability; raps adds a rank penalty",
    )
    classify.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    classify.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="RAPS penalty per rank past k-reg (default 0.01)",
    )
    classify.add_argument(
        "--k-reg",
        type=int,
        default=None,
        help="RAPS leaves the first k ranks unpenalized (default 1)",
    )
    classify.add_argument("--out", default=None, help="Write prediction sets to this CSV")
    classify.add_argument("--json", action="store_true", help="Emit JSON")

    enbpi = sub.add_parser(
        "enbpi",
        help="Sequential EnbPI intervals from a CSV of y_true,y_pred",
    )
    enbpi.add_argument("--data", required=True, help="CSV with y_true,y_pred in time order")
    enbpi.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    enbpi.add_argument(
        "--train-size",
        type=float,
        default=0.5,
        help="Fraction of rows used to seed the residual pool",
    )
    enbpi.add_argument(
        "--max-resid",
        type=int,
        default=None,
        help="Sliding residual pool length (default: keep all)",
    )
    enbpi.add_argument("--out", default=None, help="Write intervals to this CSV")
    enbpi.add_argument("--json", action="store_true", help="Emit JSON")

    weighted = sub.add_parser(
        "weighted",
        help="Weighted split conformal intervals under covariate shift",
    )
    weighted.add_argument(
        "--calibration", required=True, help="CSV with y_true,y_pred,weight"
    )
    weighted.add_argument("--test", required=True, help="CSV with y_pred,weight")
    weighted.add_argument(
        "--weight-column",
        default="weight",
        help="Likelihood-ratio column name in both CSVs (default: weight)",
    )
    weighted.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    weighted.add_argument("--out", default=None, help="Write intervals to this CSV")
    weighted.add_argument("--json", action="store_true", help="Emit JSON")

    predictive = sub.add_parser(
        "predictive",
        help="Conformal predictive distributions: quantiles, PIT and CRPS",
    )
    predictive.add_argument("--calibration", required=True, help="CSV with y_true,y_pred")
    predictive.add_argument(
        "--test", required=True, help="CSV with y_pred (and optional y_true for PIT/CRPS)"
    )
    predictive.add_argument(
        "--quantiles", default="0.05,0.25,0.5,0.75,0.95", help="Comma-separated levels"
    )
    predictive.add_argument("--alpha", type=float, default=0.1, help="1 - central interval coverage")
    predictive.add_argument("--seed", type=int, default=None, help="Seed for randomized PIT")
    predictive.add_argument("--out", default=None, help="Write quantiles to this CSV")
    predictive.add_argument("--json", action="store_true", help="Emit JSON")

    outliers = sub.add_parser(
        "outliers",
        help="Conformal outlier detection with BH false-discovery-rate control",
    )
    outliers.add_argument("--calibration", required=True, help="CSV of inlier anomaly scores")
    outliers.add_argument("--test", required=True, help="CSV of test anomaly scores")
    outliers.add_argument(
        "--score-column", default="score", help="Anomaly-score column, larger = more anomalous"
    )
    outliers.add_argument("--alpha", type=float, default=0.1, help="Target FDR (bh) or false-alarm rate")
    outliers.add_argument("--method", choices=["bh", "marginal"], default="bh")
    outliers.add_argument("--randomized", action="store_true", help="Randomized tie-breaking p-values")
    outliers.add_argument("--seed", type=int, default=None, help="Seed for --randomized")
    outliers.add_argument("--out", default=None, help="Write p-values and flags to this CSV")
    outliers.add_argument("--json", action="store_true", help="Emit JSON")

    evaluate = sub.add_parser(
        "evaluate", help="Report empirical coverage for a CSV of intervals"
    )
    evaluate.add_argument(
        "--predictions", required=True, help="CSV with y_true,lower,upper"
    )
    evaluate.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    evaluate.add_argument("--json", action="store_true", help="Emit JSON")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    if not 0.0 < args.alpha < 1.0:
        print("--alpha must be strictly between 0 and 1", file=sys.stderr)
        return 2

    handler = {
        "calibrate": _cmd_calibrate,
        "classify": _cmd_classify,
        "enbpi": _cmd_enbpi,
        "evaluate": _cmd_evaluate,
        "outliers": _cmd_outliers,
        "predictive": _cmd_predictive,
        "weighted": _cmd_weighted,
    }[args.command]
    try:
        return handler(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
