"""Command-line interface for conformal-prediction-kit."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from . import __version__
from .evaluation import interval_coverage_report
from .regression import SplitConformalRegressor

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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="conformal-kit",
        description="Distribution-free prediction intervals with coverage guarantees",
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

    handler = {"calibrate": _cmd_calibrate, "evaluate": _cmd_evaluate}[args.command]
    try:
        return handler(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
