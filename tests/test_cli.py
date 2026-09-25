"""Tests for the conformal-kit command-line interface."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conformal_kit import __version__
from conformal_kit.cli import main


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


@pytest.fixture()
def calibration_csv(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.normal(0.0, 1.0, 400)
    predictions = truth + rng.normal(0.0, 0.5, 400)
    return _write_csv(tmp_path / "cal.csv", ["y_true", "y_pred"], zip(truth, predictions))


@pytest.fixture()
def test_csv(tmp_path):
    rng = np.random.default_rng(1)
    return _write_csv(tmp_path / "test.csv", ["y_pred"], [[v] for v in rng.normal(0, 1, 50)])


class TestTopLevel:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_bare_invocation_prints_help(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_alpha_outside_the_unit_interval_is_rejected(self, capsys, calibration_csv, test_csv):
        code = main(["calibrate", "--calibration", calibration_csv,
                     "--test", test_csv, "--alpha", "1.5"])
        assert code == 2
        assert "alpha must be strictly between" in capsys.readouterr().err


class TestCalibrate:
    def test_reports_the_calibrated_width(self, calibration_csv, test_csv, capsys):
        assert main(["calibrate", "--calibration", calibration_csv, "--test", test_csv]) == 0
        out = capsys.readouterr().out
        assert "Interval width" in out
        assert "Target coverage : 90.0%" in out

    def test_json_output_is_parseable(self, calibration_csv, test_csv, capsys):
        assert main(["calibrate", "--calibration", calibration_csv,
                     "--test", test_csv, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_calibration"] == 400
        assert payload["n_test"] == 50
        assert payload["interval_width"] > 0

    def test_out_writes_intervals_that_bracket_the_prediction(
        self, calibration_csv, test_csv, tmp_path, capsys
    ):
        destination = tmp_path / "intervals.csv"
        assert main(["calibrate", "--calibration", calibration_csv,
                     "--test", test_csv, "--out", str(destination)]) == 0
        assert "Wrote" in capsys.readouterr().out
        with open(destination, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 50
        for row in rows:
            assert float(row["lower"]) <= float(row["y_pred"]) <= float(row["upper"])

    def test_a_missing_column_is_reported(self, tmp_path, test_csv, capsys):
        bad = _write_csv(tmp_path / "bad.csv", ["y_true"], [[1.0], [2.0]])
        assert main(["calibrate", "--calibration", bad, "--test", test_csv]) == 2
        assert "missing column(s): y_pred" in capsys.readouterr().err

    def test_a_non_numeric_cell_names_the_line(self, tmp_path, test_csv, capsys):
        bad = _write_csv(tmp_path / "bad.csv", ["y_true", "y_pred"], [[1.0, "oops"]])
        assert main(["calibrate", "--calibration", bad, "--test", test_csv]) == 2
        assert "line 2" in capsys.readouterr().err

    def test_a_missing_file_is_reported(self, test_csv, capsys):
        assert main(["calibrate", "--calibration", "nope.csv", "--test", test_csv]) == 2
        assert "could not read" in capsys.readouterr().err

    def test_an_empty_file_is_reported(self, tmp_path, test_csv, capsys):
        empty = _write_csv(tmp_path / "empty.csv", ["y_true", "y_pred"], [])
        assert main(["calibrate", "--calibration", empty, "--test", test_csv]) == 2
        assert "no data rows" in capsys.readouterr().err

    def test_too_few_points_for_the_level_is_reported(self, tmp_path, test_csv, capsys):
        tiny = _write_csv(tmp_path / "tiny.csv", ["y_true", "y_pred"], [[1.0, 1.0]] * 3)
        code = main(["calibrate", "--calibration", tiny, "--test", test_csv, "--alpha", "0.01"])
        assert code == 2
        assert "calibration points" in capsys.readouterr().err


class TestEvaluate:
    def test_good_coverage_exits_zero(self, tmp_path, capsys):
        rows = [[0.0, -1.0, 1.0] for _ in range(200)]
        path = _write_csv(tmp_path / "cov.csv", ["y_true", "lower", "upper"], rows)
        assert main(["evaluate", "--predictions", path]) == 0
        assert "meets target" in capsys.readouterr().out

    def test_a_coverage_shortfall_exits_nonzero(self, tmp_path, capsys):
        # Half the points sit outside their interval.
        rows = [[0.0, -1.0, 1.0] for _ in range(100)] + [[9.0, -1.0, 1.0] for _ in range(100)]
        path = _write_csv(tmp_path / "bad.csv", ["y_true", "lower", "upper"], rows)
        assert main(["evaluate", "--predictions", path]) == 1
        assert "BELOW TARGET" in capsys.readouterr().out

    def test_json_output_is_parseable(self, tmp_path, capsys):
        rows = [[0.0, -1.0, 1.0] for _ in range(50)]
        path = _write_csv(tmp_path / "cov.csv", ["y_true", "lower", "upper"], rows)
        assert main(["evaluate", "--predictions", path, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["coverage"] == 1.0
        assert payload["n"] == 50

    def test_an_inverted_interval_is_reported(self, tmp_path, capsys):
        path = _write_csv(tmp_path / "inv.csv", ["y_true", "lower", "upper"], [[0.0, 5.0, 1.0]])
        assert main(["evaluate", "--predictions", path]) == 2
        assert "greater than or equal to lower" in capsys.readouterr().err


class TestRoundTrip:
    def test_calibrated_intervals_evaluate_at_the_target(self, tmp_path, capsys):
        rng = np.random.default_rng(5)
        truth = rng.normal(0.0, 1.0, 2000)
        predictions = truth + rng.normal(0.0, 0.5, 2000)
        cal = _write_csv(
            tmp_path / "cal.csv", ["y_true", "y_pred"], zip(truth[:1000], predictions[:1000])
        )
        test = _write_csv(tmp_path / "test.csv", ["y_pred"], [[v] for v in predictions[1000:]])
        intervals = tmp_path / "intervals.csv"
        assert main(["calibrate", "--calibration", cal, "--test", test,
                     "--out", str(intervals)]) == 0

        with open(intervals, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        scored = tmp_path / "scored.csv"
        _write_csv(
            scored, ["y_true", "lower", "upper"],
            [[t, row["lower"], row["upper"]] for t, row in zip(truth[1000:], rows)],
        )
        capsys.readouterr()
        assert main(["evaluate", "--predictions", str(scored)]) == 0
        assert "meets target" in capsys.readouterr().out


def _class_rows(labels, probabilities):
    rows = []
    for label, probs in zip(labels, probabilities):
        rows.append([int(label), *probs])
    return rows


def _calibrated_class_problem(n=800, k=4, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=(n, k))
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.argmax(np.log(probabilities) + rng.gumbel(size=probabilities.shape), axis=1)
    return labels.astype(int), probabilities


class TestClassify:
    def test_aps_json_reports_the_quantile(self, tmp_path, capsys):
        labels, probabilities = _calibrated_class_problem(n=80, seed=2)
        header = ["y_true", "p0", "p1", "p2", "p3"]
        calibration = _write_csv(
            tmp_path / "cal.csv", header, _class_rows(labels[:40], probabilities[:40])
        )
        test = _write_csv(
            tmp_path / "test.csv", header[1:], [row[1:] for row in _class_rows(labels[40:], probabilities[40:])]
        )
        assert main(["classify", "--calibration", calibration, "--test", test, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["method"] == "aps"
        assert payload["n_calibration"] == 40
        assert payload["n_test"] == 40
        assert payload["quantile"] > 0

    def test_raps_with_labels_meets_the_coverage_target(self, tmp_path, capsys):
        labels, probabilities = _calibrated_class_problem(n=2000, seed=3)
        header = ["y_true"] + [f"p{i}" for i in range(4)]
        calibration = _write_csv(
            tmp_path / "cal.csv", header, _class_rows(labels[:1000], probabilities[:1000])
        )
        test = _write_csv(
            tmp_path / "test.csv", header, _class_rows(labels[1000:], probabilities[1000:])
        )
        code = main([
            "classify", "--calibration", calibration, "--test", test,
            "--method", "raps", "--penalty", "0.05", "--k-reg", "1",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "meets target" in out
        assert "(raps)" in out

    def test_out_writes_membership(self, tmp_path, capsys):
        header = ["y_true", "p0", "p1"]
        calibration = _write_csv(
            tmp_path / "cal.csv", header, [[0, 0.75, 0.25]] * 20
        )
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.6, 0.4]] * 5)
        destination = tmp_path / "sets.csv"
        assert main([
            "classify", "--calibration", calibration, "--test", test,
            "--method", "aps", "--out", str(destination),
        ]) == 0
        assert "Wrote" in capsys.readouterr().out
        with open(destination, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 5
        assert set(rows[0]) == {"in_0", "in_1", "set", "size"}
        for row in rows:
            assert row["in_0"] == "1"
            assert int(row["size"]) >= 1

    def test_a_miss_on_the_test_labels_exits_nonzero(self, tmp_path, capsys):
        calibration = _write_csv(
            tmp_path / "cal.csv", ["y_true", "p0", "p1"], [[0, 0.8, 0.2]] * 30
        )
        # The set keeps class 0; every test label is class 1.
        test = _write_csv(
            tmp_path / "test.csv", ["y_true", "p0", "p1"], [[1, 0.8, 0.2]] * 30
        )
        assert main(["classify", "--calibration", calibration, "--test", test]) == 1
        assert "BELOW TARGET" in capsys.readouterr().out

    def test_penalty_is_rejected_for_aps(self, tmp_path, capsys):
        calibration = _write_csv(
            tmp_path / "cal.csv", ["y_true", "p0", "p1"], [[0, 0.7, 0.3]] * 20
        )
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.6, 0.4]])
        code = main([
            "classify", "--calibration", calibration, "--test", test,
            "--method", "aps", "--penalty", "0.1",
        ])
        assert code == 2
        assert "only to --method raps" in capsys.readouterr().err

    def test_a_missing_probability_column_is_reported(self, tmp_path, capsys):
        calibration = _write_csv(tmp_path / "cal.csv", ["y_true", "p0"], [[0, 1.0]])
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.5, 0.5]])
        assert main(["classify", "--calibration", calibration, "--test", test]) == 2
        assert "probability columns" in capsys.readouterr().err

    def test_a_hole_in_the_probability_columns_is_reported(self, tmp_path, capsys):
        calibration = _write_csv(
            tmp_path / "cal.csv", ["y_true", "p0", "p2"], [[0, 0.5, 0.5]]
        )
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.5, 0.5]])
        assert main(["classify", "--calibration", calibration, "--test", test]) == 2
        assert "missing column(s): p1" in capsys.readouterr().err

    def test_a_missing_calibration_label_is_reported(self, tmp_path, capsys):
        calibration = _write_csv(tmp_path / "cal.csv", ["p0", "p1"], [[0.5, 0.5]])
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.5, 0.5]])
        assert main(["classify", "--calibration", calibration, "--test", test]) == 2
        assert "missing column(s): y_true" in capsys.readouterr().err

    def test_a_fractional_label_names_the_line(self, tmp_path, capsys):
        calibration = _write_csv(
            tmp_path / "cal.csv", ["y_true", "p0", "p1"], [[1.5, 0.5, 0.5]]
        )
        test = _write_csv(tmp_path / "test.csv", ["p0", "p1"], [[0.5, 0.5]])
        assert main(["classify", "--calibration", calibration, "--test", test]) == 2
        err = capsys.readouterr().err
        assert "line 2" in err
        assert "integer class index" in err
