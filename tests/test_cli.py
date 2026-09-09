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
