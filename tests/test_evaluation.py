"""Tests for coverage evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_kit.evaluation import (
    coverage_by_group,
    interval_coverage_report,
    set_coverage_report,
)


class TestIntervalCoverageReport:
    def test_full_coverage_is_reported_as_one(self):
        report = interval_coverage_report([1.0, 2.0], [0.0, 1.0], [2.0, 3.0])
        assert report["coverage"] == 1.0
        assert report["n"] == 2

    def test_a_point_outside_its_interval_is_not_covered(self):
        report = interval_coverage_report([5.0, 2.0], [0.0, 1.0], [1.0, 3.0])
        assert report["coverage"] == 0.5

    def test_interval_endpoints_count_as_covered(self):
        report = interval_coverage_report([1.0, 3.0], [1.0, 1.0], [2.0, 3.0])
        assert report["coverage"] == 1.0

    def test_widths_are_summarised(self):
        report = interval_coverage_report([1.0, 1.0, 1.0], [0, 0, 0], [1.0, 2.0, 9.0])
        assert report["mean_width"] == pytest.approx(4.0)
        assert report["median_width"] == pytest.approx(2.0)

    def test_coverage_at_the_target_is_within_tolerance(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(0.0, 1.0, 2000)
        report = interval_coverage_report(
            truth, np.full(2000, -1.645), np.full(2000, 1.645), alpha=0.1
        )
        assert report["within_tolerance"] is True

    def test_a_clear_shortfall_is_flagged(self):
        truth = np.zeros(1000)
        lower = np.where(np.arange(1000) < 500, -1.0, 5.0)
        upper = lower + 1.0
        report = interval_coverage_report(truth, lower, upper, alpha=0.1)
        assert report["coverage"] == pytest.approx(0.5)
        assert report["within_tolerance"] is False

    def test_over_covering_is_not_flagged(self):
        # Conformal guarantees at least 1 - alpha; more is conservative.
        report = interval_coverage_report(
            np.zeros(500), np.full(500, -10.0), np.full(500, 10.0), alpha=0.1
        )
        assert report["coverage"] == 1.0
        assert report["within_tolerance"] is True

    def test_tolerance_shrinks_as_the_test_set_grows(self):
        small = interval_coverage_report(np.zeros(20), np.full(20, -1.0), np.full(20, 1.0))
        large = interval_coverage_report(np.zeros(2000), np.full(2000, -1.0), np.full(2000, 1.0))
        assert large["tolerance"] < small["tolerance"]

    def test_target_coverage_follows_alpha(self):
        report = interval_coverage_report([1.0], [0.0], [2.0], alpha=0.25)
        assert report["target_coverage"] == pytest.approx(0.75)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            interval_coverage_report([1.0, 2.0], [0.0], [3.0])

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            interval_coverage_report([], [], [])

    def test_an_inverted_interval_is_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to lower"):
            interval_coverage_report([1.0], [5.0], [0.0])

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1.0])
    def test_alpha_is_validated(self, bad):
        with pytest.raises(ValueError, match="alpha must be strictly between"):
            interval_coverage_report([1.0], [0.0], [2.0], alpha=bad)


class TestSetCoverageReport:
    def test_covered_when_the_true_class_is_in_the_set(self):
        mask = np.array([[True, False], [False, True]])
        report = set_coverage_report(np.array([0, 1]), mask)
        assert report["coverage"] == 1.0

    def test_missing_the_true_class_is_not_covered(self):
        mask = np.array([[True, False], [True, False]])
        report = set_coverage_report(np.array([0, 1]), mask)
        assert report["coverage"] == 0.5

    def test_set_sizes_are_summarised(self):
        mask = np.array([[True, False, False], [True, True, True]])
        report = set_coverage_report(np.array([0, 0]), mask)
        assert report["mean_set_size"] == pytest.approx(2.0)
        assert report["median_set_size"] == pytest.approx(2.0)

    def test_singleton_rate_counts_resolved_points(self):
        mask = np.array([[True, False], [True, True], [False, True], [True, True]])
        report = set_coverage_report(np.array([0, 0, 1, 1]), mask)
        assert report["singleton_rate"] == pytest.approx(0.5)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            set_coverage_report(np.array([0, 1, 0]), np.array([[True, False]]))

    def test_a_one_dimensional_mask_is_rejected(self):
        with pytest.raises(ValueError, match="2-D boolean mask"):
            set_coverage_report(np.array([0]), np.array([True, False]))

    def test_an_out_of_range_label_is_rejected(self):
        with pytest.raises(ValueError, match="outside the prediction sets"):
            set_coverage_report(np.array([9]), np.array([[True, False]]))

    def test_non_integer_labels_are_rejected(self):
        with pytest.raises(ValueError, match="integer class indices"):
            set_coverage_report(np.array([0.0]), np.array([[True, False]]))


class TestCoverageByGroup:
    def test_reports_one_entry_per_group(self):
        covered = np.array([True, True, False, False])
        groups = np.array(["a", "a", "b", "b"])
        rows = coverage_by_group(covered, groups)
        assert {row["group"] for row in rows} == {"a", "b"}
        assert {row["n"] for row in rows} == {2}

    def test_worst_served_group_is_listed_first(self):
        covered = np.array([True, True, True, True, False, False])
        groups = np.array(["good", "good", "good", "good", "bad", "bad"])
        rows = coverage_by_group(covered, groups)
        assert rows[0]["group"] == "bad"
        assert rows[0]["coverage"] == 0.0

    def test_a_subgroup_shortfall_is_visible_behind_good_marginal_coverage(self):
        # 90% overall, but one group is covered 50% of the time.
        covered = np.array([True] * 95 + [False] * 5)
        groups = np.array(["main"] * 90 + ["rare"] * 10)
        covered[90:95] = True
        covered[95:] = False
        rows = coverage_by_group(covered, groups, alpha=0.1)
        rare = next(row for row in rows if row["group"] == "rare")
        assert rare["coverage"] == pytest.approx(0.5)
        assert rare["within_tolerance"] is False

    def test_widths_are_averaged_per_group_when_given(self):
        covered = np.array([True, True, True, True])
        groups = np.array(["a", "a", "b", "b"])
        rows = coverage_by_group(covered, groups, widths=[1.0, 3.0, 10.0, 10.0])
        widths = {row["group"]: row["mean_width"] for row in rows}
        assert widths == {"a": pytest.approx(2.0), "b": pytest.approx(10.0)}

    def test_mean_width_is_absent_without_widths(self):
        rows = coverage_by_group(np.array([True, False]), np.array(["a", "b"]))
        assert all("mean_width" not in row for row in rows)

    def test_numeric_group_labels_work(self):
        rows = coverage_by_group(np.array([True, False]), np.array([1, 2]))
        assert {row["group"] for row in rows} == {1, 2}

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            coverage_by_group(np.array([True, False]), np.array(["a"]))

    def test_mismatched_widths_are_rejected(self):
        with pytest.raises(ValueError, match="widths must have the same length"):
            coverage_by_group(np.array([True, False]), np.array(["a", "b"]), widths=[1.0])


class TestGroupsThroughReports:
    def test_interval_report_exposes_groups(self):
        report = interval_coverage_report(
            [0.0, 0.0, 9.0, 9.0], [-1, -1, -1, -1], [1, 1, 1, 1],
            groups=["a", "a", "b", "b"],
        )
        assert report["coverage"] == 0.5
        assert report["by_group"][0]["group"] == "b"
        assert report["by_group"][0]["coverage"] == 0.0

    def test_set_report_exposes_groups(self):
        mask = np.array([[True, False], [False, True], [True, False], [True, False]])
        report = set_coverage_report(
            np.array([0, 1, 1, 1]), mask, groups=["a", "a", "b", "b"]
        )
        assert report["by_group"][0]["group"] == "b"
        assert report["by_group"][0]["coverage"] == 0.0
