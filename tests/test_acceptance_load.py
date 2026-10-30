"""CPU regressions for the validity of the concurrent long-request load gate."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from acceptance_load import RequestSample, assess_load  # noqa: E402


def sample(start: float, finish: float, elapsed: float) -> RequestSample:
    return RequestSample(True, start, finish, elapsed)


def experiment() -> tuple[list[RequestSample], list[RequestSample], list[RequestSample]]:
    return (
        [sample(0.0, 0.1, 0.1), sample(0.2, 0.3, 0.1)],
        [sample(10.0, 10.12, 0.12), sample(10.2, 10.32, 0.12)],
        [sample(9.0, 11.0, 2.0)],
    )


class TestAcceptanceLoad(unittest.TestCase):
    def assert_invalid(self, result: dict, message: str) -> None:
        self.assertFalse(result["valid"], result)
        self.assertFalse(result["passed"], result)
        self.assertIn(message, "\n".join(result["problems"]))
        json.dumps(result, allow_nan=False)

    def test_successful_overlapping_experiment_passes_unchanged_gate(self):
        result = assess_load(*experiment(), expected_count=2)
        self.assertTrue(result["valid"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["solo_p50_s"], 0.1)
        self.assertEqual(result["loaded_p50_s"], 0.12)
        self.assertEqual(result["long_p50_s"], 2.0)
        self.assertAlmostEqual(result["ratio"], 1.2)
        self.assertEqual(result["overlapping_long_count"], 1)

    def test_original_failed_long_infinity_false_positive_is_rejected(self):
        solo, loaded, long = experiment()
        long[0] = RequestSample(False, 9.0, 11.0, math.inf, "failed request")
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assert_invalid(result, "did not succeed")
        self.assertIsNone(result["long_p50_s"])
        self.assertEqual(result["overlapping_long_count"], 0)

    def test_any_failed_attempt_invalidates_the_experiment(self):
        for group_index in range(3):
            with self.subTest(group=group_index):
                groups = experiment()
                groups[group_index][0] = replace(groups[group_index][0], ok=False)
                self.assert_invalid(assess_load(*groups, expected_count=2), "did not succeed")

    def test_failed_nonoverlapping_long_attempt_is_not_silently_discarded(self):
        solo, loaded, long = experiment()
        long.append(RequestSample(False, 20.0, 21.0, 1.0, "failure after short work"))
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assert_invalid(result, "long[1]")
        self.assertEqual(result["overlapping_long_count"], 1)
        self.assertEqual(result["long_p50_s"], 2.0)

    def test_error_field_prevents_success_even_with_ok_true(self):
        solo, loaded, long = experiment()
        loaded[0] = replace(loaded[0], error="contradictory error")
        self.assert_invalid(assess_load(solo, loaded, long, expected_count=2), "contains a request error")

    def test_short_request_counts_must_match_the_expected_total(self):
        for group_index in (0, 1):
            for change in ("missing", "extra", "empty"):
                with self.subTest(group=group_index, change=change):
                    groups = experiment()
                    target = groups[group_index]
                    if change == "missing":
                        target.pop()
                    elif change == "extra":
                        target.append(target[0])
                    else:
                        target.clear()
                    self.assert_invalid(assess_load(*groups, expected_count=2), "expected 2")

    def test_empty_long_load_is_invalid(self):
        solo, loaded, _ = experiment()
        result = assess_load(solo, loaded, [], expected_count=2)
        self.assert_invalid(result, "at least one attempted request")
        self.assertIsNone(result["long_p50_s"])

    def test_missing_overlap_for_even_one_loaded_request_is_invalid(self):
        solo, loaded, _ = experiment()
        long = [sample(9.0, 10.15, 1.15)]
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assert_invalid(result, "has no positive overlap")
        self.assertEqual(result["overlapping_long_count"], 1)

    def test_touching_endpoints_do_not_count_as_overlap(self):
        solo, loaded, _ = experiment()
        for long in ([sample(9.0, 10.0, 1.0)], [sample(10.32, 11.0, 0.68)]):
            with self.subTest(long=long):
                result = assess_load(solo, loaded, long, expected_count=2)
                self.assert_invalid(result, "has no positive overlap")
                self.assertEqual(result["overlapping_long_count"], 0)

    def test_overlapping_long_median_excludes_unrelated_successful_load(self):
        solo, loaded, long = experiment()
        long.extend([sample(20.0, 120.0, 100.0), sample(121.0, 321.0, 200.0)])
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["long_p50_s"], 2.0)
        self.assertEqual(result["overlapping_long_count"], 1)

    def test_each_overlapping_long_is_counted_once_and_upper_median_is_used(self):
        solo, loaded, long = experiment()
        solo[1] = replace(solo[1], elapsed_s=0.2)
        loaded[1] = replace(loaded[1], elapsed_s=0.25)
        long.append(sample(8.0, 11.0, 3.0))
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["solo_p50_s"], 0.2)
        self.assertEqual(result["loaded_p50_s"], 0.25)
        self.assertEqual(result["long_p50_s"], 3.0)
        self.assertEqual(result["overlapping_long_count"], 2)

    def test_worker_exception_or_unfinished_worker_is_invalid(self):
        self.assert_invalid(assess_load(*experiment(), expected_count=2, worker_stopped=False), "has not stopped")
        self.assert_invalid(
            assess_load(*experiment(), expected_count=2, worker_error="worker exception"), "worker failed"
        )

    def test_nonfinite_clock_or_elapsed_values_on_any_attempt_are_invalid(self):
        for group_index in range(3):
            for field in ("started_s", "finished_s", "elapsed_s"):
                for value in (math.nan, math.inf, -math.inf):
                    with self.subTest(group=group_index, field=field, value=value):
                        groups = experiment()
                        groups[group_index][0] = replace(groups[group_index][0], **{field: value})
                        self.assert_invalid(assess_load(*groups, expected_count=2), f"{field} must be finite")

    def test_nonpositive_elapsed_and_invalid_intervals_are_rejected(self):
        changes = (
            {"started_s": -1.0},
            {"finished_s": 0.0},
            {"finished_s": 9.0},
            {"finished_s": 8.0},
            {"elapsed_s": 0.0},
            {"elapsed_s": -1.0},
        )
        for change in changes:
            with self.subTest(change=change):
                solo, loaded, long = experiment()
                long[0] = replace(long[0], **change)
                result = assess_load(solo, loaded, long, expected_count=2)
                self.assertFalse(result["valid"])
                self.assertFalse(result["passed"])

    def test_ratio_overflow_does_not_produce_infinity(self):
        solo, loaded, long = experiment()
        solo = [replace(row, elapsed_s=1e-308) for row in solo]
        loaded = [replace(row, elapsed_s=1e308) for row in loaded]
        result = assess_load(solo, loaded, long, expected_count=2)
        self.assert_invalid(result, "ratio is not finite")
        self.assertIsNone(result["ratio"])

    def test_thresholds_remain_inclusive_ratio_and_strict_half_long_latency(self):
        solo = [sample(0.0, 1.0, 1.0)]
        loaded = [sample(10.0, 11.5, 1.5)]
        long = [sample(9.0, 13.0, 4.0)]
        result = assess_load(solo, loaded, long, expected_count=1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["ratio"], 1.5)

        too_slow = [replace(loaded[0], elapsed_s=1.5001)]
        result = assess_load(solo, too_slow, long, expected_count=1)
        self.assertTrue(result["valid"])
        self.assertFalse(result["passed"])
        result = assess_load(solo, loaded, [replace(long[0], elapsed_s=3.0)], expected_count=1)
        self.assertTrue(result["valid"])
        self.assertFalse(result["passed"])

    def test_invalid_configuration_and_sample_shapes_fail_without_nan(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count):
                self.assert_invalid(assess_load(*experiment(), expected_count=count), "expected_count")
        self.assert_invalid(assess_load(None, [], [], expected_count=1), "solo must be a sequence")
        self.assert_invalid(assess_load([{}], [{}], [{}], expected_count=1), "must be a RequestSample")
        result = assess_load([], [], [], expected_count=1)
        self.assertFalse(result["valid"])
        for field in ("solo_p50_s", "loaded_p50_s", "long_p50_s", "ratio"):
            self.assertIsNone(result[field])
        json.dumps(result, allow_nan=False)

    def test_boolean_and_non_numeric_clocks_are_not_measurements(self):
        for value in (True, "1.0", None, 10**400):
            with self.subTest(value=value):
                solo, loaded, long = experiment()
                long[0] = replace(long[0], elapsed_s=value)
                self.assert_invalid(assess_load(solo, loaded, long, expected_count=2), "must be finite")

    def test_import_is_standard_library_only(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "import acceptance_load; "
            "assert not any(name in sys.modules for name in ('torch', 'sglang', 'servers', 'psutil'))"
        )
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
