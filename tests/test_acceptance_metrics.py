"""CPU regressions for completeness and identity checks in acceptance metrics."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from acceptance_metrics import compare  # noqa: E402


def rows(count: int = 2, *, metadata: bool = True) -> list[dict]:
    result = []
    for index in range(count):
        row = {
            "token": 10,
            "top": [[-0.25 * (slot + 1), 10 + slot] for slot in range(5)],
            "prompt_lp": [-1.0, -2.0, -3.0],
            "secs": 0.1,
        }
        if metadata:
            row.update(input_id=f"prompt-{index}", prompt_positions=[17, 18, 19])
        result.append(row)
    return result


class TestAcceptanceMetrics(unittest.TestCase):
    def assert_invalid(self, result: dict, message: str) -> None:
        self.assertFalse(result["valid"], result)
        self.assertFalse(result["exact"], result)
        self.assertFalse(result["identity_verified"], result)
        self.assertIn(message, "\n".join(result["problems"]))
        json.dumps(result, allow_nan=False)

    def test_complete_identical_results_are_exact_with_full_coverage(self):
        reference = rows()
        result = compare(reference, copy.deepcopy(reference), expected_count=2)
        self.assertTrue(result["valid"])
        self.assertTrue(result["exact"])
        self.assertTrue(result["identity_verified"])
        self.assertEqual(result["problems"], [])
        self.assertEqual((result["n"], result["errors"], result["first_token_match"]), (2, 0, 2))
        self.assertEqual((result["prompt_positions"], result["compared_topk_entries"]), (6, 10))
        self.assertEqual((result["topk_set_mismatches"], result["topk_symmetric_difference"]), (0, 0))
        for metric in ("max_d_out", "mean_d_prompt", "p99_d_prompt", "max_d_prompt"):
            self.assertEqual(result[metric], 0.0)

    def test_known_finite_differences_are_measured(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        candidate[0]["token"] = 11
        candidate[0]["top"][0][0] -= 0.5
        candidate[0]["prompt_lp"] = [-1.0, -2.25, -3.5]
        result = compare(reference, candidate)
        self.assertTrue(result["valid"])
        self.assertFalse(result["exact"])
        self.assertEqual(result["first_token_match"], 0)
        self.assertEqual(result["max_d_out"], 0.5)
        self.assertEqual(result["mean_d_prompt"], 0.25)
        self.assertEqual(result["p99_d_prompt"], 0.5)
        self.assertEqual(result["max_d_prompt"], 0.5)

    def test_legacy_rows_validate_structure_without_claiming_identity(self):
        reference = rows(metadata=False)
        result = compare(reference, copy.deepcopy(reference))
        self.assertTrue(result["valid"])
        self.assertTrue(result["exact"])
        self.assertFalse(result["identity_verified"])

    def test_missing_request_is_not_silently_zipped_away(self):
        reference = rows()
        self.assert_invalid(compare(reference, reference[:1]), "result counts differ")
        self.assert_invalid(compare(reference[:1], reference), "result counts differ")
        self.assert_invalid(compare(reference[:1], reference[:1], expected_count=2), "expected 2")

    def test_baseline_false_positive_top1_and_one_prompt_position_is_rejected(self):
        reference = rows(16, metadata=False)
        candidate = copy.deepcopy(reference)
        for row in candidate:
            row["top"] = row["top"][:1]
            row["prompt_lp"] = row["prompt_lp"][:1]
        result = compare(reference, candidate, expected_count=16)
        self.assert_invalid(result, "expected 5")
        self.assertEqual(result["compared_topk_entries"], 0)
        self.assertEqual(result["prompt_positions"], 0)

    def test_missing_prompt_logprob_is_rejected_for_legacy_and_new_rows(self):
        for metadata in (False, True):
            with self.subTest(metadata=metadata):
                reference = rows(1, metadata=metadata)
                candidate = copy.deepcopy(reference)
                candidate[0]["prompt_lp"].pop()
                if metadata:
                    candidate[0]["prompt_positions"].pop()
                self.assert_invalid(compare(reference, candidate), "prompt_lp lengths differ")

    def test_complete_topk_membership_change_is_valid_but_not_exact(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        candidate[0]["top"][-1] = [-1.5, 99]
        result = compare(reference, candidate)
        self.assertTrue(result["valid"])
        self.assertTrue(result["identity_verified"])
        self.assertFalse(result["exact"])
        self.assertEqual(result["topk_set_mismatches"], 1)
        self.assertEqual(result["topk_symmetric_difference"], 2)
        self.assertEqual(result["compared_topk_entries"], 4)
        self.assertEqual(result["max_d_out"], 0.0)

    def test_disjoint_complete_topk_has_no_invented_delta(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        candidate[0]["top"] = [[lp, token + 100] for lp, token in candidate[0]["top"]]
        result = compare(reference, candidate)
        self.assertTrue(result["valid"])
        self.assertFalse(result["exact"])
        self.assertEqual(result["topk_symmetric_difference"], 10)
        self.assertEqual(result["compared_topk_entries"], 0)
        self.assertIsNone(result["max_d_out"])
        json.dumps(result, allow_nan=False)

    def test_topk_order_does_not_change_equal_token_maps(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        candidate[0]["top"].reverse()
        self.assertTrue(compare(reference, candidate)["exact"])

    def test_duplicate_topk_token_is_rejected_before_map_overwrites_it(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        candidate[0]["top"][-1][1] = candidate[0]["top"][0][1]
        self.assert_invalid(compare(reference, candidate), "duplicate token")
        self.assert_invalid(compare(candidate, reference), "duplicate token")

    def test_reference_and_candidate_request_errors_are_counted(self):
        reference = rows()
        candidate = copy.deepcopy(reference)
        reference[0] = {"error": "reference failed"}
        candidate[1] = {"error": "candidate failed"}
        result = compare(reference, candidate)
        self.assert_invalid(result, "request error")
        self.assertEqual(result["errors"], 2)
        self.assertIsNone(result["mean_d_prompt"])
        self.assertIsNone(result["max_d_out"])

    def test_nonfinite_logprobs_on_either_side_are_rejected(self):
        for nonfinite in (math.nan, math.inf, -math.inf):
            for side in ("reference", "candidate"):
                for field in ("top", "prompt_lp"):
                    with self.subTest(value=nonfinite, side=side, field=field):
                        reference = rows(1)
                        candidate = copy.deepcopy(reference)
                        target = reference if side == "reference" else candidate
                        if field == "top":
                            target[0][field][0][0] = nonfinite
                        else:
                            target[0][field][0] = nonfinite
                        self.assert_invalid(compare(reference, candidate), "finite number")

    def test_finite_inputs_that_overflow_delta_are_rejected(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        reference[0]["prompt_lp"][0] = -1e308
        candidate[0]["prompt_lp"][0] = 1e308
        self.assert_invalid(compare(reference, candidate), "non-finite delta")

    def test_mean_of_large_finite_deltas_does_not_overflow(self):
        reference = rows(1)
        candidate = copy.deepcopy(reference)
        reference[0]["prompt_lp"] = [0.0] * 3
        candidate[0]["prompt_lp"] = [sys.float_info.max] * 3
        result = compare(reference, candidate)
        self.assertTrue(result["valid"])
        self.assertEqual(result["mean_d_prompt"], sys.float_info.max)
        json.dumps(result, allow_nan=False)

    def test_empty_data_never_produces_a_success_or_nan(self):
        result = compare([], [])
        self.assert_invalid(result, "contain results")
        for metric in ("max_d_out", "mean_d_prompt", "p99_d_prompt", "max_d_prompt"):
            self.assertIsNone(result[metric])
        for field in ("top", "prompt_lp"):
            with self.subTest(field=field):
                candidate = rows(1)
                candidate[0][field] = []
                self.assert_invalid(compare(rows(1), candidate), field)

    def test_missing_fields_and_malformed_rows_fail_explicitly(self):
        for field in ("token", "top", "prompt_lp"):
            with self.subTest(field=field):
                candidate = rows(1)
                del candidate[0][field]
                self.assert_invalid(compare(rows(1), candidate), field)
        for malformed in (None, [], "row"):
            with self.subTest(row=malformed):
                self.assert_invalid(compare(rows(1), [malformed]), "must be an object")
        self.assert_invalid(compare(None, rows(1)), "ref must be a list")
        self.assert_invalid(compare(rows(1), {}), "got must be a list")

    def test_bad_topk_entries_and_non_numeric_values_are_rejected(self):
        changes = (
            ("token", True),
            ("token", -1),
            ("top", "not a list"),
            ("top", [[-1.0, 10, "extra"]] * 5),
            ("top", [["-1.0", token] for token in range(5)]),
            ("top", [[-1.0, True]] * 5),
            ("top", [[10**400, token] for token in range(5)]),
            ("prompt_lp", [False]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                candidate = rows(1)
                candidate[0][field] = value
                self.assert_invalid(compare(rows(1), candidate), field)

    def test_input_identity_mismatch_or_one_sided_metadata_is_rejected(self):
        for field in ("input_id", "prompt_positions"):
            with self.subTest(field=field):
                candidate = rows(1)
                del candidate[0][field]
                self.assert_invalid(compare(rows(1), candidate), f"{field} mismatch")
        candidate = rows(1)
        candidate[0]["input_id"] = "other-prompt"
        self.assert_invalid(compare(rows(1), candidate), "input_id mismatch")

    def test_wrong_prompt_position_identity_is_rejected_even_with_equal_values(self):
        candidate = rows(1)
        candidate[0]["prompt_positions"] = [27, 28, 29]
        self.assert_invalid(compare(rows(1), candidate), "prompt_positions mismatch")
        candidate[0]["prompt_positions"] = [19, 18, 17]
        self.assert_invalid(compare(rows(1), candidate), "prompt_positions mismatch")

    def test_malformed_identity_and_position_metadata_is_rejected(self):
        changes = (
            ("input_id", ""),
            ("input_id", None),
            ("input_id", 1),
            ("prompt_positions", None),
            ("prompt_positions", [17, 18]),
            ("prompt_positions", [17, 17, 19]),
            ("prompt_positions", [17, -1, 19]),
            ("prompt_positions", [17, True, 19]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                candidate = rows(1)
                candidate[0][field] = value
                self.assert_invalid(compare(rows(1), candidate), field)

    def test_matching_input_ids_without_position_metadata_are_not_fully_verified(self):
        reference = rows(1)
        del reference[0]["prompt_positions"]
        result = compare(reference, copy.deepcopy(reference))
        self.assertTrue(result["valid"])
        self.assertFalse(result["identity_verified"])

    def test_comparison_does_not_modify_its_inputs(self):
        reference = rows()
        candidate = copy.deepcopy(reference)
        before = copy.deepcopy((reference, candidate))
        compare(reference, candidate)
        self.assertEqual((reference, candidate), before)

    def test_custom_topk_is_explicit_and_positive(self):
        reference = rows(1)
        reference[0]["top"] = reference[0]["top"][:2]
        self.assertTrue(compare(reference, copy.deepcopy(reference), expected_topk=2)["exact"])
        for keyword in ("expected_topk", "expected_count"):
            for value in (0, -1, True, 1.5):
                with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                    compare(reference, reference, **{keyword: value})

    def test_import_is_standard_library_only(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "import acceptance_metrics; "
            "assert not any(name in sys.modules for name in ('torch', 'sglang', 'servers', 'psutil'))"
        )
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
