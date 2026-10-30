"""Required acceptance checks retain failure and partial-run evidence."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acceptance_report import Report


class TestAcceptanceReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asap_report_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    @staticmethod
    def add(report, cid, verdict=True):
        report.add(cid, f"check {cid}", "observed", "expected", verdict)

    def test_empty_report_is_failure_before_and_after_save(self):
        report = Report(str(self.root))
        self.assertFalse(report.ok())
        self.assertEqual(report.outcome(), "FAIL")
        report.save()
        self.assertEqual(report.outcome(), "FAIL")
        self.assertEqual(len(report.rows), 1)
        self.assertEqual(report.rows[0]["status"], "FAIL")
        self.assertEqual(json.loads((self.root / "summary.json").read_text())["outcome"], "FAIL")

    def test_missing_required_check_prevents_pass(self):
        report = Report(str(self.root), {"A", "B"})
        self.add(report, "A")
        self.assertEqual(report.missing(), {"B"})
        self.assertEqual(report.outcome(), "FAIL")
        self.assertFalse(report.ok())

    def test_complete_required_checks_pass(self):
        report = Report(str(self.root), {"A", "B"})
        self.add(report, "B")
        self.add(report, "A")
        self.assertEqual(report.missing(), set())
        self.assertEqual(report.outcome(), "PASS")
        self.assertTrue(report.ok())

    def test_explicit_skip_is_partial_not_a_complete_pass(self):
        report = Report(str(self.root), {"A", "B"})
        self.add(report, "A")
        self.add(report, "B", None)
        self.assertEqual(report.outcome(), "PARTIAL")
        self.assertFalse(report.ok())
        report.save()
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["skipped"], ["B"])
        self.assertEqual(summary["outcome"], "PARTIAL")

    def test_partial_scope_cannot_be_promoted_by_passing_rows(self):
        report = Report(str(self.root), {"A"}, partial=True)
        self.add(report, "A")
        self.assertEqual(report.outcome(), "PARTIAL")
        self.assertFalse(report.ok())
        report.save()
        self.assertTrue(json.loads((self.root / "summary.json").read_text())["partial_scope"])

    def test_failure_takes_precedence_over_skips_and_partial_scope(self):
        report = Report(str(self.root), {"A", "B"}, partial=True)
        self.add(report, "A", False)
        self.add(report, "B", None)
        self.assertEqual(report.outcome(), "FAIL")
        self.assertFalse(report.ok())

    def test_duplicate_id_stays_failed_if_caller_catches_exception(self):
        report = Report(str(self.root), {"A"})
        self.add(report, "A")
        with self.assertRaisesRegex(ValueError, "duplicate acceptance check ID: A"):
            self.add(report, "A")
        self.assertEqual(report.outcome(), "FAIL")
        self.assertFalse(report.ok())
        self.assertTrue(any(row["status"] == "FAIL" for row in report.rows))
        report.save()
        self.assertEqual(json.loads((self.root / "summary.json").read_text())["outcome"], "FAIL")

    def test_save_materializes_missing_checks_once_and_is_idempotent(self):
        report = Report(str(self.root), {"A", "B", "C"})
        self.add(report, "A")
        report.save()
        rows = json.loads((self.root / "report.json").read_text())
        self.assertEqual({row["id"]: row["status"] for row in rows}, {"A": "PASS", "B": "FAIL", "C": "FAIL"})
        self.assertEqual(report.missing(), set())
        self.assertEqual(report.outcome(), "FAIL")
        before = {name: (self.root / name).read_bytes() for name in ("report.json", "summary.json", "report.md")}
        report.save()
        after = {name: (self.root / name).read_bytes() for name in before}
        self.assertEqual(before, after)
        self.assertEqual(len(report.rows), 3)

    def test_json_rows_remain_a_list_and_summary_has_version_one(self):
        report = Report(str(self.root), {"B", "A"})
        self.add(report, "A")
        self.add(report, "B")
        report.save()
        rows = json.loads((self.root / "report.json").read_text())
        self.assertIsInstance(rows, list)
        self.assertEqual(set(rows[0]), {"id", "title", "measured", "criterion", "status"})
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary, {"schema_version": 1, "outcome": "PASS", "required": ["A", "B"],
                                   "partial_scope": False, "skipped": []})

    def test_markdown_escapes_table_separators_and_newlines(self):
        report = Report(str(self.root), {"A|1"})
        report.add("A|1", "title|part\nnext", "value|other\r\nnext", "criterion|part\nnext", True)
        report.save()
        markdown = (self.root / "report.md").read_text()
        self.assertIn("| A\\|1 | title\\|part<br>next | value\\|other<br>next | criterion\\|part<br>next | PASS |", markdown)
        self.assertNotIn("\r", markdown)
        self.assertEqual(len(markdown.splitlines()), 5)
        # JSON retains raw content for machine consumers instead of Markdown escapes.
        rows = json.loads((self.root / "report.json").read_text())
        self.assertEqual(rows[0]["title"], "title|part\nnext")


if __name__ == "__main__":
    unittest.main()
