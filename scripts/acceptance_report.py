"""Required-check accounting, independent of GPU services and test execution."""

from __future__ import annotations

import json
from pathlib import Path


class Report:
    def __init__(self, out: str, required: set[str] | None = None, *, partial: bool = False):
        self.out = Path(out)
        self.required = set(required or ())
        self.partial = partial
        self.rows: list[dict[str, str]] = []

    def add(self, cid: str, title: str, measured: object, criterion: str, ok: bool | None) -> None:
        if any(row["id"] == cid for row in self.rows):
            # Retain the invalidity even if the caller catches the exception.
            self.rows.append({"id": "REPORT_ERROR", "title": "duplicate check ID",
                              "measured": cid, "criterion": "unique IDs", "status": "FAIL"})
            raise ValueError(f"duplicate acceptance check ID: {cid}")
        if ok is not None and not isinstance(ok, bool):
            raise TypeError(f"{cid}: verdict must be bool or None")
        status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        self.rows.append({"id": cid, "title": title, "measured": str(measured),
                          "criterion": criterion, "status": status})
        print(f"[{status}] {cid} {title}: {measured}   (criterion: {criterion})", flush=True)

    def missing(self) -> set[str]:
        return self.required - {row["id"] for row in self.rows}

    def outcome(self) -> str:
        if not self.rows or self.missing() or any(row["status"] == "FAIL" for row in self.rows):
            return "FAIL"
        if self.partial or any(row["status"] == "SKIP" for row in self.rows):
            return "PARTIAL"
        return "PASS"

    def ok(self) -> bool:
        return self.outcome() == "PASS"

    def save(self) -> None:
        for cid in sorted(self.missing()):
            self.add(cid, "required check not completed", "not run", "must execute", False)
        if not self.rows:
            self.add("REPORT_ERROR", "empty report", "0 checks", "at least one check", False)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "report.json").write_text(
            json.dumps(self.rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        summary = {"schema_version": 1, "outcome": self.outcome(), "required": sorted(self.required),
                   "partial_scope": self.partial,
                   "skipped": [row["id"] for row in self.rows if row["status"] == "SKIP"]}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        escape = lambda value: value.replace("|", "\\|").replace("\n", "<br>").replace("\r", "")
        lines = [f"Outcome: {self.outcome()}", "", "| ID | Check | Measured | Criterion | Result |",
                 "|---|---|---|---|---|"]
        lines.extend("| " + " | ".join(escape(row[key]) for key in
                     ("id", "title", "measured", "criterion", "status")) + " |" for row in self.rows)
        (self.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
