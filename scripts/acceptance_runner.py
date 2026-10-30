"""Run unittest suites with structured outcomes and an owned process deadline.

The parent never derives verdicts from human-readable logs. Each child writes a
separate JSON result after unittest completes; missing or incomplete results fail
acceptance even when the child exits zero. Timeout cleanup only signals the new
session's process group on every exit path. Tests must not detach their subprocesses
from that group; ASAP_TEST_SUPERVISED tells service helpers to inherit it.
Supported platforms need POSIX waitid(WNOWAIT). Darwin additionally uses /bin/ps
to distinguish its empty-group EPERM from a live group's permission failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@dataclass
class SuiteResult:
    statuses: dict[str, str]
    returncode: int
    stdout: str
    elapsed_s: float
    problems: list[str]
    timed_out: bool

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.problems
            and bool(self.statuses)
            and all(status == "ok" for status in self.statuses.values())
        )


class _StructuredResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statuses: dict[str, str] = {}
        self.started: list[str] = []
        self.events: list[dict[str, str]] = []
        self.problems: list[str] = []

    def _record(self, test, status: str, kind: str, detail: str = "") -> None:
        # unittest passes a _SubTest to addSkip, but addSubTest supplies its parent.
        parent = getattr(test, "test_case", test)
        test_id = parent.id()
        previous = self.statuses.get(test_id)
        severity = lambda value: 3 if value == "ERROR" else 2 if value == "FAIL" else 1 if value.startswith("skipped") else 0
        if previous is None or severity(status) > severity(previous):
            self.statuses[test_id] = status
        self.events.append({"test_id": test_id, "event_id": test.id(), "kind": kind, "detail": detail})
        if status != "ok":
            self.problems.append(f"{kind}: {test.id()}: {detail}".rstrip(": "))

    def startTest(self, test):
        self.started.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        if test.id() not in self.statuses:
            self._record(test, "ERROR", "missing_outcome", "test stopped without a recorded outcome")
        super().stopTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "ok", "success")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL", "failure", str(err[1]))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", "error", str(err[1]))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, f"skipped: {reason}", "skip", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "FAIL", "expected_failure", str(err[1]))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "FAIL", "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
            self._record(subtest, status, "subtest_failure" if status == "FAIL" else "subtest_error", str(err[1]))


def _test_ids(suite) -> list[str]:
    if isinstance(suite, unittest.TestSuite):
        return [test_id for test in suite for test_id in _test_ids(test)]
    return [suite.id()]


def _run_child(names: list[str], result_path: str) -> int:
    payload = {
        "schema_version": 1, "completed": False, "planned_test_ids": [],
        "started_test_ids": [], "tests_run": 0, "statuses": {}, "events": [], "problems": [],
    }
    try:
        # Executing this file by absolute path otherwise puts scripts/, not cwd,
        # first on sys.path. Match `python -m unittest` invoked from the tests dir.
        sys.path.insert(0, os.getcwd())
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(names)
        planned = _test_ids(suite)
        payload["planned_test_ids"] = planned
        payload["problems"].extend(f"test loading failed: {error}" for error in loader.errors)
        if len(planned) != len(set(planned)):
            payload["problems"].append("duplicate full test IDs in requested suite")
        runner = unittest.TextTestRunner(verbosity=2, resultclass=_StructuredResult)
        result = runner.run(suite)
        payload.update(
            completed=True, statuses=result.statuses, started_test_ids=result.started,
            tests_run=result.testsRun, events=result.events,
        )
        payload["problems"].extend(result.problems)
    except BaseException as exc:
        payload["problems"].append(f"suite execution aborted: {type(exc).__name__}: {exc}")
    if not payload["planned_test_ids"]:
        payload["problems"].append("suite contains no tests")
    missing = sorted(set(payload["planned_test_ids"]) - set(payload["statuses"]))
    if missing:
        payload["problems"].append(f"tests without outcomes: {missing}")
    path = Path(result_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return 0 if payload["completed"] and not payload["problems"] else 1


def _read_result(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"structured suite result unavailable: {exc}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {}, ["invalid structured suite result schema"]
    statuses = payload.get("statuses")
    problems = payload.get("problems")
    planned = payload.get("planned_test_ids")
    started = payload.get("started_test_ids")
    events = payload.get("events")
    string_list = lambda value: isinstance(value, list) and all(isinstance(item, str) for item in value)
    if (
        not isinstance(statuses, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in statuses.items())
        or not string_list(problems) or not string_list(planned) or not string_list(started)
        or not isinstance(events, list) or not all(isinstance(event, dict) for event in events)
        or type(payload.get("tests_run")) is not int
    ):
        return {}, ["incomplete or malformed structured suite result"]
    problems = list(problems)
    if payload.get("completed") is not True:
        problems.append("suite did not complete")
    if not planned or not statuses:
        problems.append("suite contains no complete test outcomes")
    if len(planned) != len(set(planned)) or len(started) != len(set(started)):
        problems.append("duplicate full test IDs in structured result")
    if set(planned) != set(statuses):
        missing = sorted(set(planned) - set(statuses))
        unexpected = sorted(set(statuses) - set(planned))
        problems.append(f"outcome IDs do not match planned test IDs: missing={missing}, unexpected={unexpected}")
    if set(planned) != set(started) or payload["tests_run"] != len(started):
        problems.append("planned, started, and run test counts do not agree")
    if any(value not in ("ok", "FAIL", "ERROR") and not value.startswith("skipped") for value in statuses.values()):
        problems.append("unrecognized test outcome in structured result")
    return statuses, problems


def _wait_without_reaping(process: subprocess.Popen, timeout_s: float) -> bool:
    """Keep the session leader's PID reserved until group cleanup finishes."""
    deadline = time.monotonic() + timeout_s
    while True:
        observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if observed is not None and observed.si_pid == process.pid:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _stop_group(process: subprocess.Popen) -> None:
    """Clean only our unreaped session leader's group, then release its PID.

    No poll(), wait(), or communicate() may reap the leader before this function.
    Holding its PID prevents a newly created, unrelated group reusing the ID.
    """
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin reports EPERM for an empty group whose leader is a zombie.
            # Confirm absence; never treat a live group's permission error as a
            # successful cleanup. Its unreaped leader prevents PGID reuse here.
            if sys.platform != "darwin":
                raise
            members = subprocess.run(
                ["/bin/ps", "-o", "pid=,stat=", "-g", str(process.pid)],
                capture_output=True, text=True, timeout=1,
            )
            rows = [line.split() for line in members.stdout.splitlines()]
            if members.returncode not in (0, 1) or members.stderr or any(
                len(row) != 2 or not row[1].startswith("Z") for row in rows
            ):
                raise

    send(signal.SIGCONT)
    send(signal.SIGTERM)
    try:
        # Allow cooperative children to close resources before forced cleanup.
        time.sleep(0.1)
    finally:
        send(signal.SIGKILL)
        process.wait(timeout=0.5)


def run_suite(names: list[str], *, cwd: str, env: dict[str, str], log_path: str,
              timeout_s: float = 3600) -> SuiteResult:
    """Run requested test IDs; preserve raw logs and a structured result sidecar.

    `log_path + '.result.json'` preserves the child's raw structured result when
    available, otherwise an explicit parent-side missing-result record. A child
    exit of zero alone never makes SuiteResult.ok true.
    """
    if os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED")):
        raise RuntimeError("acceptance runner requires POSIX sessions and waitid(WNOWAIT)")
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("names must contain at least one nonempty unittest name")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="asap_suite_") as directory:
        result_path = Path(directory) / "result.json"
        stdout_path = Path(directory) / "stdout.log"
        stderr_path = Path(directory) / "stderr.log"
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--result-file", str(result_path), *names]
        child_env = dict(env, ASAP_TEST_SUPERVISED="1")
        # Files cannot keep communicate() blocked when a service inherits stdout.
        # More importantly, waitid observes completion without releasing our PID.
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                argv, cwd=cwd, env=child_env, stdout=stdout_file, stderr=stderr_file,
                start_new_session=True,
            )
            try:
                if not _wait_without_reaping(process, timeout_s):
                    timed_out = True
                    problems.append(f"suite deadline exceeded ({timeout_s:g} seconds)")
            finally:
                _stop_group(process)
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        statuses, structured_problems = _read_result(result_path)
        problems.extend(structured_problems)
        if process.returncode != 0:
            problems.append(f"test process exited with code {process.returncode}")
        if result_path.exists():
            Path(str(log) + ".result.json").write_bytes(result_path.read_bytes())
        else:
            # Do not leave an earlier successful sidecar next to a failed rerun.
            missing_result = {"origin": "parent", "completed": False, "result_unavailable": True,
                              "returncode": process.returncode, "timed_out": timed_out, "problems": problems}
            Path(str(log) + ".result.json").write_text(
                json.dumps(missing_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.write_text(stderr + "\n--- stdout ---\n" + stdout, encoding="utf-8")
    return SuiteResult(statuses, process.returncode, stdout, time.monotonic() - start, problems, timed_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    return _run_child(args.names, args.result_file)


if __name__ == "__main__":
    raise SystemExit(main())
