"""Stdlib subprocess regressions for trustworthy acceptance suite execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance_runner
from acceptance_runner import run_suite


@unittest.skipUnless(os.name == "posix", "acceptance runner requires POSIX sessions")
class TestAcceptanceRunner(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asap_runner_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        self.counter = 0

    def run_fixture(self, source, *, names=None, timeout_s=10):
        self.counter += 1
        name = f"fixture_{self.counter}"
        (self.root / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
        log = self.root / f"{name}.log"
        result = run_suite(names or [name], cwd=str(self.root), env=self.env,
                           log_path=str(log), timeout_s=timeout_s)
        self.assertTrue(log.is_file())
        return result, log, name

    def test_success_and_same_method_names_are_counted_by_full_id(self):
        result, log, name = self.run_fixture("""
            import unittest
            class First(unittest.TestCase):
                def test_same(self):
                    print('ordinary stdout')
            class Second(unittest.TestCase):
                def test_same(self):
                    pass
        """)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.statuses, {f"{name}.First.test_same": "ok", f"{name}.Second.test_same": "ok"})
        self.assertIn("ordinary stdout", result.stdout)
        payload = json.loads(Path(str(log) + ".result.json").read_text())
        self.assertEqual(payload["tests_run"], 2)
        self.assertTrue(payload["completed"])

    def test_supervision_flag_is_set_only_in_child_environment(self):
        self.env["ASAP_TEST_SUPERVISED"] = "0"
        result, _, _ = self.run_fixture("""
            import os, unittest
            class Cases(unittest.TestCase):
                def test_supervised(self):
                    self.assertEqual(os.environ['ASAP_TEST_SUPERVISED'], '1')
        """)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(self.env["ASAP_TEST_SUPERVISED"], "0")

    def test_required_platform_capability_is_checked_before_launch(self):
        with patch("acceptance_runner.os.name", "nt"), patch("acceptance_runner.subprocess.Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "waitid"):
                run_suite(["some_test"], cwd=str(self.root), env=self.env,
                          log_path=str(self.root / "unsupported.log"))
        launch.assert_not_called()

    def test_exited_leader_is_not_reaped_until_owned_group_cleanup(self):
        cleanup = acceptance_runner._stop_group
        observed_pids = []

        def observe_cleanup(process):
            try:
                self.assertIsNone(process.returncode)
                observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                self.assertIsNotNone(observed, "leader was reaped before group cleanup")
                self.assertEqual(observed.si_pid, process.pid)
                observed_pids.append(process.pid)
            finally:
                cleanup(process)

        with patch("acceptance_runner._stop_group", side_effect=observe_cleanup):
            result, _, _ = self.run_fixture("""
                import unittest
                class Cases(unittest.TestCase):
                    def test_ok(self):
                        pass
            """)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(len(observed_pids), 1)

    def test_subtest_failure_and_error_do_not_become_success(self):
        result, log, name = self.run_fixture("""
            import unittest
            class Cases(unittest.TestCase):
                def test_nested(self):
                    with self.subTest(value=1):
                        self.fail('assertion failed')
                    with self.subTest(value=2):
                        raise RuntimeError('subtest crashed')
        """)
        self.assertFalse(result.ok)
        self.assertEqual(result.statuses[f"{name}.Cases.test_nested"], "ERROR")
        events = json.loads(Path(str(log) + ".result.json").read_text())["events"]
        self.assertEqual([event["kind"] for event in events], ["subtest_failure", "subtest_error"])
        self.assertEqual(len(result.statuses), 1)

    def test_class_setup_and_teardown_errors_fail_completed_suites(self):
        for hook in ("setUpClass", "tearDownClass"):
            with self.subTest(hook=hook):
                result, log, name = self.run_fixture(f"""
                    import unittest
                    class Cases(unittest.TestCase):
                        @classmethod
                        def {hook}(cls):
                            raise RuntimeError('fixture failed')
                        def test_ok(self):
                            pass
                """)
                self.assertFalse(result.ok)
                self.assertIn("ERROR", result.statuses.values())
                self.assertTrue(any(hook in problem for problem in result.problems))
                self.assertTrue(json.loads(Path(str(log) + ".result.json").read_text())["completed"])
                if hook == "tearDownClass":
                    self.assertEqual(result.statuses[f"{name}.Cases.test_ok"], "ok")

    def test_module_fixture_errors_are_preserved(self):
        for hook in ("setUpModule", "tearDownModule"):
            with self.subTest(hook=hook):
                result, _, _ = self.run_fixture(f"""
                    import unittest
                    def {hook}():
                        raise RuntimeError('module fixture failed')
                    class Cases(unittest.TestCase):
                        def test_ok(self):
                            pass
                """)
                self.assertFalse(result.ok)
                self.assertTrue(any(hook in problem for problem in result.problems))

    def test_regular_and_subtest_skips_fail_acceptance(self):
        result, _, name = self.run_fixture("""
            import unittest
            class Cases(unittest.TestCase):
                @unittest.skip('unavailable')
                def test_skipped(self):
                    pass
                def test_subtest(self):
                    with self.subTest(value=1):
                        self.skipTest('subtest unavailable')
                    with self.subTest(value=2):
                        pass
        """)
        self.assertFalse(result.ok)
        self.assertTrue(result.statuses[f"{name}.Cases.test_skipped"].startswith("skipped"))
        self.assertTrue(result.statuses[f"{name}.Cases.test_subtest"].startswith("skipped"))

    def test_setup_class_skip_is_not_an_empty_pass(self):
        result, _, _ = self.run_fixture("""
            import unittest
            class Cases(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise unittest.SkipTest('class unavailable')
                def test_ok(self):
                    pass
        """)
        self.assertFalse(result.ok)
        self.assertTrue(any(status.startswith("skipped") for status in result.statuses.values()))

    def test_expected_failure_and_unexpected_success_are_explicit(self):
        result, log, _ = self.run_fixture("""
            import unittest
            class Cases(unittest.TestCase):
                @unittest.expectedFailure
                def test_known_failure(self):
                    self.fail('known failure')
                @unittest.expectedFailure
                def test_unexpected_success(self):
                    pass
        """)
        self.assertFalse(result.ok)
        kinds = {event["kind"] for event in json.loads(Path(str(log) + ".result.json").read_text())["events"]}
        self.assertEqual(kinds, {"expected_failure", "unexpected_success"})

    def test_loading_error_is_a_problem(self):
        result, _, _ = self.run_fixture("", names=["module_that_does_not_exist_asap"])
        self.assertFalse(result.ok)
        self.assertTrue(any("test loading failed" in problem for problem in result.problems))

    def test_empty_suite_cannot_pass(self):
        result, _, _ = self.run_fixture("import unittest\n")
        self.assertFalse(result.ok)
        self.assertTrue(any("no tests" in problem for problem in result.problems))

    def test_duplicate_full_test_id_cannot_hide_a_second_execution(self):
        result, _, _ = self.run_fixture("""
            import unittest
            class Cases(unittest.TestCase):
                def test_ok(self):
                    pass
        """, names=["fixture_1.Cases.test_ok", "fixture_1.Cases.test_ok"])
        self.assertFalse(result.ok)
        self.assertTrue(any("duplicate full test IDs" in problem for problem in result.problems))

    def test_unplanned_success_outcome_cannot_inflate_test_count(self):
        result, log, name = self.run_fixture("""
            import unittest
            class Real(unittest.TestCase):
                def test_real(self):
                    class Ghost(unittest.TestCase):
                        def test_ghost(self):
                            pass
                    self._outcome.result.addSuccess(Ghost('test_ghost'))
        """)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.statuses), 2)
        self.assertTrue(any("outcome IDs do not match planned test IDs" in problem for problem in result.problems))
        raw = json.loads(Path(str(log) + ".result.json").read_text())
        self.assertEqual(raw["planned_test_ids"], [f"{name}.Real.test_real"])
        self.assertEqual(raw["tests_run"], 1)

    def test_process_exit_requires_complete_structured_result_even_at_zero(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                result, log, _ = self.run_fixture(f"""
                    import os, unittest
                    class Cases(unittest.TestCase):
                        def test_exit(self):
                            print('before abnormal exit', flush=True)
                            os._exit({exit_code})
                """)
                self.assertEqual(result.returncode, exit_code)
                self.assertFalse(result.ok)
                self.assertIn("before abnormal exit", result.stdout)
                self.assertIn("before abnormal exit", log.read_text())
                self.assertTrue(any("structured suite result unavailable" in problem for problem in result.problems))
                missing = json.loads(Path(str(log) + ".result.json").read_text())
                self.assertTrue(missing["result_unavailable"])
                self.assertEqual(missing["returncode"], exit_code)

    def test_timeout_reaps_a_single_process_that_accepts_term(self):
        result, _, _ = self.run_fixture("""
            import time, unittest
            class Cases(unittest.TestCase):
                def test_hang(self):
                    print('waiting for deadline', flush=True)
                    time.sleep(60)
        """, timeout_s=0.2)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)
        self.assertIn("waiting for deadline", result.stdout)
        self.assertIsInstance(result.returncode, int)
        self.assertLess(result.elapsed_s, 2)

    def test_timeout_cleans_owned_group_without_touching_another_session(self):
        import fcntl

        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

        def stop_sentinel():
            if sentinel.poll() is None:
                sentinel.terminate()
            sentinel.wait(timeout=2)

        self.addCleanup(stop_sentinel)
        result, _, _ = self.run_fixture("""
            import os, subprocess, sys, time, unittest
            class Cases(unittest.TestCase):
                def test_hang(self):
                    child = subprocess.Popen([sys.executable, '-u', '-c',
                        'import fcntl,signal,time; lock=open("child.lock", "w"); fcntl.flock(lock,fcntl.LOCK_EX); signal.signal(signal.SIGTERM, signal.SIG_IGN); print("child ready", flush=True); time.sleep(60)'])
                    print('child_pid=' + str(child.pid), flush=True)
                    time.sleep(60)
        """, timeout_s=0.4)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)
        self.assertLess(result.elapsed_s, 3)
        self.assertIsNone(sentinel.poll(), "unrelated process was signaled")
        self.assertIn("child ready", result.stdout)
        # Lock release proves the TERM-ignoring child exited, including when its
        # PID briefly remains a zombie. It needs no ps/psutil or process inventory.
        with (self.root / "child.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def check_real_ffn_helper_cleanup(self, action, *, timeout_s=10):
        """Exercise the actual test_gpu helper with a stdlib service stand-in."""
        import fcntl

        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

        def stop_sentinel():
            if sentinel.poll() is None:
                sentinel.terminate()
            sentinel.wait(timeout=2)

        self.addCleanup(stop_sentinel)
        repo = Path(__file__).resolve().parents[1]
        # The service closes stdout/stderr after acquiring its lock. A completed
        # suite therefore cannot mistake closed streams for child termination.
        service_source = (
            'import fcntl,os,signal,time; '
            'lock=open("ffn.lock", "w"); fcntl.flock(lock,fcntl.LOCK_EX); '
            'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            'print("[ASAP] ffn ready: ranks=1",flush=True); '
            'os.close(1); os.close(2); time.sleep(60)'
        )
        result, log, name = self.run_fixture(f"""
            import os, sys, time, unittest
            sys.path[:0] = {[str(repo / 'tests'), str(repo / 'python')]!r}
            # This exercises process supervision without importing torch/CUDA.
            sys.modules['torch'] = None
            from test_gpu import FfnService
            class Cases(unittest.TestCase):
                def test_helper(self):
                    svc = FfnService(1, 1, 1, 0, 9710)
                    svc.log_path = 'ffn.log'
                    svc.argv = [sys.executable, '-u', '-c', {service_source!r}]
                    svc.start(timeout=5)
                    self.assertTrue(svc.supervised)
                    self.assertEqual(os.getpgid(svc.proc.pid), os.getpgrp())
                    print('actual FfnService ready', flush=True)
                    {action}
        """, timeout_s=timeout_s)
        self.assertIn("actual FfnService ready", result.stdout)
        self.assertIsNone(sentinel.poll(), "unrelated session was signaled")
        deadline = time.monotonic() + 2
        with (self.root / "ffn.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self.fail("supervised FfnService survived runner cleanup")
                    time.sleep(0.01)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return result, log, name

    def test_real_ffn_helper_is_reaped_on_suite_timeout(self):
        result, _, _ = self.check_real_ffn_helper_cleanup("time.sleep(60)", timeout_s=2)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)
        self.assertLess(result.elapsed_s, 4)

    def test_real_ffn_helper_is_reaped_after_success_with_closed_stdout(self):
        result, _, _ = self.check_real_ffn_helper_cleanup("pass")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.ok, result.problems)

    def test_real_ffn_helper_stop_timeout_records_error_before_group_cleanup(self):
        result, log, name = self.check_real_ffn_helper_cleanup("svc.stop(timeout=0.05)")
        self.assertFalse(result.timed_out)
        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.statuses, {f"{name}.Cases.test_helper": "ERROR"})
        payload = json.loads(Path(str(log) + ".result.json").read_text())
        self.assertTrue(payload["completed"])
        self.assertTrue(any("stop exceeded" in problem for problem in payload["problems"]))


if __name__ == "__main__":
    unittest.main()
