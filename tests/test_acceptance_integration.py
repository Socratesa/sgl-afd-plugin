"""CPU tests of the real CLI wiring, using explicit service doubles."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance as ac
from acceptance_runner import SuiteResult


class TestAcceptanceIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def cli(self, *args):
        return ac.main([*args, "--out", str(self.output)])

    def summary(self):
        return json.loads((self.output / "summary.json").read_text())

    def rows(self):
        return {row["id"]: row for row in json.loads((self.output / "report.json").read_text())}

    def response(self, ids, tail=256):
        start = max(0, len(ids) - tail)
        return {"meta_info": {
            "output_token_logprobs": [[-0.1, 7]],
            "output_top_logprobs": [[[-0.1 * i, i + 6] for i in range(1, 6)]],
            "input_token_logprobs": [[None if i == start else -0.5, ids[i]] for i in range(start, len(ids))],
        }}

    def collected(self, ids, tail=256):
        with patch.object(ac.sv, "generate", return_value=(True, 0.1, self.response(ids, tail))):
            return ac.collect(1, [ids], tail=tail)

    def test_collection_records_expected_token_suffix_and_positions(self):
        ids = list(range(300))
        rows = self.collected(ids)
        self.assertEqual(rows[0]["prompt_positions"], list(range(45, 300)))
        self.assertEqual(len(rows[0]["prompt_lp"]), 255)
        self.assertTrue(ac.compare_results(rows, rows, [ids])["exact"])

    def test_collect_rejects_truncated_wrong_token_and_unexpected_none(self):
        ids = [1, 2, 3]
        for malformed in ([[None, 1], [-0.5, 2]], [[None, 1], [-0.5, 9], [-0.5, 3]],
                          [[None, 1], [None, 2], [-0.5, 3]], [[-0.5, 1], [-0.5, 2], [-0.5, 3]]):
            with self.subTest(raw=malformed):
                response = self.response(ids)
                response["meta_info"]["input_token_logprobs"] = malformed
                with patch.object(ac.sv, "generate", return_value=(True, 0.1, response)):
                    self.assertIn("error", ac.collect(1, [ids])[0])

    def test_collection_rejects_missing_or_multiple_output_positions(self):
        response = self.response([1, 2, 3])
        for raw in ({}, {"meta_info": {}}, response):
            response["meta_info"]["output_token_logprobs"] *= 2
            with patch.object(ac.sv, "generate", return_value=(True, 0.1, raw)):
                self.assertIn("error", ac.collect(1, [[1, 2, 3]])[0])

    def test_equal_but_jointly_truncated_positions_fail_against_requested_inputs(self):
        ids = [1, 2, 3]
        rows = self.collected(ids)
        rows[0]["prompt_lp"] = rows[0]["prompt_lp"][:1]
        rows[0]["prompt_positions"] = rows[0]["prompt_positions"][:1]
        self.assertTrue(ac.compare(rows, rows)["exact"])
        self.assertFalse(ac.compare_results(rows, rows, [ids])["valid"])

    def test_legacy_and_wrong_input_cache_cannot_supply_formal_reference(self):
        ids = [1, 2, 3]
        for rows in (self.collected(ids), self.collected(ids)):
            rows[0].pop("input_id")
            self.assertFalse(ac.compare_results(rows, rows, [ids])["valid"])
        rows = self.collected(ids)
        self.assertFalse(ac.compare_results(rows, rows, [[3, 2, 1]])["valid"])

    def test_reference_failure_is_saved_and_prevents_afd_launch(self):
        def failed_reference(rep, *args):
            rep.add("R0", "bad reference", "request failed", "all successful", False)
            return None, None

        with patch.object(ac, "build_prompts", return_value=[[1, 2, 3]]), \
             patch.object(ac, "reference", side_effect=failed_reference), \
             patch.object(ac, "start_afd") as launch:
            self.assertEqual(self.cli("e2e", "--replicas", "1", "--prompts", "1"), 1)
        launch.assert_not_called()
        self.assertEqual(self.rows()["R0"]["status"], "FAIL")
        self.assertEqual(self.rows()["A1"]["status"], "FAIL")

    def test_invalid_topology_fails_before_any_stage_or_prompt_loading(self):
        cases = (("--replicas", "4"), ("--replicas", "0"), ("--replicas", "1,1"),
                 ("--replicas", "2"), ("--f-gpus", "0"), ("--a-gpus", "0,0"))
        for flags in cases:
            with self.subTest(flags=flags), patch.object(ac, "stage_e2e") as stage:
                self.assertEqual(self.cli("e2e", *flags), 1)
                stage.assert_not_called()
                self.assertEqual(self.summary()["outcome"], "FAIL")

    def test_stage_exception_and_keyboard_interrupt_always_save_failures(self):
        for error in (RuntimeError("stage failed"), KeyboardInterrupt()):
            with patch.object(ac, "stage_pr04", side_effect=error):
                self.assertEqual(self.cli("pr04"), 1)
            self.assertEqual(set(self.rows()), {"A1", "A2", "U", "EXCEPTION"})
            self.assertTrue((self.output / "exception.log").exists())

    def test_empty_or_incomplete_stage_cannot_pass(self):
        for completed in ((), ("A1",)):
            def stage(rep, args):
                for cid in completed:
                    rep.add(cid, cid, "ok", "ok", True)
            with patch.object(ac, "stage_pr04", side_effect=stage):
                self.assertEqual(self.cli("pr04"), 1)

    def test_complete_stage_preserves_zero_success_exit(self):
        def stage(rep, args):
            for cid in ac.required_checks(args):
                rep.add(cid, cid, "ok", "ok", True)
        with patch.object(ac, "stage_pr04", side_effect=stage):
            self.assertEqual(self.cli("pr04"), 0)
        self.assertEqual(self.summary()["outcome"], "PASS")

    def test_reference_only_is_partial_even_with_successful_reference(self):
        def reference(rep, *args):
            rep.add("R0", "reference", "ok", "ok", True)
            return [{}], {}
        with patch.object(ac, "build_prompts", return_value=[[1, 2, 3]]), \
             patch.object(ac, "reference", side_effect=reference):
            self.assertEqual(self.cli("e2e", "--replicas", "", "--prompts", "1"), 2)
        self.assertEqual(self.summary()["outcome"], "PARTIAL")

    def test_cpu_only_m0_marks_every_gpu_check_skipped(self):
        suite = SuiteResult({f"module.C.test_{i}": "ok" for i in range(28)}, 0, "", 1, [], False)
        ep = type("EntryPoint", (), {"name": "sgl_asap_plugin", "value": "sgl_asap_plugin:register"})()
        child = SimpleNamespace(stdout="0.01 False False", returncode=0)
        with patch.object(ac, "run_tests", return_value=suite), \
             patch.object(ac.importlib.metadata, "entry_points", return_value=[ep]), \
             patch.object(ac.subprocess, "run", return_value=child):
            self.assertEqual(self.cli("m0", "--skip-gpu"), 2)
        self.assertEqual(self.summary()["skipped"], ["G0", "G1", "G2", "G3", "G4", "N1"])

    def test_full_ids_are_not_collapsed_and_suite_failure_controls_gate(self):
        checks = [("A1", "test", ["test_same"], "ok")]
        for suite in (SuiteResult({"x.A.test_same": "ok", "x.B.test_same": "ok"}, 0, "", 1, [], False),
                      SuiteResult({"x.A.test_same": "ok"}, 7, "", 1, [], False)):
            report = ac.Report(str(self.output), {"A1", "U"})
            ac.add_mapped(report, checks, suite, min_tests=1)
            self.assertFalse(report.ok())

    def test_multi_replica_uses_current_run_baseline_and_runs_one_first(self):
        baseline = [{"source": "this run"}]
        calls = []
        def one(rep, *args):
            calls.append(1)
            return baseline
        def multi(rep, args, d, a, f, prompts, received):
            calls.append(d)
            self.assertIs(received, baseline)
        with patch.object(ac, "build_prompts", return_value=[list(range(33000))]), \
             patch.object(ac, "phase_one_replica", side_effect=one), \
             patch.object(ac, "phase_replicas", side_effect=multi):
            self.cli("e2e", "--replicas", "2,1", "--prompts", "1", "--skip-reference")
        self.assertEqual(calls, [1, 2])

    def test_background_exception_and_no_background_work_fail(self):
        def request(ids, rank):
            if rank == 0:
                raise RuntimeError("long failed")
            return True, 0.1, {}
        for longs in ([1], []):
            result = ac.run_load([1, 2], longs, request, short_count=1, startup_timeout_s=0.01)
            self.assertFalse(result["assessment"]["passed"])
            json.dumps(result, allow_nan=False)

    def test_actual_thread_records_overlapping_requests_and_stops(self):
        entered, loaded_finished = threading.Event(), threading.Event()
        short_calls = 0
        def request(ids, rank):
            nonlocal short_calls
            if rank == 0:
                entered.set()
                self.assertTrue(loaded_finished.wait(2))
                return True, 2.0, {}
            short_calls += 1
            if short_calls > 1:
                self.assertTrue(entered.wait(2))
                loaded_finished.set()
            return True, 0.1, {}
        clocks = {threading.current_thread().ident: iter((0, 1, 3, 4))}
        def monotonic():
            return next(clocks.setdefault(threading.current_thread().ident, iter((2, 10))))
        with patch.object(ac, "time", SimpleNamespace(monotonic=monotonic)):
            result = ac.run_load([1, 2], [3], request, short_count=1, join_timeout_s=2)
        self.assertTrue(result["assessment"]["passed"], result)
        self.assertEqual(len(result["long"]), 1)
        self.assertEqual(result["long"][0]["elapsed_s"], 8)

    def test_reported_wall_clock_duration_cannot_inflate_long_request(self):
        clocks = {threading.current_thread().ident: iter((0, 0.1, 2.01, 2.11))}
        def monotonic():
            return next(clocks.setdefault(threading.current_thread().ident, iter((2, 2.12))))
        def request(ids, rank):
            return True, 1000 if rank == 0 else 0.1, {}
        with patch.object(ac, "time", SimpleNamespace(monotonic=monotonic)):
            result = ac.run_load([1, 2], [3], request, short_count=1)
        self.assertTrue(result["assessment"]["valid"], result)
        self.assertFalse(result["assessment"]["passed"])
        self.assertAlmostEqual(result["assessment"]["long_p50_s"], 0.12)

    def test_background_system_exit_after_a_success_is_not_lost(self):
        second_long = threading.Event()
        short_calls, long_calls = 0, 0
        clocks = {threading.current_thread().ident: iter((0, 1, 3, 4))}
        def monotonic():
            return next(clocks.setdefault(threading.current_thread().ident, iter((2, 10, 11))))
        def request(ids, rank):
            nonlocal short_calls, long_calls
            if rank == 0:
                long_calls += 1
                if long_calls == 2:
                    second_long.set()
                    raise SystemExit("background aborted")
            else:
                short_calls += 1
                if short_calls == 2:
                    self.assertTrue(second_long.wait(2))
            return True, 1.0, {}
        with patch.object(ac, "time", SimpleNamespace(monotonic=monotonic)):
            result = ac.run_load([1, 2], [3, 4], request, short_count=1)
        self.assertFalse(result["assessment"]["valid"])
        self.assertFalse(result["assessment"]["passed"])
        self.assertIn("SystemExit", str(result["assessment"]["problems"]))

    def test_collection_rejects_nonfinite_data_before_writing_json(self):
        response = self.response([1, 2, 3])
        response["meta_info"]["input_token_logprobs"][1][0] = float("nan")
        with patch.object(ac.sv, "generate", return_value=(True, 0.1, response)):
            rows = ac.collect(1, [[1, 2, 3]])
        self.assertIn("error", rows[0])
        json.dumps(rows, allow_nan=False)

    def test_collection_rejects_nonfinite_generated_logprob(self):
        response = self.response([1, 2, 3])
        response["meta_info"]["output_token_logprobs"][0][0] = float("nan")
        with patch.object(ac.sv, "generate", return_value=(True, 0.1, response)):
            self.assertIn("error", ac.collect(1, [[1, 2, 3]])[0])

    def services(self, replicas, stats=None):
        from unittest.mock import Mock
        if stats is None:
            stats = {0: {"layers_served": 43, "jobs": 10, "lost_messages": 0, "replica_switches": 2,
                         "trace_tail": [(0, 1, 1), (1, 1, 2)]}}
        ffn = SimpleNamespace(
            reports=lambda kind: {0: {"weights_gb": 147.2, "total_s": 1}} if kind == "ready" else stats,
            log_path="ffn.log", stop=Mock(),
        )
        attn = SimpleNamespace(
            attn_ranks=lambda: [{"replica": str(i), "world": "1", "ok": "9", "total": "9"}
                                for i in range(replicas)],
            log_text=lambda: "[ASAP] server args ok: role=attn afd=True hooks=9/9\n"
                             "[ASAP] A rank 0 connected to 1 F ranks\nLoad weight end. mem usage=10 GB",
            log_path="attn.log", stop=Mock(),
        )
        return ffn, attn

    def phase_args(self):
        return SimpleNamespace(out=str(self.output), port=1, startup_timeout=2, max_prompt=128,
                               soak_requests=1, short_requests=1, long_len=1024)

    def test_single_replica_checks_exact_request_sizes_and_complete_results(self):
        prompts = [[1, 2, 3]]
        rows = self.collected(prompts[0])
        noise = ac.compare_results(rows, rows, prompts)
        ffn, attn = self.services(1)
        report = ac.Report(str(self.output), {*(f"E{i}" for i in range(1, 7)), "A1", "A2", "A3"})
        with patch.object(ac, "start_afd", return_value=(ffn, attn, True, 1)), \
             patch.object(ac, "collect", return_value=rows), \
             patch.object(ac, "soak", return_value=(0, 0.1, 1)), \
             patch.object(ac.sv, "generate", return_value=(True, 0.1, {})) as generate:
            got = ac.phase_one_replica(report, self.phase_args(), [0], [1], prompts, rows, noise)
        self.assertIs(got, rows)
        self.assertTrue(report.ok(), report.rows)
        self.assertEqual([len(call.kwargs["input_ids"]) for call in generate.call_args_list],
                         [10, 500, 3000, 6000, 16000])
        ffn.stop.assert_called_once()
        attn.stop.assert_called_once()

    def test_multi_replica_rejects_missing_baseline_and_missing_rank_statistics(self):
        ffn, attn = self.services(2)
        report = ac.Report(str(self.output), {f"M2.{i}" for i in range(5)})
        with patch.object(ac, "start_afd", return_value=(ffn, attn, True, 1)), \
             patch.object(ac, "soak", return_value=(0, 0.1, 1)), \
             patch.object(ac, "run_load", return_value={"assessment": {"passed": True}}):
            ac.phase_replicas(report, self.phase_args(), 2, [0, 1], [2, 3], [list(range(4096))], None)
        by_id = {row["id"]: row["status"] for row in report.rows}
        self.assertEqual(by_id["M2.1"], "FAIL")
        self.assertEqual(by_id["M2.4"], "FAIL")
        ffn.stop.assert_called_once()
        attn.stop.assert_called_once()

    def test_missing_psutil_fails_before_server_process_creation(self):
        with patch.object(ac.sv.importlib, "import_module", side_effect=ImportError("missing")), \
             patch.object(ac.sv.subprocess, "Popen") as launch:
            server = ac.sv.Server("fake", ["unused"], {}, str(self.output), 1)
            with self.assertRaisesRegex(RuntimeError, "psutil"):
                server.start()
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
