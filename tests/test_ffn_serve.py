import json
import os
import signal
import subprocess
import sys
import unittest

from _util import HAS_MODEL, MODEL_DIR

LAYOUT_PREFIX = "[ASAP] ffn layout "


def ffn_cmd(*extra):
    return [sys.executable, "-m", "sgl_asap_plugin.ffn_serve", "--model-path", MODEL_DIR, *extra]


def child_env():
    env = dict(os.environ)
    env.pop("ASAP_ROLE", None)
    return env


def dry_run(*extra):
    return subprocess.run(ffn_cmd("--asap-dry-run", *extra), capture_output=True, text=True, timeout=900,
                          env=child_env())


def layout_of(proc):
    lines = [l for l in proc.stdout.splitlines() if l.startswith(LAYOUT_PREFIX)]
    assert lines, f"no layout line:\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-1500:]}"
    return json.loads(lines[-1][len(LAYOUT_PREFIX):])


@unittest.skipUnless(HAS_MODEL, "needs the DeepSeek-V4-Flash config on this machine")
class TestFfnServe(unittest.TestCase):
    def test_layout_for_four_f_ranks_matches_the_design_doc(self):
        proc = dry_run("--tp-size", "4", "--ep-size", "4", "--asap-num-replicas", "4", "--asap-replica-tp", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        s = layout_of(proc)
        self.assertEqual((s["n_experts"], s["padded_experts"], s["experts_per_rank"]), (256, 256, 64))
        self.assertEqual(s["expert_ranges"], [[0, 64], [64, 128], [128, 192], [192, 256]])
        self.assertEqual((s["moe"]["hidden"], s["moe"]["topk"], s["moe"]["layers"]), (4096, 6, 43))
        self.assertEqual(s["arena"]["f_recv_arena_bytes"], 1_080_033_280)  # 02b section 6: ~1.08GB
        self.assertEqual(s["arena"]["a_result_arena_bytes"], 1_075_838_976)  # ~1.07GB, 2 MiB of it done flags

    def test_three_f_ranks_need_two_dummy_experts(self):
        proc = dry_run("--tp-size", "3", "--ep-size", "3", "--asap-num-replicas", "1", "--asap-replica-tp", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        s = layout_of(proc)
        self.assertEqual((s["padded_experts"], s["padding_experts"], s["experts_per_rank"]), (258, 2, 86))
        self.assertEqual(s["expert_ranges"][-1], [172, 256])

    def test_missing_ep_size_is_rejected_before_anything_starts(self):
        proc = dry_run("--tp-size", "4", "--asap-num-replicas", "4", "--asap-replica-tp", "1")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--ep-size", proc.stderr)

    def test_missing_replica_shape_is_rejected(self):
        proc = dry_run("--tp-size", "2", "--ep-size", "2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("num_replicas", proc.stderr)

    def test_worker_lifecycle_ready_line_then_clean_exit_on_sigterm(self):
        proc = subprocess.Popen(
            ffn_cmd("--tp-size", "2", "--ep-size", "2", "--asap-num-replicas", "2", "--asap-replica-tp", "1"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=child_env())
        try:
            seen = []
            for line in proc.stdout:
                seen.append(line)
                if "ffn worker started" in line:
                    break
            else:
                self.fail("worker exited before becoming ready:\n" + "".join(seen)[-2000:])
            proc.send_signal(signal.SIGTERM)
            rest = proc.communicate(timeout=30)[0]
            self.assertEqual(proc.returncode, 0)
            self.assertIn("ffn worker stopped", rest)
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
