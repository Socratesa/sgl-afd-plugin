import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from _util import HAS_MODEL, MODEL_DIR
from test_transport import HAVE_GPUS, geometry, pick_devices

if HAVE_GPUS:
    from transport_harness import Cluster

READY = re.compile(r"\[ASAP\] ffn ready: ranks=(\d+)")
RANK_READY = re.compile(r"\[ASAP\] ffn rank (\d+) ready: (\{.*\})")
RANK_STATS = re.compile(r"\[ASAP\] ffn rank (\d+) stats (\{.*?\}) layer_calls=(\d+)")


class FfnService:
    """The real F service as a subprocess; its output goes to a log file outside the shared volume."""

    def __init__(self, ranks, layers, a_ranks, base_gpu, port, n_max=256):
        self.ranks, self.layers, self.a_ranks, self.base_gpu, self.port, self.n_max = (
            ranks, layers, a_ranks, base_gpu, port, n_max)
        self.log_path = tempfile.mktemp(prefix="asap_ffn_test_", suffix=".log", dir="/tmp")
        self.proc = None

    def start(self, timeout=1200):
        env = dict(os.environ, SGLANG_JIT_DEEPGEMM_PRECOMPILE="0")
        for k in ("ASAP_ROLE", "SGLANG_PLUGINS", "CUDA_VISIBLE_DEVICES"):
            env.pop(k, None)
        argv = [sys.executable, "-m", "sgl_asap_plugin.ffn_serve", "--model-path", MODEL_DIR, "--tp-size",
                str(self.ranks), "--ep-size", str(self.ranks), "--asap-role", "ffn", "--asap-num-replicas",
                str(self.a_ranks), "--asap-replica-tp", "1", "--asap-ffn-ranks", str(self.ranks),
                "--asap-rdv-addr", f"127.0.0.1:{self.port}", "--asap-n-max", str(self.n_max), "--base-gpu-id",
                str(self.base_gpu), "--json-model-override-args", json.dumps({"num_hidden_layers": self.layers})]
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            if READY.search(self.text()):
                return self
            time.sleep(1)
        raise RuntimeError(f"F service did not become ready (exit={self.proc.poll()}):\n{self.text()[-3000:]}")

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()
        return self.proc.returncode


@unittest.skipUnless(HAVE_GPUS and HAS_MODEL, "needs GPUs and the DeepSeek-V4-Flash checkpoint")
class TestFfnService(unittest.TestCase):
    def test_lifecycle_one_rank_one_layer(self):
        (base,), _ = pick_devices(1, 0)
        svc = FfnService(1, 1, 1, base, 9710)
        self.addCleanup(svc.stop)
        svc.start()
        (rank, info), = [(int(m.group(1)), json.loads(m.group(2))) for m in RANK_READY.finditer(svc.text())]
        self.assertEqual((rank, info["experts"], info["quant_method"]), (0, [0, 256], "Mxfp4FlashinferCutlassMoEMethod"))
        self.assertAlmostEqual(info["weights_gb"], 3.42, delta=0.2)  # one layer of 256 experts, nothing else
        self.assertEqual(svc.stop(), 0)
        self.assertIn("[ASAP] ffn stopped", svc.text())

    def test_three_ranks_match_the_reference_with_out_of_order_arrivals(self):
        ranks, a_ranks, layers, rounds = 3, 3, 4, 20  # 3 ranks: 258 experts with 2 dummy ones at the end
        gpus = pick_devices(ranks + 1, 0)[0]
        svc = FfnService(ranks, layers, a_ranks, gpus[0], 9720)
        self.addCleanup(svc.stop)
        svc.start()
        g = geometry(a_ranks, ranks, n_max=256, hidden=4096, topk=6)
        c = Cluster(g, [])
        self.addCleanup(c.close)
        c.endpoints = [("127.0.0.1", 9720 + f) for f in range(ranks)]
        pool = [3, 40, 85, 86, 100, 150, 171, 172, 200, 230, 254, 255, 17, 120, 190, 64]  # edges of all 3 ranges
        for a in range(a_ranks):
            c.start_a(a, gpus[ranks], "ffn_numerics", model_dir=MODEL_DIR, pool=pool, layers=layers, rounds=rounds,
                      sizes=[1, 5, 64], seed=a, think_us=400,
                      ranges=[[0, 86], [86, 172], [172, 256]])
        results = [c.result(a, timeout=900) for a in range(a_ranks)]
        print(f"\n[ffn] worst relative error vs the torch reference, per A client: "
              f"{[round(r['worst_rel_err'], 4) for r in results]}; layers in arrival order {results[0]['layer_order_head']}")
        for r in results:
            self.assertLess(r["worst_rel_err"], 2e-2, r)
            self.assertGreaterEqual(len(r["layers_visited"]), 3, r)  # layers arrive in no particular order
        self.assertEqual(svc.stop(), 0)
        stats = {int(m.group(1)): (json.loads(m.group(2).replace("'", '"')), int(m.group(3)))
                 for m in RANK_STATS.finditer(svc.text())}
        self.assertEqual(sorted(stats), [0, 1, 2])
        for f, (st, calls) in stats.items():
            self.assertEqual((st["jobs"], calls), (a_ranks * rounds, a_ranks * rounds), f)  # every message, every rank
            self.assertEqual(st.get("lost_messages", 0), 0)
        weights = [json.loads(m.group(2))["weights_gb"] for m in RANK_READY.finditer(svc.text())]
        self.assertTrue(all(abs(w - layers * 86 * 0.01337) < 0.6 for w in weights), weights)  # experts only


if __name__ == "__main__":
    unittest.main()
