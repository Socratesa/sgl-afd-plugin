"""GPU tests. A and F ranks run as separate processes on separate GPUs (ASAP_TEST_GPUS, default 0,1,2,3).

TestPingPong / TestBackpressure / TestFaults / TestPerf: the transport (PR-03 acceptance C*, B*, F*, P*).
TestAttnForward: attn.moe_forward against synthetic F ranks (PR-04).
TestFfnService: the real expert service against a torch reference (PR-05), needs the checkpoint.
"""

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

try:
    import torch

    HAVE_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
except Exception:  # no torch / no driver: GPU tests are skipped
    HAVE_GPUS = False

if HAVE_GPUS:
    from transport_harness import Cluster, gpu_ordinals

from sgl_asap_plugin.transport.layout import Geometry


def geometry(num_a, num_f, **kw):
    return Geometry(**{**dict(num_a=num_a, group=1, slots=2, n_max=64, hidden=64, topk=6, num_f=num_f), **kw})


def pick_devices(n_a, n_f):
    gpus = gpu_ordinals()
    if len(gpus) < n_a + n_f:
        raise unittest.SkipTest(f"needs {n_a + n_f} GPUs, ASAP_TEST_GPUS has {len(gpus)}")
    return gpus[:n_a], gpus[n_a:n_a + n_f]


@unittest.skipUnless(HAVE_GPUS, "needs >= 2 GPUs")
class ClusterCase(unittest.TestCase):
    """Starts a cluster per test and always tears it down."""

    def cluster(self, g, f_devs, **kw):
        c = Cluster(g, f_devs, **kw)
        self.addCleanup(c.close)
        return c.start_f()

    def run_scenario(self, g, n_a, n_f, scenario="pingpong", **kw):
        a_devs, f_devs = pick_devices(n_a, n_f)
        c = self.cluster(g, f_devs, seq_base=kw.pop("seq_base", 0), f_mode=kw.pop("f_mode", "compute"),
                         f_spin_s=kw.pop("f_spin_s", None))
        for a in range(n_a):
            c.start_a(a, a_devs[a], scenario, **kw)
        results = [c.result(a) for a in range(n_a)]
        return c, results, c.finish()


class TestPingPong(ClusterCase):
    def test_one_a_one_f_1000_rounds(self):
        _, (r,), f = self.run_scenario(geometry(1, 1), 1, 1, rounds=1000)
        self.assertEqual(r["bad"], 0)
        self.assertEqual((f[0][0]["jobs"], f[0][0].get("lost_messages", 0)), (1000, 0))

    def test_three_f_ranks_partial_sums_add_up(self):
        _, (r,), f = self.run_scenario(geometry(1, 3), 1, 3, rounds=200)
        self.assertEqual(r["bad"], 0)
        self.assertTrue(all(s[0]["jobs"] == 200 for s in f.values()))

    def test_sizes_from_8kb_to_64mb(self):
        _, (r,), _ = self.run_scenario(geometry(1, 1, n_max=8192, hidden=4096), 1, 1, rounds=24,
                                       sizes=[1, 2, 16, 128, 1024, 8192])
        self.assertEqual(r["bad"], 0)

    def test_sequence_numbers_wrap_around_2_pow_32(self):
        _, (r,), f = self.run_scenario(geometry(1, 1), 1, 1, rounds=100, seq_base=0xFFFFFFF0)
        self.assertEqual((r["bad"], f[0][0].get("lost_messages", 0)), (0, 0))

    def test_stress_18k_rounds(self):
        _, (r,), f = self.run_scenario(geometry(1, 1), 1, 1, rounds=18000, n=2)
        self.assertEqual((r["bad"], f[0][0]["jobs"]), (0, 18000))

    def test_first_come_first_served_with_three_senders(self):
        _, results, f = self.run_scenario(geometry(3, 1), 3, 1, rounds=300, think_us=300)
        self.assertEqual([r["bad"] for r in results], [0, 0, 0])
        stats, trace, _ = f[0]
        self.assertEqual(stats["jobs"], 900)
        self.assertEqual(sorted({replica for replica, _, _ in trace}), [0, 1, 2])
        self.assertGreater(stats["replica_switches"], 100)  # arrival order, not one sender at a time


class TestBackpressure(ClusterCase):
    def run_hold(self, **channel_kwargs):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), f_devs, hold_f=True)
        c.start_a(0, a_devs[0], "backpressure", hold_s=1.0, channel_kwargs=channel_kwargs)
        return c.result(0), c.finish()[0]

    def test_second_send_into_a_busy_slot_blocks_and_nothing_is_lost(self):
        result, (stats, _, rows) = self.run_hold()
        self.assertEqual(result["send_states_while_f_held"], ["done", "pending"])
        self.assertEqual([r[1] for r in rows], [1, 2])  # both messages delivered, in order
        self.assertEqual(stats.get("lost_messages", 0), 0)

    def test_control_without_backpressure_the_loss_is_detected(self):
        result, (stats, _, rows) = self.run_hold(_backpressure=False)
        self.assertEqual(result["send_states_while_f_held"], ["done", "done"])  # overwrote the unread message
        self.assertGreaterEqual(stats["lost_messages"], 1)
        self.assertEqual([r[1] for r in rows], [2])


class TestFaults(ClusterCase):
    def test_dead_f_times_out_instead_of_hanging(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), f_devs, hold_f=True)  # F never answers
        c.start_a(0, a_devs[0], "expect_timeout", timeout_s=2.0)
        c.wait("a_waiting")
        c.f_procs[0].kill()  # F dies while A's stream is parked in the wait
        r = c.result(0, timeout=60)
        self.assertEqual((r["check"], r["send_after"]), ("ChannelPoisoned", "ChannelPoisoned"))
        self.assertTrue(1.5 <= r["waited_s"] <= 10, r)

    def test_stale_epoch_message_is_dropped_and_reconnect_recovers(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), f_devs)
        c.start_a(0, a_devs[0], "expect_timeout", timeout_s=2.0, recover=True, channel_kwargs={"_epoch_override": 31})
        r = c.result(0, timeout=90)
        self.assertEqual((r["check"], r["after_reconnect"]["bad"]), ("ChannelPoisoned", 0))
        stats = c.finish()[0][0]
        self.assertGreaterEqual(stats["stale_epoch"], 1)
        self.assertEqual(stats["reconnects"], 1)

    def test_killed_a_does_not_disturb_f_or_the_other_a(self):
        a_devs, f_devs = pick_devices(2, 1)
        c = self.cluster(geometry(2, 1), f_devs)
        c.start_a(0, a_devs[0], "pingpong", rounds=8000, think_us=300)
        victim = c.start_a(1, a_devs[1], "pingpong", rounds=10 ** 7, n=8)
        time.sleep(4)
        victim.kill()  # F keeps replying into the dead A's IPC memory for a moment
        self.assertEqual(c.result(0)["bad"], 0)
        c.start_a(1, a_devs[1], "pingpong", rounds=100)  # a new incarnation of rank 1 reconnects
        self.assertEqual(c.result(1)["bad"], 0)
        stats = c.finish()[0][0]
        self.assertEqual((stats["reconnects"], stats.get("lost_messages", 0)), (1, 0))

    def test_handshake_rejects_a_geometry_mismatch(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), f_devs)
        c.start_a(0, a_devs[0], "pingpong", rounds=1, geometry_override={"n_max": 128})
        (msg,) = c.wait("a_error", 1, 60, errors_ok=True)
        self.assertIn("geometry mismatch", msg[2])
        self.assertIn("n_max", msg[2])

    def test_parked_wait_delays_cold_work_but_does_not_deadlock(self):
        a_devs, _ = pick_devices(2, 0)
        c = self.cluster(geometry(1, 1), [])
        c.start_a(0, a_devs[0], "parked_wait_delays_cold_work_only", release_device=a_devs[1])
        r = c.result(0, 120)
        print(f"\n[pitfall] cold work behind a parked wait: {r['cold_s']}s (released after {r['delay_s']}s); "
              f"warm work: {r['warm_s']}s")
        self.assertLess(r["cold_s"], r["delay_s"] + 15)  # delayed until the release, never deadlocked
        self.assertLess(r["warm_s"], 0.5)


class TestPerf(ClusterCase):
    """Echo F (no compute): the numbers are the transport plus its Python, nothing else."""

    def rtt_n1(self, **kw):
        # think_us decorrelates A's send times from the poller's wake-ups
        _, (r,), _ = self.run_scenario(geometry(1, 1, hidden=4096, n_max=8192), 1, 1, rounds=2000, n=1,
                                       verify=False, f_mode="echo", think_us=500, **kw)
        return r["lat_us"]

    def test_small_message_round_trip(self):
        spin, nap = self.rtt_n1(), self.rtt_n1(f_spin_s=0.0)
        print(f"\n[perf] RTT n=1 us, F spinning: {spin}; F napping on every idle poll: {nap}")
        self.assertLess(spin["p50"], 150)  # spike: ~77 us for the bare primitives; Python adds the rest
        self.assertGreater(nap["p50"], spin["p50"] + 40)  # why the poller spins: a nap lands on every layer

    def test_64mb_round_trip_bandwidth(self):
        _, (r,), _ = self.run_scenario(geometry(1, 1, hidden=4096, n_max=8192), 1, 1, rounds=60, n=8192,
                                       verify=False, f_mode="echo")
        gbps = 2 * 8192 * 4096 * 2 / (r["lat_us"]["p50"] * 1e-6) / 1e9  # both directions carry 64MB
        print(f"\n[perf] 64MB each way: RTT us {r['lat_us']}, aggregate {gbps:.0f} GB/s")
        self.assertGreater(gbps, 150)

    def test_64mb_one_way_and_full_duplex(self):
        big = geometry(1, 1, hidden=4096, n_max=8192)
        _, (one,), _ = self.run_scenario(big, 1, 1, "oneway", rounds=40, f_mode="echo")
        _, (both,), _ = self.run_scenario(big, 1, 1, "duplex", rounds=100, warmup=10, f_mode="echo")
        print(f"\n[perf] 64MB one way {one['gbps']:.0f} GB/s; send and reply overlapped: {both['gbps']:.0f} GB/s "
              f"({both['us_per_message']:.0f} us per message)")
        self.assertGreater(one["gbps"], 330)  # spike gate; a raw IPC cuMemcpyAsync does ~385
        self.assertGreater(both["gbps"], 650)  # spike gate; raw copies in both directions do ~770
        self.assertGreater(both["gbps"], one["gbps"] * 1.4)


ATTN_CASES = [  # (name, n_tokens, is_hash, skip_shared, return_moe_output)
    ("plain", 40, False, False, False),
    ("hash_layer", 33, True, False, False),
    ("no_shared_expert", 17, False, True, False),
    ("moe_output", 24, False, False, True),
    ("empty_batch", 0, False, False, False),
    ("larger_than_one_message", 150, False, False, False),  # n_max = 64: three messages
]


class TestAttnForward(ClusterCase):
    def check(self, num_f):
        _, (r,), _ = self.run_scenario(geometry(1, num_f), 1, num_f, "attn_forward", cases=ATTN_CASES)
        for name, n, *_ in ATTN_CASES:
            with self.subTest(case=name, num_f=num_f):
                self.assertTrue(r[name]["shape_ok"] and r[name]["equal"], r[name])
                self.assertEqual(r[name]["messages"], -(-n // 64))  # 0 for an empty batch
        self.assertTrue(r["hash_layer"]["hash_ids_passed"])
        self.assertTrue(r["local_layer_untouched"])

    def test_one_f_rank(self):
        self.check(1)

    def test_two_f_ranks_partial_sums(self):
        self.check(2)


class FfnService:
    """The real expert service as a subprocess; its log goes to /tmp (the shared volume can be full)."""

    READY = re.compile(r"\[ASAP\] ffn ready: ranks=(\d+)")
    RANK = re.compile(r"\[ASAP\] ffn rank (\d+) (ready|stats): (\{.*\})")

    def __init__(self, ranks, layers, a_ranks, base_gpu, port, n_max=256):
        self.argv = [sys.executable, "-m", "sgl_asap_plugin.ffn", "--model-path", MODEL_DIR, "--tp-size", str(ranks),
                     "--ep-size", str(ranks), "--asap-num-replicas", str(a_ranks), "--asap-replica-tp", "1",
                     "--asap-rdv-addr", f"127.0.0.1:{port}", "--asap-n-max", str(n_max), "--base-gpu-id",
                     str(base_gpu), "--json-model-override-args", json.dumps({"num_hidden_layers": layers})]
        self.log_path = tempfile.mktemp(prefix="asap_ffn_test_", suffix=".log", dir="/tmp")
        self.proc = None

    def start(self, timeout=1200):
        # the venv's ninja first, as after `source activate` (see scripts/servers.py: server_env)
        env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        for k in ("ASAP_ROLE", "SGLANG_PLUGINS", "CUDA_VISIBLE_DEVICES"):
            env.pop(k, None)
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.argv, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            if self.READY.search(self.text()):
                return self
            time.sleep(1)
        raise RuntimeError(f"F service did not become ready (exit={self.proc.poll()}):\n{self.text()[-3000:]}")

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def reports(self, kind):
        return {int(m.group(1)): json.loads(m.group(3)) for m in self.RANK.finditer(self.text()) if m.group(2) == kind}

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
class TestFfnService(ClusterCase):
    def test_lifecycle_one_rank_one_layer(self):
        (base,), _ = pick_devices(1, 0)
        svc = FfnService(1, 1, 1, base, 9710)
        self.addCleanup(svc.stop)
        svc.start()
        info = svc.reports("ready")[0]
        self.assertEqual((info["experts"], info["quant_method"]), ([0, 256], "Mxfp4FlashinferCutlassMoEMethod"))
        self.assertAlmostEqual(info["weights_gb"], 3.42, delta=0.2)  # one layer of 256 experts, nothing else
        self.assertEqual(svc.stop(), 0)
        self.assertIn("[ASAP] ffn stopped", svc.text())

    def test_three_ranks_match_the_reference_with_out_of_order_arrivals(self):
        ranks, a_ranks, layers, rounds = 3, 3, 4, 20  # 3 ranks: 258 slots, 2 padding experts at the end
        gpus = pick_devices(ranks + 1, 0)[0]
        svc = FfnService(ranks, layers, a_ranks, gpus[0], 9720)
        self.addCleanup(svc.stop)
        svc.start()
        c = Cluster(geometry(a_ranks, ranks, n_max=256, hidden=4096, topk=6), [])
        self.addCleanup(c.close)
        c.endpoints = [("127.0.0.1", 9720 + f) for f in range(ranks)]
        fingerprint = {"hidden": 4096, "topk": 6, "n_experts": 256, "moe_layers": list(range(layers))}
        pool = [3, 40, 85, 86, 100, 150, 171, 172, 200, 230, 254, 255, 17, 120, 190, 64]  # edges of all 3 ranges
        for a in range(a_ranks):
            c.start_a(a, gpus[ranks], "ffn_numerics", model_dir=MODEL_DIR, pool=pool, layers=layers, rounds=rounds,
                      sizes=[1, 5, 64], seed=a, think_us=400, channel_kwargs={"hello": {"model": fingerprint}})
        results = [c.result(a, timeout=900) for a in range(a_ranks)]
        print(f"\n[ffn] worst relative error vs the torch reference per A client: "
              f"{[round(r['worst_rel_err'], 4) for r in results]}; layers in arrival order {results[0]['layer_order_head']}")
        for r in results:
            self.assertLess(r["worst_rel_err"], 2e-2, r)
            self.assertGreaterEqual(len(r["layers_visited"]), 3, r)  # layers arrive in no particular order
        self.assertEqual(svc.stop(), 0)
        stats = svc.reports("stats")
        self.assertEqual(sorted(stats), [0, 1, 2])
        for f, st in stats.items():
            self.assertEqual((st["jobs"], st.get("lost_messages", 0)), (a_ranks * rounds, 0), f)
        weights = [info["weights_gb"] for info in svc.reports("ready").values()]
        self.assertTrue(all(abs(w - layers * 86 * 0.01337) < 0.6 for w in weights), weights)  # experts only

    def test_model_mismatch_is_rejected_in_the_handshake(self):
        (base, a_dev), _ = pick_devices(2, 0)
        svc = FfnService(1, 2, 1, base, 9730)
        self.addCleanup(svc.stop)
        svc.start()
        c = Cluster(geometry(1, 1, n_max=256, hidden=4096, topk=6), [])
        self.addCleanup(c.close)
        c.endpoints = [("127.0.0.1", 9730)]
        wrong = {"hidden": 4096, "topk": 6, "n_experts": 256, "moe_layers": list(range(43))}
        c.start_a(0, a_dev, "pingpong", rounds=1, channel_kwargs={"hello": {"model": wrong}})
        (msg,) = c.wait("a_error", 1, 120, errors_ok=True)
        self.assertIn("moe_layers", msg[2])


if __name__ == "__main__":
    unittest.main()
