import unittest

try:
    import torch

    HAVE_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
except Exception:  # no torch / no driver: the GPU tests are skipped
    HAVE_GPUS = False

if HAVE_GPUS:
    from transport_harness import Cluster, gpu_ordinals

from sgl_asap_plugin.transport.arena import Geometry


def geometry(num_a, num_f, **kw):
    base = dict(num_a=num_a, group=1, slots=2, n_max=64, hidden=64, topk=6, num_f=num_f)
    base.update(kw)
    return Geometry(**base)


def pick_devices(n_a, n_f):
    gpus = gpu_ordinals()
    if len(gpus) < n_a + n_f:
        raise unittest.SkipTest(f"needs {n_a + n_f} GPUs, ASAP_TEST_GPUS has {len(gpus)}")
    return gpus[:n_a], gpus[n_a:n_a + n_f]


@unittest.skipUnless(HAVE_GPUS, "needs >= 2 GPUs")
class TransportCase(unittest.TestCase):
    """Starts a cluster per test and always tears it down."""

    def cluster(self, g, a_devs, f_devs, **kw):
        c = Cluster(g, f_devs, **kw)
        self.addCleanup(c.close)
        return c.start_f()

    def run_pingpong(self, g, n_a, n_f, scenario="pingpong", **kw):
        a_devs, f_devs = pick_devices(n_a, n_f)
        c = self.cluster(g, a_devs, f_devs, seq_base=kw.pop("seq_base", 0), f_mode=kw.pop("f_mode", "compute"),
                         f_spin_s=kw.pop("f_spin_s", None))
        for a in range(n_a):
            c.start_a(a, a_devs[a], scenario, **kw)
        results = [c.result(a) for a in range(n_a)]
        return c, results, c.finish()


class TestGeometry(unittest.TestCase):
    def test_sizes_at_the_design_point(self):
        g = geometry(4, 4, n_max=16384, hidden=4096)  # D*T=4, S=2, n_max=16K, E=4 (02b section 6)
        self.assertEqual(g.recv_bytes, 1_080_033_280)  # ~1.08GB per F rank
        self.assertEqual(g.result_bytes, 1_075_838_976)  # ~1.07GB per A rank, 2 MiB of it done flags

    def test_done_flags_never_share_a_page_with_payload(self):
        from sgl_asap_plugin.transport.arena import FLAG_STRIDE, FLAGS_PAGE

        g = geometry(2, 4, n_max=64, hidden=64)
        self.assertLessEqual(g.done_off(g.num_f - 1, g.slots - 1) + FLAG_STRIDE, g.flags_bytes)
        self.assertEqual(g.flags_bytes % FLAGS_PAGE, 0)
        self.assertEqual(g.seg_off(0, 0), g.flags_bytes)  # payload starts a whole page behind the first flag

    def test_invalid_geometry_names_the_field(self):
        from sgl_asap_plugin.errors import AsapConfigError

        with self.assertRaises(AsapConfigError) as cm:
            geometry(3, 1, group=2)
        self.assertIn("group", str(cm.exception))

    def test_doorbell_round_trip_at_the_field_limits(self):
        from sgl_asap_plugin.transport.protocol import MAX_EPOCH, MAX_LAYER, MAX_TOKENS, Doorbell

        for bell in (Doorbell(1, 1, 0, 1), Doorbell(2 ** 32 - 1, MAX_TOKENS, MAX_LAYER, MAX_EPOCH),
                     Doorbell(0xFFFFFFF0, 8192, 42, 7)):
            word = bell.pack()
            self.assertLess(word, 2 ** 64)
            self.assertEqual(Doorbell.unpack(word), bell)

    def test_doorbell_rejects_out_of_range_fields(self):
        from sgl_asap_plugin.transport.protocol import MAX_EPOCH, MAX_LAYER, MAX_TOKENS, Doorbell

        for bad in (Doorbell(1, 0, 0, 1), Doorbell(1, MAX_TOKENS + 1, 0, 1), Doorbell(1, 1, MAX_LAYER + 1, 1),
                    Doorbell(1, 1, 0, 0), Doorbell(1, 1, 0, MAX_EPOCH + 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bad.pack()

    def test_geometry_cannot_exceed_what_the_doorbell_can_announce(self):
        from sgl_asap_plugin.errors import AsapConfigError
        from sgl_asap_plugin.transport.protocol import MAX_TOKENS

        with self.assertRaises(AsapConfigError):
            geometry(1, 1, n_max=MAX_TOKENS + 1)


class TestPingPong(TransportCase):
    def test_one_a_one_f_1000_rounds(self):
        _, (r,), f = self.run_pingpong(geometry(1, 1), 1, 1, rounds=1000)
        self.assertEqual(r["bad"], 0)
        stats = f[0][0]
        self.assertEqual(stats["jobs"], 1000)
        self.assertEqual(stats.get("lost_messages", 0), 0)

    def test_three_f_ranks_partial_sums_add_up(self):
        _, (r,), f = self.run_pingpong(geometry(1, 3), 1, 3, rounds=200)
        self.assertEqual(r["bad"], 0)
        self.assertTrue(all(s[0]["jobs"] == 200 for s in f.values()))

    def test_sizes_from_8kb_to_64mb(self):
        g = geometry(1, 1, n_max=8192, hidden=4096)
        _, (r,), _ = self.run_pingpong(g, 1, 1, rounds=24, sizes=[1, 2, 16, 128, 1024, 8192])
        self.assertEqual(r["bad"], 0)

    def test_sequence_numbers_wrap_around_2_pow_32(self):
        _, (r,), f = self.run_pingpong(geometry(1, 1), 1, 1, rounds=100, seq_base=0xFFFFFFF0)
        self.assertEqual(r["bad"], 0)
        self.assertEqual(f[0][0].get("lost_messages", 0), 0)

    def test_stress_18k_rounds(self):
        _, (r,), f = self.run_pingpong(geometry(1, 1), 1, 1, rounds=18000, n=2)
        self.assertEqual(r["bad"], 0)
        self.assertEqual(f[0][0]["jobs"], 18000)

    def test_first_come_first_served_with_three_senders(self):
        _, results, f = self.run_pingpong(geometry(3, 1), 3, 1, rounds=300, think_us=300)
        self.assertEqual([r["bad"] for r in results], [0, 0, 0])
        stats, order, jobs = f[0]
        self.assertEqual(stats["jobs"], 900)
        self.assertEqual(sorted({g for g, _ in order}), [0, 1, 2])
        switches = sum(1 for p, q in zip(order, order[1:]) if p[0] != q[0])
        self.assertGreater(switches, 100)  # served in arrival order, not forced one sender at a time


class TestBackpressure(TransportCase):
    def run_hold(self, **channel_kwargs):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), a_devs, f_devs, hold_f=True)
        c.start_a(0, a_devs[0], "backpressure", hold_s=1.0, channel_kwargs=channel_kwargs)
        result = c.result(0)
        return result, c.finish()[0]

    def test_second_send_into_a_busy_slot_blocks_and_nothing_is_lost(self):
        result, (stats, _, jobs) = self.run_hold()
        self.assertEqual(result["send_states_while_f_held"], ["done", "pending"])
        self.assertEqual([j[1] for j in jobs], [1, 2])  # both messages delivered, in order
        self.assertEqual(stats.get("lost_messages", 0), 0)

    def test_control_without_backpressure_the_loss_is_detected(self):
        result, (stats, _, jobs) = self.run_hold(_backpressure=False)
        self.assertEqual(result["send_states_while_f_held"], ["done", "done"])  # overwrote the unread message
        self.assertGreaterEqual(stats["lost_messages"], 1)
        self.assertEqual([j[1] for j in jobs], [2])


class TestPerf(TransportCase):
    """Echo F (no compute): the numbers are the transport plus its Python, nothing else."""

    def rtt_n1(self, **kw):
        small = geometry(1, 1, hidden=4096, n_max=8192)
        # think_us decorrelates A's send times from the poller's wake-ups
        _, (r,), _ = self.run_pingpong(small, 1, 1, rounds=2000, n=1, verify=False, f_mode="echo", think_us=500, **kw)
        return r["lat_us"]

    def test_small_message_round_trip(self):
        spin, nap = self.rtt_n1(), self.rtt_n1(f_spin_s=0.0)
        print(f"\n[perf] RTT n=1 us, F spinning: {spin}; F napping on every idle poll: {nap}")
        self.assertLess(spin["p50"], 150)  # spike: ~77 us for the bare primitives; Python adds the rest
        self.assertGreater(nap["p50"], spin["p50"] + 40)  # why the poller spins: a nap lands on every layer

    def test_64mb_round_trip_bandwidth(self):
        big = geometry(1, 1, hidden=4096, n_max=8192)
        _, (r,), _ = self.run_pingpong(big, 1, 1, rounds=60, n=8192, verify=False, f_mode="echo")
        gbps = 2 * 8192 * 4096 * 2 / (r["lat_us"]["p50"] * 1e-6) / 1e9  # both directions carry 64MB
        print(f"\n[perf] 64MB each way: RTT us {r['lat_us']}, aggregate {gbps:.0f} GB/s")
        self.assertGreater(gbps, 150)  # sequential directions, so about the one-way rate minus latencies

    def test_64mb_one_way_and_full_duplex(self):
        big = geometry(1, 1, hidden=4096, n_max=8192)
        _, (one,), _ = self.run_pingpong(big, 1, 1, "oneway", rounds=40, f_mode="echo")
        _, (both,), _ = self.run_pingpong(big, 1, 1, "duplex", rounds=100, warmup=10, f_mode="echo")
        print(f"\n[perf] 64MB one way {one['gbps']:.0f} GB/s; send and reply overlapped: {both['gbps']:.0f} GB/s "
              f"({both['us_per_message']:.0f} us per message)")
        self.assertGreater(one["gbps"], 330)  # spike gate; a raw IPC cuMemcpyAsync does ~385
        self.assertGreater(both["gbps"], 650)  # spike gate; raw copies in both directions do ~770
        self.assertGreater(both["gbps"], one["gbps"] * 1.4)  # the two directions really run at the same time


if __name__ == "__main__":
    unittest.main()
