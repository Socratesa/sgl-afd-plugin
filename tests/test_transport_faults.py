import time
import unittest

from test_transport import HAVE_GPUS, TransportCase, geometry, pick_devices


@unittest.skipUnless(HAVE_GPUS, "needs >= 2 GPUs")
class TestFaults(TransportCase):
    def test_dead_f_times_out_instead_of_hanging(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), a_devs, f_devs, hold_f=True)  # F never answers
        c.start_a(0, a_devs[0], "expect_timeout", timeout_s=2.0)
        c.wait("a_waiting")
        c.f_procs[0].kill()  # F dies while A's stream is parked in the wait
        r = c.result(0, timeout=60)
        self.assertEqual(r["check"], "ChannelPoisoned")
        self.assertEqual(r["send_after"], "ChannelPoisoned")  # a poisoned channel refuses new messages
        self.assertTrue(1.5 <= r["waited_s"] <= 10, r)

    def test_stale_epoch_message_is_dropped_and_reconnect_recovers(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), a_devs, f_devs)
        c.start_a(0, a_devs[0], "expect_timeout", timeout_s=2.0, recover=True,
                  channel_kwargs={"_epoch_override": 31})
        r = c.result(0, timeout=90)
        self.assertEqual(r["check"], "ChannelPoisoned")
        self.assertEqual(r["after_reconnect"]["bad"], 0)
        stats = c.finish()[0][0]
        self.assertGreaterEqual(stats["stale_epoch"], 1)
        self.assertEqual(stats["reconnects"], 1)

    def test_killed_a_does_not_disturb_f_or_the_other_a(self):
        a_devs, f_devs = pick_devices(2, 1)
        c = self.cluster(geometry(2, 1), a_devs, f_devs)
        c.start_a(0, a_devs[0], "pingpong", rounds=8000, think_us=300)
        victim = c.start_a(1, a_devs[1], "pingpong", rounds=10 ** 7, n=8)
        time.sleep(4)
        victim.kill()  # F keeps replying into the dead A's IPC memory for a moment
        self.assertEqual(c.result(0)["bad"], 0)
        c.start_a(1, a_devs[1], "pingpong", rounds=100)  # a new incarnation of rank 1 reconnects
        self.assertEqual(c.result(1)["bad"], 0)
        stats = c.finish()[0][0]
        self.assertEqual(stats["reconnects"], 1)
        self.assertEqual(stats.get("lost_messages", 0), 0)

    def test_handshake_rejects_a_geometry_mismatch(self):
        a_devs, f_devs = pick_devices(1, 1)
        c = self.cluster(geometry(1, 1), a_devs, f_devs)
        c.start_a(0, a_devs[0], "pingpong", rounds=1, geometry_override={"n_max": 128})
        (msg,) = c.wait("a_error", 1, 60, errors_ok=True)
        self.assertIn("geometry mismatch", msg[2])
        self.assertIn("n_max", msg[2])

    def test_parked_wait_delays_cold_work_but_does_not_deadlock(self):
        a_devs, _ = pick_devices(2, 0)
        c = self.cluster(geometry(1, 1), a_devs, [])
        c.start_a(0, a_devs[0], "parked_wait_delays_cold_work_only", release_device=a_devs[1])
        r = c.result(0, 120)
        print(f"\n[pitfall] cold work behind a parked wait: {r['cold_s']}s (released after {r['delay_s']}s); "
              f"warm work: {r['warm_s']}s")
        self.assertLess(r["cold_s"], r["delay_s"] + 15)  # delayed until the release, never deadlocked
        self.assertLess(r["warm_s"], 0.5)  # warm work is not affected


if __name__ == "__main__":
    unittest.main()
