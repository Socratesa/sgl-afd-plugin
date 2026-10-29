import unittest

from test_transport import HAVE_GPUS, TransportCase, geometry, pick_devices

CASES = [
    # name, rows, valid rows, hash layer, skip shared expert
    ("plain", 7, 7, False, False),
    ("hash_layer", 5, 5, True, False),
    ("empty_batch", 0, 0, False, False),
    ("padded_rows", 10, 6, False, False),
    ("no_shared_expert", 4, 4, False, True),
]


@unittest.skipUnless(HAVE_GPUS, "needs >= 2 GPUs")
class TestAfdForward(TransportCase):
    def run_cases(self, n_f, fused):
        g = geometry(1, n_f)
        a_devs, f_devs = pick_devices(1, n_f)
        c = self.cluster(g, a_devs, f_devs)
        c.start_a(0, a_devs[0], "afd_forward", cases=CASES, fused=fused)
        result = c.result(0)
        return result, c.finish()

    def check(self, n_f, fused):
        result, f_stats = self.run_cases(n_f, fused)
        for name, *_ in CASES:
            with self.subTest(case=name):
                self.assertTrue(result[name]["shape_ok"], result[name])
                self.assertTrue(result[name]["equal"], result[name])
        self.assertEqual(result["empty_batch"]["messages"], 0)  # a rank without tokens sends nothing
        self.assertEqual([result[n]["messages"] for n in ("plain", "hash_layer", "padded_rows", "no_shared_expert")],
                         [1, 1, 1, 1])
        for stats, _, _ in f_stats.values():
            self.assertEqual(stats["jobs"], 4)  # one job per non-empty layer call, on every F rank

    def test_one_f_rank_with_scaling_fused_in_topk(self):
        self.check(1, True)

    def test_two_f_ranks_sum_their_partials(self):
        self.check(2, True)

    def test_unfused_scaling_is_applied_after_the_sum(self):
        self.check(2, False)


if __name__ == "__main__":
    unittest.main()
