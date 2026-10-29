import json
import os
import tempfile
import unittest

from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.layout import arena_plan, partition_experts, read_moe_shape
from sgl_asap_plugin.transport.arena import FLAGS_PAGE


class TestExpertPartition(unittest.TestCase):
    def test_ranges_cover_every_expert_exactly_once(self):
        for ep in (1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 256):
            with self.subTest(ep=ep):
                part = partition_experts(256, ep)
                covered = [e for lo, hi in part.ranges() for e in range(lo, hi)]
                self.assertEqual(covered, list(range(256)))
                self.assertEqual(part.padded, part.per_rank * ep)
                self.assertEqual(part.padding, part.padded - 256)
                self.assertLess(part.padding, ep)

    def test_padding_matches_the_design_doc(self):
        # 02a section 4.1: E=3/5/6 need dummy experts to reach 258/260/258.
        for ep, padded, per_rank in ((2, 256, 128), (3, 258, 86), (4, 256, 64), (5, 260, 52), (6, 258, 43),
                                     (8, 256, 32)):
            with self.subTest(ep=ep):
                part = partition_experts(256, ep)
                self.assertEqual((part.padded, part.per_rank), (padded, per_rank))

    def test_dummy_experts_sit_in_the_last_rank(self):
        part = partition_experts(256, 3)
        self.assertEqual(part.ranges(), [(0, 86), (86, 172), (172, 256)])
        self.assertEqual(part.padding, 2)

    def test_invalid(self):
        for n, ep in ((0, 1), (256, 0), (4, 5)):
            with self.subTest(n=n, ep=ep), self.assertRaises(AsapConfigError):
                partition_experts(n, ep)


class TestArenaPlan(unittest.TestCase):
    def test_matches_the_design_doc_numbers(self):
        # 02b section 6: D*T=4, S=2, n_max=16K, E=4 -> F recv ~1.08GB, A result ~1.07GB.
        plan = arena_plan(num_replicas=4, replica_tp=1, ep_size=4, slots=2, n_max=16384, hidden=4096, topk=6)
        self.assertEqual(plan.bytes_per_token, 8240)
        self.assertEqual(plan.region_bytes, 16384 * 8240)
        self.assertEqual(plan.f_recv_arena_bytes, 1_080_033_280)  # 8 regions
        self.assertEqual(plan.a_result_arena_bytes, FLAGS_PAGE + 1_073_741_824)  # flags page + 8 segments

    def test_scales_with_replicas_and_tp(self):
        one = arena_plan(1, 1, 4, 2, 1024, 4096, 6)
        self.assertEqual(arena_plan(2, 2, 4, 2, 1024, 4096, 6).f_recv_arena_bytes, 4 * one.f_recv_arena_bytes)
        twice = arena_plan(1, 1, 8, 2, 1024, 4096, 6)
        self.assertEqual(twice.a_result_arena_bytes - FLAGS_PAGE, 2 * (one.a_result_arena_bytes - FLAGS_PAGE))


class TestMoeShape(unittest.TestCase):
    def _model(self, cfg):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        return d

    def test_read_with_override(self):
        d = self._model({"n_routed_experts": 256, "hidden_size": 4096, "num_experts_per_tok": 6,
                         "num_hidden_layers": 43})
        shape = read_moe_shape(d)
        self.assertEqual((shape.n_experts, shape.hidden, shape.topk, shape.layers), (256, 4096, 6, 43))
        shape = read_moe_shape(d, '{"num_hidden_layers": 4, "n_routed_experts": 128}')
        self.assertEqual((shape.n_experts, shape.layers), (128, 4))

    def test_alternative_keys_and_text_config(self):
        d = self._model({"text_config": {"num_experts": 64, "hidden_size": 2048, "num_experts_per_tok": 8,
                                         "num_hidden_layers": 48}})
        self.assertEqual(read_moe_shape(d).n_experts, 64)

    def test_missing_or_incomplete_config(self):
        with self.assertRaises(AsapConfigError):
            read_moe_shape(tempfile.mkdtemp())
        with self.assertRaises(AsapConfigError):
            read_moe_shape(self._model({"hidden_size": 8}))


if __name__ == "__main__":
    unittest.main()
