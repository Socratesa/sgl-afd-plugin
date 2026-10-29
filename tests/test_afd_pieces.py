import json
import os
import tempfile
import unittest

from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.model import weights as W

NAMES = [
    "embed.weight", "head.weight", "norm.weight", "hc_head_base",
    "layers.0.attn.wq_a.weight", "layers.0.ffn.gate.weight", "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.shared_experts.w1.weight", "layers.0.ffn.shared_experts.w1.scale",
    "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
    "layers.0.ffn.experts.255.w2.weight",
    "layers.3.ffn.experts.64.w3.weight", "layers.3.ffn.experts.63.w3.weight",
    "layers.7.ffn.experts.64.w1.weight",
    "mtp.0.ffn.experts.0.w1.weight", "mtp.0.attn.wq_a.weight",
]


class TestSelection(unittest.TestCase):
    def test_attn_keeps_everything_but_routed_experts_and_mtp(self):
        kept = [n for n in NAMES if W.keep_for_attn(n)]
        self.assertEqual(kept, ["embed.weight", "head.weight", "norm.weight", "hc_head_base",
                                "layers.0.attn.wq_a.weight", "layers.0.ffn.gate.weight",
                                "layers.0.ffn.gate.tid2eid", "layers.0.ffn.shared_experts.w1.weight",
                                "layers.0.ffn.shared_experts.w1.scale"])

    def test_the_shared_expert_is_not_a_routed_expert(self):
        self.assertTrue(W.keep_for_attn("layers.9.ffn.shared_experts.w2.weight"))
        self.assertFalse(W.keep_for_attn("model.layers.9.ffn.experts.3.w2.scale"))

    def test_ffn_rank_reads_only_its_expert_range_and_layers(self):
        keep = W.expert_selector(layers=4, lo=64, hi=128)
        self.assertEqual([n for n in NAMES if keep(n)], ["layers.3.ffn.experts.64.w3.weight"])  # not 63, not layer 7

    def test_ranges_of_all_ranks_partition_the_experts_without_overlap(self):
        names = [f"layers.1.ffn.experts.{e}.w1.weight" for e in range(256)]
        picked = [{n for n in names if W.expert_selector(2, lo, lo + 64)(n)} for lo in range(0, 256, 64)]
        self.assertEqual(sum(len(p) for p in picked), 256)
        self.assertEqual(len(set().union(*picked)), 256)


class TestFilteredIterator(unittest.TestCase):
    def test_only_kept_tensors_are_read_and_names_come_back_sorted_per_shard(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            files = []
            for i in range(3):
                path = os.path.join(d, f"shard-{i}.safetensors")
                save_file({n: torch.full((2,), float(i)) for n in NAMES[i * 5:(i + 1) * 5]}, path)
                files.append(path)
            got = list(W.filtered_safetensors_iterator(files, W.keep_for_attn, max_workers=2))
            self.assertEqual({n for n, _ in got}, {n for n in NAMES[:15] if W.keep_for_attn(n)})
            per_shard = {}
            for n, t in got:
                self.assertEqual(float(t[0]), float(NAMES.index(n) // 5))  # the tensor came from its own shard
                per_shard.setdefault(float(t[0]), []).append(n)
            for names in per_shard.values():
                self.assertEqual(names, sorted(names))

    def test_select_files_lists_only_shards_with_wanted_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            wm = {"layers.0.attn.wq_a.weight": "a.safetensors", "layers.0.ffn.experts.1.w1.weight": "b.safetensors",
                  "layers.1.ffn.experts.1.w1.weight": "c.safetensors"}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": wm}, f)
            files, count = W.select_files(d, W.expert_selector(1, 0, 8))
            self.assertEqual(([os.path.basename(p) for p in files], count), (["b.safetensors"], 1))


class TestConfigAndClientPieces(unittest.TestCase):
    def test_ffn_ranks_round_trips_through_the_environment(self):
        c = AsapConfig(role="attn", ffn_ranks=4, num_replicas=2, replica_tp=1)
        env = {}
        c.export_env(env)
        self.assertEqual(env["ASAP_FFN_RANKS"], "4")
        self.assertEqual(AsapConfig.from_env(env).ffn_ranks, 4)

    def test_ffn_ranks_must_be_positive(self):
        with self.assertRaises(AsapConfigError):
            AsapConfig(role="attn", ffn_ranks=0)

    def test_f_rank_f_listens_on_the_base_port_plus_f(self):
        from sgl_asap_plugin.model.client import endpoints_of

        cfg = AsapConfig(role="attn", rdv_addr="10.1.2.3:29600", ffn_ranks=3, num_replicas=1, replica_tp=1)
        self.assertEqual(endpoints_of(cfg), [("10.1.2.3", 29600), ("10.1.2.3", 29601), ("10.1.2.3", 29602)])

    def test_model_mismatch_between_a_and_f_is_named(self):
        from sgl_asap_plugin.model.client import check_model

        desc = {"model": {"hidden": 4096, "topk": 6, "layers": [0, 1, 2, 3]}}
        check_model(desc, {"hidden": 4096, "topk": 6, "layers": [0, 1, 2, 3]})
        with self.assertRaises(AsapConfigError) as cm:
            check_model(desc, {"hidden": 4096, "topk": 8, "layers": list(range(43))})
        self.assertIn("topk", str(cm.exception))
        self.assertIn("MoE layers", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
