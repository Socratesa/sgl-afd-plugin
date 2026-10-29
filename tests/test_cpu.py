"""CPU-only tests: configuration, deployment rules, transport layout, checkpoint selection, hook activation."""

import argparse
import os
import tempfile
import types
import unittest

from _util import HAS_MODEL, MODEL_DIR, TINY_OVERRIDE, last_json, run_python

from sgl_asap_plugin import config
from sgl_asap_plugin.config import AsapConfig, AsapConfigError, check_attn, check_ffn
from sgl_asap_plugin.transport.layout import (FLAG_STRIDE, FLAGS_PAGE, MAX_EPOCH, MAX_LAYER, MAX_TOKENS, Geometry,
                                              pack_doorbell, unpack_doorbell)

ATTN = dict(attn_dp_size=1, enable_dp_attention=False, dp_size=2, tp_size=1, pp_size=1, nnodes=1, ep_size=1,
            moe_a2a_backend="none", enable_eplb=False, expert_distribution_recorder_mode=None,
            disaggregation_mode="null", speculative_algorithm=None, chunked_prefill_size=8192, max_prefill_tokens=8192)
FFN = dict(tp_size=4, ep_size=4, dp_size=1, attn_dp_size=1, enable_dp_attention=False, pp_size=1, nnodes=1,
           moe_a2a_backend="none", enable_eplb=False, disaggregation_mode="null")


def view(base, **override):
    return types.SimpleNamespace(**{**base, **override})


def parse(argv):
    parser = argparse.ArgumentParser()
    config.add_cli_args(parser)
    return parser.parse_args(argv)


class TestConfig(unittest.TestCase):
    def test_defaults_and_afd_switch(self):
        c = AsapConfig(role="attn")
        self.assertEqual((c.rdv_addr, c.n_max, c.slots, c.afd), ("127.0.0.1:9600", 16384, 2, False))
        self.assertTrue(AsapConfig(role="attn", ffn_ranks=2).afd)
        self.assertFalse(AsapConfig(role="ffn", num_replicas=1, replica_tp=1, ffn_ranks=2).afd)

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig(role="x", rdv_addr="nohost", n_max=0, slots=99, ffn_ranks=0)
        for token in ("role", "rdv_addr", "n_max", "slots", "ffn_ranks"):
            self.assertIn(token, str(cm.exception))

    def test_bad_addresses(self):
        for addr in ("", "localhost", "h:0", "h:65536", "h:x", "h:1:2", ":80"):
            with self.subTest(addr=addr), self.assertRaises(AsapConfigError):
                AsapConfig(role="attn", rdv_addr=addr)

    def test_n_max_is_bounded_by_the_doorbell(self):
        AsapConfig(role="attn", n_max=MAX_TOKENS)
        with self.assertRaises(AsapConfigError):
            AsapConfig(role="attn", n_max=MAX_TOKENS + 1)

    def test_ffn_needs_the_replica_shape(self):
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig(role="ffn")
        self.assertIn("--asap-num-replicas", str(cm.exception))
        AsapConfig(role="ffn", num_replicas=4, replica_tp=1)

    def test_env_round_trip_replaces_stale_values(self):
        c = AsapConfig(role="ffn", rdv_addr="10.0.0.1:1234", n_max=4096, slots=3, num_replicas=4, replica_tp=2,
                       ffn_ranks=3)
        env = {"ASAP_SLOTS": "7", "OTHER": "1"}
        c.export_env(env)
        self.assertEqual(AsapConfig.from_env(env), c)
        AsapConfig(role="attn").export_env(env)
        self.assertNotIn("ASAP_NUM_REPLICAS", env)
        self.assertEqual(env["OTHER"], "1")
        self.assertIsNone(AsapConfig.from_env({}))
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig.from_env({"ASAP_ROLE": "attn", "ASAP_SLOTS": "abc"})
        self.assertIn("ASAP_SLOTS", str(cm.exception))

    def test_cli(self):
        c = AsapConfig.from_namespace(parse(["--asap-role", "ffn", "--asap-rdv-addr", "h:1", "--asap-num-replicas",
                                             "4", "--asap-replica-tp", "1", "--asap-ffn-ranks", "2"]))
        self.assertEqual((c.role, c.rdv_addr, c.num_replicas, c.replica_tp, c.ffn_ranks), ("ffn", "h:1", 4, 1, 2))
        self.assertIsNone(AsapConfig.from_namespace(parse([])))
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig.from_namespace(parse(["--asap-slots", "3"]))
        self.assertIn("--asap-role", str(cm.exception))

    def test_defaults_fill_only_unset_fields_and_shape_the_geometry(self):
        c = AsapConfig(role="attn", ffn_ranks=3, replica_tp=1).with_defaults(num_replicas=2, replica_tp=8)
        self.assertEqual((c.num_replicas, c.replica_tp), (2, 1))
        self.assertEqual(c.endpoints(), [("127.0.0.1", 9600), ("127.0.0.1", 9601), ("127.0.0.1", 9602)])
        g = c.geometry(hidden=4096, topk=6)
        self.assertEqual((g.num_a, g.group, g.num_f, g.slots), (2, 1, 3, 2))

    def test_experts_per_rank_pads_the_last_rank(self):
        self.assertEqual(config.experts_per_rank(256, 4), 64)
        self.assertEqual(config.experts_per_rank(256, 3), 86)  # ranks own [0,86) [86,172) [172,256) + 2 padding
        for bad in (0, 257):
            with self.subTest(bad=bad), self.assertRaises(AsapConfigError):
                config.experts_per_rank(256, bad)


class TestRules(unittest.TestCase):
    plain, afd = AsapConfig(role="attn"), AsapConfig(role="attn", ffn_ranks=4, n_max=8192)
    ffn = AsapConfig(role="ffn", num_replicas=4, replica_tp=1)

    def test_valid_attention_launches(self):
        for cfg, override in ((self.plain, {}), (self.plain, {"tp_size": 2}), (self.plain, {"dp_size": 8}),
                              (self.afd, {}), (self.afd, {"disaggregation_mode": "prefill"})):
            with self.subTest(afd=cfg.afd, override=override):
                check_attn(view(ATTN, **override), cfg)

    def test_attention_violations_point_at_the_flag(self):
        cases = [
            (self.plain, {"attn_dp_size": 2}, "attention data parallelism"),
            (self.plain, {"enable_dp_attention": True}, "attention data parallelism"),
            (self.plain, {"nnodes": 2}, "--nnodes 1"),
            (self.plain, {"pp_size": 2}, "--pp-size"),
            (self.plain, {"moe_a2a_backend": "deepep"}, "--moe-a2a-backend"),
            (self.plain, {"enable_eplb": True}, "--enable-eplb"),
            (self.plain, {"disaggregation_mode": "decode"}, "prefill phase only"),
            (AsapConfig(role="attn", num_replicas=3), {}, "--dp-size"),
            (AsapConfig(role="attn", replica_tp=2), {}, "--tp-size"),
            (self.afd, {"ep_size": 2}, "--ep-size"),
            (self.afd, {"tp_size": 2}, "--tp-size 1"),
            (self.afd, {"speculative_algorithm": "EAGLE"}, "speculative"),
            (self.afd, {"expert_distribution_recorder_mode": "stat"}, "--expert-distribution-recorder-mode"),
            (self.afd, {"chunked_prefill_size": 16384}, "--chunked-prefill-size"),
            (self.afd, {"chunked_prefill_size": -1}, "--chunked-prefill-size"),
            (self.afd, {"max_prefill_tokens": 16384}, "--max-prefill-tokens"),
        ]
        for cfg, override, token in cases:
            with self.subTest(override=override), self.assertRaises(AsapConfigError) as cm:
                check_attn(view(ATTN, **override), cfg)
            self.assertIn(token, str(cm.exception))

    def test_all_violations_together_and_version_drift(self):
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(view(ATTN, attn_dp_size=2, pp_size=2, nnodes=2), self.plain)
        for token in ("attention data parallelism", "--pp-size", "--nnodes"):
            self.assertIn(token, str(cm.exception))
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(types.SimpleNamespace(**{k: v for k, v in ATTN.items() if k != "pp_size"}), self.plain)
        self.assertIn("version drift", str(cm.exception))

    def test_expert_service_rules(self):
        check_ffn(view(FFN), self.ffn)
        check_ffn(view(FFN, tp_size=3, ep_size=3), self.ffn)
        cases = [({"ep_size": 1}, "--ep-size"), ({"dp_size": 2}, "no data-parallel dimension"),
                 ({"moe_a2a_backend": "deepep"}, "--moe-a2a-backend"), ({"nnodes": 2}, "--nnodes 1"),
                 ({"disaggregation_mode": "prefill"}, "--disaggregation-mode")]
        for override, token in cases:
            with self.subTest(override=override), self.assertRaises(AsapConfigError) as cm:
                check_ffn(view(FFN, **override), self.ffn)
            self.assertIn(token, str(cm.exception))
        with self.assertRaises(AsapConfigError) as cm:
            check_ffn(view(FFN), AsapConfig(role="ffn", num_replicas=4, replica_tp=1, ffn_ranks=2))
        self.assertIn("--asap-ffn-ranks", str(cm.exception))


class TestLayout(unittest.TestCase):
    def g(self, **kw):
        return Geometry(**{**dict(num_a=1, group=1, slots=2, n_max=64, hidden=64, topk=6, num_f=1), **kw})

    def test_sizes_at_the_design_point(self):
        g = self.g(num_a=4, num_f=4, n_max=16384, hidden=4096)  # D*T=4, S=2, n_max=16K, E=4 (02b section 6)
        self.assertEqual(g.recv_bytes, 1_080_033_280)
        self.assertEqual(g.result_bytes, 1_075_838_976)

    def test_regions_are_aligned_and_flags_have_a_page_of_their_own(self):
        g = self.g(num_a=3, num_f=4, n_max=100, hidden=96)
        for off in (g.ids_off, g.w_off, g.region_bytes, g.region_off(2, 1), g.seg_off(3, 1), g.seg_bytes):
            self.assertEqual(off % 256, 0)
        self.assertLessEqual(g.done_off(g.num_f - 1, g.slots - 1) + FLAG_STRIDE, g.flags_bytes)
        self.assertEqual((g.flags_bytes % FLAGS_PAGE, g.seg_off(0, 0)), (0, g.flags_bytes))
        self.assertEqual(g.bell_off(2, 1) + 64, g.num_a * g.slots * 64)

    def test_invalid_geometry_names_the_field(self):
        for kw, token in (({"num_a": 3, "group": 2}, "group"), ({"n_max": MAX_TOKENS + 1}, "n_max"),
                          ({"slots": 0}, "slots")):
            with self.subTest(kw=kw), self.assertRaises(ValueError) as cm:
                self.g(**kw)
            self.assertIn(token, str(cm.exception))
        self.assertEqual(Geometry.from_dict(self.g(num_f=3).as_dict()), self.g(num_f=3))

    def test_doorbell_round_trip_at_the_field_limits(self):
        for fields in ((1, 1, 0, 1), (2 ** 32 - 1, MAX_TOKENS, MAX_LAYER, MAX_EPOCH), (0xFFFFFFF0, 8192, 42, 7)):
            word = pack_doorbell(*fields)
            self.assertLess(word, 2 ** 64)
            self.assertEqual(unpack_doorbell(word), fields)
        for bad in ((1, 0, 0, 1), (1, MAX_TOKENS + 1, 0, 1), (1, 1, MAX_LAYER + 1, 1), (1, 1, 0, 0),
                    (1, 1, 0, MAX_EPOCH + 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pack_doorbell(*bad)


class TestCheckpoint(unittest.TestCase):
    def test_name_selection(self):
        from sgl_asap_plugin.checkpoint import expert_selector, keep_for_attn

        for name, attn in (("layers.3.ffn.experts.17.w1.weight", False), ("layers.3.ffn.experts.17.w2.scale", False),
                           ("layers.3.ffn.shared_experts.w1.weight", True), ("layers.3.ffn.gate.weight", True),
                           ("layers.3.attn.wq_a.weight", True), ("mtp.0.ffn.experts.1.w1.weight", False),
                           ("mtp.0.attn.wq_a.weight", False), ("embed.weight", True)):
            self.assertEqual(keep_for_attn(name), attn, name)
        keep = expert_selector(4, 86, 172)
        self.assertEqual([keep(f"layers.{l}.ffn.experts.{e}.w3.weight") for l, e in
                          ((0, 86), (3, 171), (0, 85), (0, 172), (4, 100))], [True, True, False, False, False])
        self.assertFalse(keep("layers.0.ffn.shared_experts.w1.weight"))

    def test_filtered_iterator_reads_only_kept_tensors(self):
        import torch
        from safetensors.torch import save_file

        from sgl_asap_plugin.checkpoint import filtered_safetensors_iterator, keep_for_attn

        with tempfile.TemporaryDirectory() as d:
            files = []
            for i in range(3):
                path = os.path.join(d, f"s{i}.safetensors")
                save_file({f"layers.{i}.ffn.experts.0.w1.weight": torch.full((2,), float(i)),
                           f"layers.{i}.attn.wq_a.weight": torch.full((2,), float(i))}, path)
                files.append(path)
            for workers in (1, 2):
                got = dict(filtered_safetensors_iterator(files, keep_for_attn, max_workers=workers))
                self.assertEqual(sorted(got), [f"layers.{i}.attn.wq_a.weight" for i in range(3)])
                self.assertEqual(float(got["layers.2.attn.wq_a.weight"][0]), 2.0)


def _targets(tier):
    from sgl_asap_plugin import hooks

    return [s.target for s in hooks.SPECS if s.tier == tier]


PATCH_SNIPPET = """
import json, argparse
import sgl_asap_plugin
sgl_asap_plugin.register()
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.apply_hooks()
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p)
print(json.dumps({"patched": sorted(HookRegistry._patched), "flag": "--asap-role" in p._option_string_actions}))
"""


class TestHookActivation(unittest.TestCase):
    """Each case runs in a fresh interpreter: hooks patch sglang globally."""

    def test_every_spec_matches_the_installed_sglang(self):
        from sgl_asap_plugin import hooks

        hooks.preflight(hooks.SPECS)
        self.assertEqual(len({s.target for s in hooks.SPECS}), len(hooks.SPECS))

    def test_preflight_names_every_broken_target(self):
        from sglang.srt.plugins.hook_registry import HookType

        from sgl_asap_plugin import hooks
        from sgl_asap_plugin.config import AsapInstallError

        def spec(target, params=()):
            return hooks.HookSpec(target, HookType.AFTER, params, hooks.ROLE, lambda *a: None, "test")

        with self.assertRaises(AsapInstallError) as cm:
            hooks.preflight([spec("no_such_module_xyz.f"), spec("json.no_such_attr"), spec("json.dumps", ("x",)),
                             spec("json.loads", ("s",))])
        for token in ("no_such_module_xyz.f", "json.no_such_attr", "json.dumps: expected leading parameters"):
            self.assertIn(token, str(cm.exception))
        self.assertNotIn("json.loads", str(cm.exception))

    def check_tiers(self, env, tiers):
        r = run_python(PATCH_SNIPPET, env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = last_json(r.stdout)
        self.assertEqual(out["patched"], sorted(t for tier in tiers for t in _targets(tier)))
        self.assertTrue(out["flag"])

    def test_tiers_follow_the_environment(self):
        from sgl_asap_plugin.hooks import AFD, BASE, ROLE

        self.check_tiers({}, [BASE])
        self.check_tiers({"ASAP_ROLE": "attn"}, [BASE, ROLE])
        self.check_tiers({"ASAP_ROLE": "attn", "ASAP_FFN_RANKS": "2"}, [BASE, ROLE, AFD])

    def test_bad_environments_fail_loudly(self):
        for env, token in (({"ASAP_ROLE": "ffn", "ASAP_NUM_REPLICAS": "2", "ASAP_REPLICA_TP": "1"}, "sgl_asap_plugin.ffn"),
                           ({"ASAP_ROLE": "attn", "ASAP_SLOTS": "abc"}, "ASAP_SLOTS")):
            with self.subTest(env=env):
                r = run_python(PATCH_SNIPPET, env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(token, r.stderr)

    def test_target_drift_fails_loudly(self):
        code = """
import sgl_asap_plugin.hooks as h
h.SPECS.append(h.HookSpec("sglang.srt.managers.scheduler.Scheduler.no_such_method", h.HookType.AFTER, (), h.ROLE,
                          lambda *a: None, "bogus"))
h.register_plugin()
print("registered")
"""
        r = run_python(code, {"ASAP_ROLE": "attn"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no_such_method", r.stderr)
        self.assertNotIn("registered", r.stdout)


FLOW_SNIPPET = """
import json, os
from sglang.srt.plugins import load_plugins
load_plugins()
from sglang.srt.server_args import prepare_server_args
BASE = ["--model-path", os.environ["T_MODEL"], "--load-format", "dummy",
        "--json-model-override-args", os.environ["T_OVERRIDE"]]
res = {}
for name, extra in json.loads(os.environ["T_CASES"]):
    try:
        sa = prepare_server_args(BASE + extra)
        sa.resolve_once()  # Engine._launch_subprocesses resolves before it checks
        sa.check_server_args()
        from sglang.srt.arg_groups.overrides import resolving_view
        v = resolving_view(sa)
        res[name] = "ok"
        res[name + ":cuda_graph_off"] = bool(v.disable_cuda_graph)
    except BaseException as e:
        res[name] = type(e).__name__ + ": " + " ".join(str(e).split())[:700]
    res[name + ":env_role"] = os.environ.get("ASAP_ROLE")
print(json.dumps(res))
"""

FLOW_CASES = [
    ("plain", ["--dp-size", "2", "--tp-size", "1"]),
    ("valid", ["--asap-role", "attn", "--dp-size", "2", "--tp-size", "1"]),
    ("afd", ["--asap-role", "attn", "--dp-size", "2", "--asap-ffn-ranks", "2", "--chunked-prefill-size", "8192",
             "--max-prefill-tokens", "8192"]),
    ("afd_big_chunk", ["--asap-role", "attn", "--asap-ffn-ranks", "2", "--chunked-prefill-size", "32768"]),
    ("attn_dp", ["--asap-role", "attn", "--dp-size", "2", "--tp-size", "2", "--enable-dp-attention"]),
    ("pp", ["--asap-role", "attn", "--pp-size", "2"]),
    ("ffn_under_serve", ["--asap-role", "ffn", "--asap-num-replicas", "2", "--asap-replica-tp", "1"]),
    ("bad_slots", ["--asap-role", "attn", "--asap-slots", "99"]),
    ("replica_mismatch", ["--asap-role", "attn", "--dp-size", "2", "--asap-num-replicas", "3"]),
    ("options_without_role", ["--asap-slots", "3"]),
]


@unittest.skipUnless(HAS_MODEL, "needs the DeepSeek-V4-Flash config on this machine")
class TestServerArgsFlow(unittest.TestCase):
    """The real launcher path: plugin loaded by entry point, flags parsed by sglang, args resolved and checked."""

    @classmethod
    def setUpClass(cls):
        import json

        env = {"T_MODEL": MODEL_DIR, "T_OVERRIDE": TINY_OVERRIDE, "T_CASES": json.dumps(FLOW_CASES)}
        r = run_python(FLOW_SNIPPET, env)
        assert r.returncode == 0, r.stderr[-3000:]
        cls.res = last_json(r.stdout)

    def test_plain_launch_is_untouched(self):
        self.assertEqual(self.res["plain"], "ok")
        self.assertIsNone(self.res["plain:env_role"])

    def test_valid_launches_activate_and_afd_turns_cuda_graphs_off(self):
        self.assertEqual((self.res["valid"], self.res["valid:env_role"]), ("ok", "attn"))
        self.assertEqual(self.res["afd"], "ok")
        self.assertTrue(self.res["afd:cuda_graph_off"])

    def test_rejections_carry_an_actionable_message(self):
        expected = {"afd_big_chunk": "--chunked-prefill-size", "attn_dp": "attention data parallelism",
                    "pp": "--pp-size", "ffn_under_serve": "sgl_asap_plugin.ffn", "bad_slots": "slots",
                    "replica_mismatch": "--dp-size", "options_without_role": "--asap-role"}
        for case, token in expected.items():
            with self.subTest(case=case):
                self.assertIn("AsapConfigError", self.res[case])
                self.assertIn(token, self.res[case])


class TestDocsInSync(unittest.TestCase):
    """The README is the operator's reference: it must name every hook target and every flag."""

    def test_readme_names_every_hook_and_flag(self):
        from sgl_asap_plugin import hooks

        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")) as f:
            text = f.read()
        self.assertEqual([s.target for s in hooks.SPECS if s.target not in text], [])
        parser = argparse.ArgumentParser()
        config.add_cli_args(parser)
        flags = [o for a in parser._actions for o in a.option_strings if o.startswith("--asap-")]
        self.assertGreaterEqual(len(flags), 7)
        self.assertEqual([f for f in flags if f not in text], [])


if __name__ == "__main__":
    unittest.main()
