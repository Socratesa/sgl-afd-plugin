import unittest

from _util import HAS_MODEL, MODEL_DIR, TINY_OVERRIDE, last_json, run_python

BASE_TARGETS = ["sglang.srt.server_args.ServerArgs.add_cli_args", "sglang.srt.server_args.ServerArgs.from_cli_args"]
ROLE_TARGETS = ["sglang.srt.arg_groups.validation_hook.check_server_args",
                "sglang.srt.managers.scheduler.Scheduler.__init__"]

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


class TestPreflight(unittest.TestCase):
    def test_every_spec_matches_the_installed_sglang(self):
        from sgl_asap_plugin import compat, hooks

        specs = hooks.all_specs()
        self.assertEqual(sorted(s.target for s in specs), sorted(BASE_TARGETS + ROLE_TARGETS))
        compat.preflight(specs)

    def test_preflight_names_every_broken_target(self):
        from sglang.srt.plugins.hook_registry import HookType

        from sgl_asap_plugin import compat, hooks
        from sgl_asap_plugin.errors import AsapInstallError

        def spec(target, params=()):
            return hooks.HookSpec(target, lambda *a: None, HookType.AFTER, params, hooks.ROLE)

        specs = [spec("no_such_module_xyz.f"), spec("json.no_such_attr"), spec("json.dumps", ("not_obj",)),
                 spec("json.loads", ("s",))]
        with self.assertRaises(AsapInstallError) as cm:
            compat.preflight(specs)
        msg = str(cm.exception)
        for token in ("no_such_module_xyz.f", "json.no_such_attr", "json.dumps: expected leading parameters"):
            self.assertIn(token, msg)
        self.assertNotIn("json.loads", msg)  # the valid spec is not reported


class TestActivationInFreshProcess(unittest.TestCase):
    def test_plain_process_gets_only_the_base_tier(self):
        r = run_python(PATCH_SNIPPET)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = last_json(r.stdout)
        self.assertEqual(out["patched"], sorted(BASE_TARGETS))
        self.assertTrue(out["flag"])

    def test_env_role_adds_the_role_tier(self):
        r = run_python(PATCH_SNIPPET, {"ASAP_ROLE": "attn"})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(last_json(r.stdout)["patched"], sorted(BASE_TARGETS + ROLE_TARGETS))

    def test_ffn_role_cannot_run_under_sglang_serve(self):
        r = run_python(PATCH_SNIPPET, {"ASAP_ROLE": "ffn", "ASAP_NUM_REPLICAS": "2", "ASAP_REPLICA_TP": "1"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ffn_serve", r.stderr)

    def test_bad_environment_fails_loudly(self):
        r = run_python(PATCH_SNIPPET, {"ASAP_ROLE": "attn", "ASAP_SLOTS": "abc"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ASAP_SLOTS", r.stderr)

    def test_target_drift_fails_loudly(self):
        code = """
import sgl_asap_plugin.hooks as h
orig = h.all_specs
def with_bogus(tier=None):
    specs = orig(tier)
    if tier == h.ROLE:
        specs = specs + [h.HookSpec("sglang.srt.managers.scheduler.Scheduler.no_such_method",
                                    lambda *a: None, h.HookType.AFTER, (), h.ROLE)]
    return specs
h.all_specs = with_bogus
from sgl_asap_plugin.activation import register_plugin
register_plugin()
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
        res[name] = "ok"
    except BaseException as e:
        res[name] = type(e).__name__ + ": " + " ".join(str(e).split())[:700]
    res[name + ":env_role"] = os.environ.get("ASAP_ROLE")
print(json.dumps(res))
"""

FLOW_CASES = [
    ("plain", ["--dp-size", "2", "--tp-size", "1"]),
    ("valid", ["--asap-role", "attn", "--dp-size", "2", "--tp-size", "1"]),
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

    def test_valid_attn_launch_activates_and_exports_the_role(self):
        self.assertEqual(self.res["valid"], "ok")
        self.assertEqual(self.res["valid:env_role"], "attn")

    def test_rejections_carry_an_actionable_message(self):
        expected = {
            "attn_dp": "attention data parallelism",
            "pp": "--pp-size",
            "ffn_under_serve": "ffn_serve",
            "bad_slots": "slots",
            "replica_mismatch": "--dp-size",
            "options_without_role": "--asap-role",
        }
        for case, token in expected.items():
            with self.subTest(case=case):
                self.assertIn("AsapConfigError", self.res[case])
                self.assertIn(token, self.res[case])


if __name__ == "__main__":
    unittest.main()
