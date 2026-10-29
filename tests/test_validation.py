import types
import unittest

from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.validation import check_attn, check_ffn, check_for_role

ATTN = dict(attn_dp_size=1, enable_dp_attention=False, dp_size=2, tp_size=1, pp_size=1, nnodes=1, ep_size=1,
            moe_a2a_backend="none", enable_eplb=False, expert_distribution_recorder_mode=None,
            disaggregation_mode="null")
FFN = dict(tp_size=4, ep_size=4, dp_size=1, attn_dp_size=1, enable_dp_attention=False, pp_size=1, nnodes=1,
           moe_a2a_backend="none", enable_eplb=False, disaggregation_mode="null")


def view(base, **override):
    return types.SimpleNamespace(**{**base, **override})


class TestAttnRules(unittest.TestCase):
    asap = AsapConfig(role="attn")

    def test_valid_configurations_pass(self):
        for override in ({}, {"dp_size": 1}, {"dp_size": 8}, {"tp_size": 2}, {"disaggregation_mode": "prefill"},
                         {"moe_a2a_backend": None}):
            with self.subTest(override=override):
                check_attn(view(ATTN, **override), self.asap)

    def test_each_violation_is_rejected_with_a_pointer_to_the_flag(self):
        cases = [
            ({"attn_dp_size": 2}, "attention data parallelism"),
            ({"enable_dp_attention": True}, "attention data parallelism"),
            ({"nnodes": 2}, "--nnodes 1"),
            ({"pp_size": 2}, "--pp-size"),
            ({"moe_a2a_backend": "deepep"}, "--moe-a2a-backend"),
            ({"ep_size": 2}, "--ep-size"),
            ({"enable_eplb": True}, "--enable-eplb"),
            ({"expert_distribution_recorder_mode": "stat"}, "--expert-distribution-recorder-mode"),
            ({"disaggregation_mode": "decode"}, "prefill phase only"),
        ]
        for override, token in cases:
            with self.subTest(override=override), self.assertRaises(AsapConfigError) as cm:
                check_attn(view(ATTN, **override), self.asap)
            self.assertIn(token, str(cm.exception))

    def test_all_violations_are_reported_together(self):
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(view(ATTN, attn_dp_size=2, pp_size=2, nnodes=2), self.asap)
        for token in ("attention data parallelism", "--pp-size", "--nnodes"):
            self.assertIn(token, str(cm.exception))

    def test_declared_replica_shape_must_match_sglang_flags(self):
        check_attn(view(ATTN), AsapConfig(role="attn", num_replicas=2, replica_tp=1))
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(view(ATTN), AsapConfig(role="attn", num_replicas=3))
        self.assertIn("--dp-size", str(cm.exception))
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(view(ATTN), AsapConfig(role="attn", replica_tp=2))
        self.assertIn("--tp-size", str(cm.exception))

    def test_missing_sglang_field_is_reported_as_version_drift(self):
        broken = types.SimpleNamespace(**{k: v for k, v in ATTN.items() if k != "pp_size"})
        with self.assertRaises(AsapConfigError) as cm:
            check_attn(broken, self.asap)
        self.assertIn("pp_size", str(cm.exception))
        self.assertIn("version drift", str(cm.exception))


class TestFfnRules(unittest.TestCase):
    asap = AsapConfig(role="ffn", num_replicas=4, replica_tp=1)

    def test_valid(self):
        check_ffn(view(FFN), self.asap)
        check_ffn(view(FFN, tp_size=3, ep_size=3), self.asap)

    def test_violations(self):
        cases = [
            ({"ep_size": 1}, "--ep-size"),
            ({"dp_size": 2}, "no data-parallel dimension"),
            ({"enable_dp_attention": True}, "no data-parallel dimension"),
            ({"moe_a2a_backend": "deepep"}, "--moe-a2a-backend"),
            ({"nnodes": 2}, "--nnodes 1"),
            ({"disaggregation_mode": "prefill"}, "--disaggregation-mode"),
        ]
        for override, token in cases:
            with self.subTest(override=override), self.assertRaises(AsapConfigError) as cm:
                check_ffn(view(FFN, **override), self.asap)
            self.assertIn(token, str(cm.exception))


class TestDispatch(unittest.TestCase):
    def test_dispatch_by_role(self):
        check_for_role(view(ATTN), AsapConfig(role="attn"))
        check_for_role(view(FFN), AsapConfig(role="ffn", num_replicas=4, replica_tp=1))
        with self.assertRaises(AsapConfigError):
            check_for_role(view(ATTN, attn_dp_size=2), AsapConfig(role="attn"))


if __name__ == "__main__":
    unittest.main()
