import argparse
import unittest

from sgl_asap_plugin import config
from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.errors import AsapConfigError


class TestAsapConfig(unittest.TestCase):
    def test_defaults(self):
        c = AsapConfig(role="attn")
        self.assertEqual((c.rdv_addr, c.n_max, c.slots), ("127.0.0.1:29600", 16384, 2))

    def test_every_problem_is_reported(self):
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig(role="x", rdv_addr="nohost", n_max=0, slots=99)
        for token in ("role", "rdv_addr", "n_max", "slots"):
            self.assertIn(token, str(cm.exception))

    def test_bad_addresses(self):
        for addr in ("", "localhost", "h:0", "h:65536", "h:x", "h:1:2", ":80"):
            with self.subTest(addr=addr), self.assertRaises(AsapConfigError):
                AsapConfig(role="attn", rdv_addr=addr)

    def test_ffn_needs_replica_shape(self):
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig(role="ffn")
        self.assertIn("num_replicas", str(cm.exception))
        self.assertIn("replica_tp", str(cm.exception))
        AsapConfig(role="ffn", num_replicas=4, replica_tp=1)

    def test_env_round_trip(self):
        c = AsapConfig(role="ffn", rdv_addr="10.0.0.1:1234", n_max=4096, slots=3, num_replicas=4, replica_tp=2)
        env = c.to_env()
        self.assertEqual(env["ASAP_ROLE"], "ffn")
        self.assertEqual(AsapConfig.from_env(env), c)

    def test_from_env_unset_and_bad(self):
        self.assertIsNone(AsapConfig.from_env({}))
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig.from_env({"ASAP_ROLE": "attn", "ASAP_SLOTS": "abc"})
        self.assertIn("ASAP_SLOTS", str(cm.exception))

    def test_export_env_drops_stale_values(self):
        env = {"ASAP_NUM_REPLICAS": "9", "ASAP_N_MAX": "7", "OTHER": "1"}
        AsapConfig(role="attn", n_max=100).export_env(env)
        self.assertEqual(env["ASAP_N_MAX"], "100")
        self.assertNotIn("ASAP_NUM_REPLICAS", env)
        self.assertEqual(env["OTHER"], "1")

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        config.add_cli_args(parser)
        return parser.parse_args(argv)

    def test_cli(self):
        ns = self._parse(["--asap-role", "ffn", "--asap-rdv-addr", "h:1", "--asap-num-replicas", "4",
                          "--asap-replica-tp", "1"])
        c = AsapConfig.from_namespace(ns)
        self.assertEqual((c.role, c.rdv_addr, c.num_replicas, c.replica_tp), ("ffn", "h:1", 4, 1))
        self.assertIsNone(AsapConfig.from_namespace(self._parse([])))

    def test_cli_options_without_role(self):
        with self.assertRaises(AsapConfigError) as cm:
            AsapConfig.from_namespace(self._parse(["--asap-slots", "3"]))
        self.assertIn("--asap-role", str(cm.exception))

    def test_cli_rejects_unknown_role(self):
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse(["--asap-role", "decode"])


if __name__ == "__main__":
    unittest.main()
