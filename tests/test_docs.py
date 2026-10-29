import os
import unittest

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


class TestDocsInSync(unittest.TestCase):
    """The README is the operator's reference: it must name every hook target and every flag."""

    @classmethod
    def setUpClass(cls):
        with open(README, encoding="utf-8") as f:
            cls.text = f.read()

    def test_every_hook_target_is_documented(self):
        from sgl_asap_plugin import hooks

        missing = [s.target for s in hooks.all_specs() if s.target not in self.text]
        self.assertEqual(missing, [])

    def test_every_flag_is_documented(self):
        import argparse

        from sgl_asap_plugin import config

        parser = argparse.ArgumentParser()
        config.add_cli_args(parser)
        flags = [o for a in parser._actions for o in a.option_strings if o.startswith("--asap-")]
        self.assertGreaterEqual(len(flags), 6)
        self.assertEqual([f for f in flags if f not in self.text], [])


if __name__ == "__main__":
    unittest.main()
