"""CPU startup contracts using the real upstream registry and plugin loader.

The target functions are synthetic: this suite proves activation, dispatch, and
failure boundaries, not model execution or GPU compatibility. Set
ASAP_TEST_SGLANG_SOURCE to a supported SGLang checkout to select its implementation.
"""

import argparse
from contextlib import ExitStack
import importlib
import os
import sys
import unittest
from unittest import mock

from _hook_support import registry_sandbox


class TestHookActivationContract(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.runtime = self.stack.enter_context(registry_sandbox())
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(mock.patch.object(sys, "argv", ["sglang", "serve"]))
        self.plugin = importlib.import_module("sgl_asap_plugin")
        self.hooks = importlib.import_module("sgl_asap_plugin.hooks")
        self.registry = self.runtime.registry.HookRegistry
        self.loader = self.runtime.loader
        self.events = []
        self.module = self.runtime.module("asap_hook_test.activation")
        events = self.events

        class ServerArgs:
            @staticmethod
            def add_cli_args(parser):
                parser.add_argument("--native-value", default="native")

            @classmethod
            def from_cli_args(cls, args):
                events.append(("native_cli", os.environ.get("ASAP_ROLE")))
                return args

        def role_target(value):
            events.append(("native_role", value))
            return value

        def afd_target(value):
            events.append(("native_afd", value))
            return value

        def role_callback(value):
            events.append(("role_hook", value))

        def afd_callback(value):
            events.append(("afd_hook", value))
            return "remote", value

        self.module.ServerArgs = ServerArgs
        self.module.role_target = role_target
        self.module.afd_target = afd_target
        h = self.hooks
        ht = self.runtime.registry.HookType
        prefix = self.module.__name__ + "."
        self.role_target = prefix + "role_target"
        self.afd_target = prefix + "afd_target"
        self.specs = [
            h.HookSpec(prefix + "ServerArgs.add_cli_args", ht.AFTER, ("parser",), h.BASE,
                       h._add_cli_args, "CPU CLI fixture"),
            h.HookSpec(prefix + "ServerArgs.from_cli_args", ht.BEFORE, ("cls", "args"), h.BASE,
                       h._activate_from_cli, "CPU CLI activation fixture"),
            h.HookSpec(self.role_target, ht.BEFORE, ("value",), h.ROLE,
                       role_callback, "CPU role fixture"),
            h.HookSpec(self.afd_target, ht.REPLACE, ("value",), h.AFD,
                       afd_callback, "CPU remote-expert fixture"),
        ]
        self.stack.enter_context(mock.patch.object(h, "SPECS", self.specs))
        self.stack.enter_context(mock.patch.object(h, "_warn_if_untested"))
        self.plugins = {"sgl_asap_plugin": (self.plugin.register, None)}
        self.stack.enter_context(mock.patch.object(
            self.loader, "load_plugins_by_group", side_effect=lambda *a, **k: self.plugins,
        ))
        self.stack.enter_context(mock.patch.object(self.loader, "_get_excluded_dists", return_value=set()))

    def invalidate(self, index, *, wrong_signature=False):
        from dataclasses import replace

        original = self.specs[index]
        if wrong_signature:
            self.specs[index] = replace(original, params=("incompatible_parameter",))
        else:
            self.specs[index] = replace(original, target=original.target + "_missing")

    def enable(self, *, afd=False):
        os.environ["ASAP_ROLE"] = "attn"
        if afd:
            os.environ["ASAP_FFN_RANKS"] = "2"

    def parse(self, *arguments):
        parser = argparse.ArgumentParser()
        self.module.ServerArgs.add_cli_args(parser)
        return parser.parse_args(arguments)

    def assert_loader_stops(self, message=None):
        with self.assertRaises(SystemExit) as raised:
            self.loader.load_plugins()
        if message:
            self.assertIn(message, str(raised.exception))
        self.assertEqual(self.events, [])

    def test_env_enabled_base_missing_stops_through_loader(self):
        self.enable()
        self.invalidate(0)
        self.assert_loader_stops("add_cli_args_missing")

    def test_env_enabled_base_signature_drift_stops_through_loader(self):
        self.enable()
        self.invalidate(0, wrong_signature=True)
        self.assert_loader_stops("incompatible_parameter")

    def test_disabled_base_drift_keeps_native_server_args(self):
        self.invalidate(0)
        with self.assertLogs("sgl_asap_plugin", level="WARNING"):
            self.loader.load_plugins()
        args = self.parse("--native-value", "unchanged")
        self.assertEqual(self.module.ServerArgs.from_cli_args(args).native_value, "unchanged")
        self.assertEqual(self.events, [("native_cli", None)])
        self.assertNotIn("ASAP_ROLE", os.environ)

    def test_disabled_healthy_hooks_preserve_native_behavior(self):
        self.loader.load_plugins()
        args = self.parse()
        self.assertIs(self.module.ServerArgs.from_cli_args(args), args)
        self.assertEqual(self.module.role_target("value"), "value")
        self.assertEqual(self.module.afd_target("value"), "value")
        self.assertEqual(self.events, [("native_cli", None), ("native_role", "value"),
                                       ("native_afd", "value")])
        self.assertNotIn("ASAP_ROLE", os.environ)

    def test_partial_base_application_never_exposes_inactive_asap_flags(self):
        original_apply = self.registry._apply_target
        activation_target = self.specs[1].target

        def fail_activation(target, hooks):
            if target == activation_target:
                raise RuntimeError("injected from_cli_args application failure")
            return original_apply(target, hooks)

        with mock.patch.object(self.registry, "_apply_target", side_effect=fail_activation):
            self.loader.load_plugins()
        parser = argparse.ArgumentParser()
        self.module.ServerArgs.add_cli_args(parser)
        self.assertEqual(parser.parse_args([]).native_value, "native")
        self.assertEqual(parser.parse_args(["--native-value", "retained"]).native_value, "retained")
        self.assertNotIn("--asap-role", parser._option_string_actions)
        # Configuration files become parser arguments upstream. Accepting the
        # role here without its activation hook would silently drop the opt-in.
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--asap-role", "attn"])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.events, [])

    def test_env_enabled_role_drift_stops(self):
        self.enable()
        self.invalidate(2)
        self.assert_loader_stops("role_target_missing")

    def test_env_enabled_afd_drift_stops(self):
        self.enable(afd=True)
        self.invalidate(3)
        self.assert_loader_stops("afd_target_missing")

    def test_inactive_afd_target_drift_does_not_break_attention_role(self):
        self.enable()
        self.invalidate(3)
        self.loader.load_plugins()
        self.assertEqual(self.module.role_target("value"), "value")
        self.assertEqual(self.events, [("role_hook", "value"), ("native_role", "value")])

    def test_invalid_environment_stops_through_loader(self):
        os.environ["ASAP_ROLE"] = "invalid-role"
        self.assert_loader_stops("role")

    def import_failure(self):
        original_import = __import__

        def broken_import(name, *args, **kwargs):
            if name == "sgl_asap_plugin.hooks":
                raise ImportError("injected hook dependency unavailable")
            return original_import(name, *args, **kwargs)

        return mock.patch("builtins.__import__", side_effect=broken_import)

    def test_entrypoint_import_failure_is_fatal_when_env_enabled(self):
        self.enable()
        with self.import_failure():
            self.assert_loader_stops("injected hook dependency unavailable")

    def test_entrypoint_import_failure_is_fatal_for_explicit_cli_option(self):
        sys.argv.extend(["--asap-role=attn"])
        with self.import_failure():
            self.assert_loader_stops("injected hook dependency unavailable")

    def test_entrypoint_import_failure_is_fatal_for_empty_explicit_role(self):
        sys.argv.extend(["--asap-role="])
        with self.import_failure():
            self.assert_loader_stops("injected hook dependency unavailable")

    def test_entrypoint_import_failure_keeps_disabled_launch_native(self):
        with self.import_failure(), self.assertLogs("sgl_asap_plugin", level="WARNING"):
            self.loader.load_plugins()
        args = self.parse()
        self.module.ServerArgs.from_cli_args(args)
        self.assertEqual(self.events, [("native_cli", None)])

    def test_cli_activation_failure_prevents_native_call_and_env_export(self):
        from sgl_asap_plugin.config import AsapInstallError

        self.invalidate(2)
        self.loader.load_plugins()
        args = self.parse("--asap-role", "attn")
        with self.assertRaises(AsapInstallError):
            self.module.ServerArgs.from_cli_args(args)
        self.assertEqual(self.events, [])
        self.assertNotIn("ASAP_ROLE", os.environ)

    def test_cli_afd_failure_prevents_native_call_and_env_export(self):
        from sgl_asap_plugin.config import AsapInstallError

        self.invalidate(3)
        self.loader.load_plugins()
        args = self.parse("--asap-role", "attn", "--asap-ffn-ranks", "2")
        with self.assertRaises(AsapInstallError):
            self.module.ServerArgs.from_cli_args(args)
        self.assertEqual(self.events, [])
        self.assertNotIn("ASAP_ROLE", os.environ)
        self.assertNotIn("ASAP_FFN_RANKS", os.environ)

    def test_cli_activation_exports_before_native_resolution(self):
        self.loader.load_plugins()
        args = self.parse("--asap-role", "attn", "--asap-ffn-ranks", "2")
        self.assertIs(self.module.ServerArgs.from_cli_args(args), args)
        self.assertEqual(self.events, [("native_cli", "attn")])
        self.assertEqual(os.environ["ASAP_FFN_RANKS"], "2")
        self.assertEqual(self.module.afd_target("value"), ("remote", "value"))
        self.assertEqual(self.events[-1], ("afd_hook", "value"))

    def test_cli_cannot_downgrade_an_installed_afd_role(self):
        from sgl_asap_plugin.config import AsapConfigError

        self.enable(afd=True)
        self.loader.load_plugins()
        args = self.parse("--asap-role", "attn")
        with self.assertRaises(AsapConfigError):
            self.module.ServerArgs.from_cli_args(args)
        self.assertEqual(self.events, [])
        self.assertEqual(os.environ["ASAP_ROLE"], "attn")
        self.assertEqual(os.environ["ASAP_FFN_RANKS"], "2")
        self.assertEqual(self.module.afd_target("value"), ("remote", "value"))
        self.assertEqual(self.events, [("afd_hook", "value")])

    def test_repeated_registration_and_application_do_not_stack_callbacks(self):
        self.enable(afd=True)
        self.loader.load_plugins()
        self.plugin.register()
        self.registry.apply_hooks()
        self.plugin.register()
        self.registry.apply_hooks()
        self.assertEqual(self.module.role_target("value"), "value")
        self.assertEqual(self.module.afd_target("value"), ("remote", "value"))
        self.assertEqual(self.events, [("role_hook", "value"), ("native_role", "value"),
                                       ("afd_hook", "value")])

    def test_prior_foreign_replacement_stops_enabled_activation(self):
        self.enable(afd=True)
        self.registry.register(self.afd_target, lambda value: "foreign", self.runtime.registry.HookType.REPLACE)
        self.registry.apply_hooks()
        self.assert_loader_stops("afd_target")

    def test_later_plugin_queued_conflict_stops_at_final_loader_apply(self):
        self.enable(afd=True)

        def foreign_plugin():
            self.registry.register(self.afd_target, lambda value: "foreign", self.runtime.registry.HookType.REPLACE)

        self.plugins["later_plugin"] = foreign_plugin, None
        self.assert_loader_stops("afd_target")

    def test_later_plugin_direct_replacement_stops_at_final_loader_apply(self):
        self.enable(afd=True)

        def foreign_plugin():
            self.module.afd_target = lambda value: "foreign"

        self.plugins["later_plugin"] = foreign_plugin, None
        self.assert_loader_stops("afd_target")

    def test_registry_apply_failure_cannot_be_hidden_by_loader(self):
        self.enable(afd=True)
        original_apply = self.registry._apply_target

        def fail_afd(target, hooks):
            if target == self.afd_target:
                raise RuntimeError("injected apply failure")
            return original_apply(target, hooks)

        with mock.patch.object(self.registry, "_apply_target", side_effect=fail_afd):
            self.assert_loader_stops("afd_target")


if __name__ == "__main__":
    unittest.main()
