"""Exercise the compatibility adapter against real SGLang registry code."""

from __future__ import annotations

import functools
import importlib
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from _hook_support import registry_sandbox

from sgl_asap_plugin.config import AsapInstallError


class TestHookRegistry(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.runtime = self.stack.enter_context(registry_sandbox())
        self.compat = importlib.import_module("sgl_asap_plugin.compat")
        self.registry = self.runtime.registry.HookRegistry
        self.kind = self.compat.HookType
        self.target = self.runtime.module("asap_hook_test.targets")
        self.events = []

        def original(value):
            self.events.append("original")
            return value + 1

        self.target.original = original
        self.adapter = self.compat.HookAdapter(self.fail_install)

    @staticmethod
    def fail_install(error):
        raise error

    def spec(self, callback, kind=None, attr="original", params=("value",)):
        return types.SimpleNamespace(target=f"asap_hook_test.targets.{attr}", fn=callback,
                                     type=kind or self.kind.AFTER, params=params)

    def test_all_callback_types_execute(self):
        for kind in self.kind:
            with self.subTest(kind=kind):
                attr = f"target_{kind.value}"
                setattr(self.target, attr, self.target.original)

                if kind == self.kind.BEFORE:
                    def callback(value):
                        self.events.append("asap")
                elif kind == self.kind.AFTER:
                    def callback(result, value):
                        self.events.append("asap")
                elif kind == self.kind.AROUND:
                    def callback(original, value):
                        self.events.append("asap")
                        return original(value)
                else:
                    def callback(value):
                        self.events.append("asap")
                        return value + 1
                spec = self.spec(callback, kind, attr)
                self.adapter.register([spec])
                self.adapter.apply()
                self.events.clear()
                self.assertEqual(getattr(self.target, attr)(3), 4)
                self.assertEqual(self.events.count("asap"), 1)
                self.assertEqual(self.events.count("original"), int(kind != self.kind.REPLACE))
                self.adapter.ensure([spec])

    def test_before_and_after_other_plugins_work_in_both_orders(self):
        for first in (True, False):
            attr = f"order_{first}"
            setattr(self.target, attr, self.target.original)

            def ours(result, value):
                self.events.append("asap")

            def before(value):
                self.events.append("before")

            def after(result, value):
                self.events.append("after")

            spec = self.spec(ours, attr=attr)
            if not first:
                self.adapter.register([spec])
            self.registry.register(spec.target, before, self.kind.BEFORE)
            self.registry.register(spec.target, after, self.kind.AFTER)
            if first:
                self.adapter.register([spec])
            self.registry.apply_hooks()
            self.events.clear()
            self.assertEqual(getattr(self.target, attr)(3), 4)
            self.assertCountEqual(self.events, ["before", "after", "asap", "original"])

    def test_rejects_conflicting_callbacks_registered_first(self):
        for kind in (self.kind.REPLACE, self.kind.AROUND):
            attr = f"conflict_{kind.value}"
            setattr(self.target, attr, self.target.original)
            spec = self.spec(lambda *args: None, attr=attr)
            self.registry.register(spec.target, lambda *args: None, kind)
            with self.assertRaisesRegex(AsapInstallError, "conflicts"):
                self.adapter.register([spec])

    def test_rejects_conflicting_callbacks_registered_later(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        self.registry.register(spec.target, lambda value: value, self.kind.REPLACE)
        with self.assertRaisesRegex(AsapInstallError, "conflicts"):
            self.registry.apply_hooks()
        self.assertEqual(self.target.original(1), 2)

    def test_optional_failure_removes_pending_own_hooks(self):
        warnings = []
        adapter = self.compat.HookAdapter(warnings.append)
        spec = self.spec(lambda *args: self.events.append("asap"))
        adapter.register([spec])
        self.registry.register(spec.target, lambda value: 42, self.kind.REPLACE)
        self.registry.apply_hooks()
        self.assertEqual(self.target.original(1), 42)
        self.assertEqual(len(warnings), 1)
        self.assertFalse(any(record[1] is spec.fn for record in self.registry._hooks[spec.target]))

    def test_already_applied_target_rejects_late_registration(self):
        spec = self.spec(lambda *args: None)
        self.registry.register(spec.target, lambda *args: None, self.kind.AFTER)
        self.registry.apply_hooks()
        with self.assertRaisesRegex(AsapInstallError, "late installation"):
            self.adapter.register([spec])

    def test_replace_wrapped_metadata_cannot_fake_ownership(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        self.adapter.apply()
        actual = self.target.original
        self.target.original = self.runtime.registry._wrap_fn(actual, lambda value: 42, self.kind.REPLACE)
        self.assertIs(self.target.original.__wrapped__, actual)
        self.assertEqual(self.target.original(3), 42)
        with self.assertRaisesRegex(AsapInstallError, "execution chain"):
            self.adapter.ensure([spec])

    def test_unknown_wrapper_metadata_cannot_fake_ownership(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        self.adapter.apply()

        @functools.wraps(self.target.original)
        def wrapper(value):
            return 42

        self.target.original = wrapper
        with self.assertRaisesRegex(AsapInstallError, "unrecognized wrapper"):
            self.adapter.ensure([spec])

    def test_registry_patched_marker_is_not_evidence(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        self.registry._patched.add(spec.target)
        with self.assertRaisesRegex(AsapInstallError, "execution chain"):
            self.adapter.apply()

    def test_swallowed_apply_failure_is_reported(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        with patch.object(self.registry, "_apply_target", side_effect=ValueError("injected")):
            with self.assertLogs("sglang.srt.plugins.hook_registry", level="ERROR"):
                with self.assertRaisesRegex(AsapInstallError, "execution chain"):
                    self.adapter.apply()

    def test_parent_class_replacement_is_detected(self):
        class Owner:
            def run(self, value):
                return value + 1

        class Other:
            def run(self, value):
                return value + 10

        self.target.Owner = Owner
        spec = self.spec(lambda *args: None, attr="Owner.run", params=("self", "value"))
        self.adapter.register([spec])
        self.adapter.apply()
        self.registry.register("asap_hook_test.targets.Owner", Other, self.kind.REPLACE)
        with self.assertRaisesRegex(AsapInstallError, "replacement of parent"):
            self.registry.apply_hooks()

    def test_registry_guard_replacement_is_detected(self):
        spec = self.spec(lambda *args: None)
        self.adapter.register([spec])
        self.adapter.apply()
        self.registry.apply_hooks = classmethod(lambda cls: None)
        with self.assertRaisesRegex(AsapInstallError, "apply_hooks was replaced"):
            self.adapter.ensure([spec])

    def test_preexisting_own_callback_is_not_registered_twice(self):
        spec = self.spec(lambda *args: None)
        self.registry.register(spec.target, spec.fn, spec.type)
        with self.assertRaisesRegex(AsapInstallError, "outside this adapter"):
            self.adapter.register([spec])

    def test_classmethod_and_staticmethod_are_preserved(self):
        events = self.events

        class Owner:
            @classmethod
            def from_value(cls, value):
                return value + 1

            @staticmethod
            def from_static(value):
                return value + 2

        self.target.Owner = Owner
        cls_spec = self.spec(lambda *args: events.append("class"), attr="Owner.from_value", params=("cls", "value"))
        static_spec = self.spec(lambda *args: events.append("static"), attr="Owner.from_static")
        self.adapter.register([cls_spec, static_spec])
        self.adapter.apply()
        self.assertIsInstance(vars(Owner)["from_value"], classmethod)
        self.assertIsInstance(vars(Owner)["from_static"], staticmethod)
        self.assertEqual(Owner.from_value(1), 2)
        self.assertEqual(Owner.from_static(1), 3)
        self.assertEqual(events, ["class", "static"])

    def test_registration_and_application_are_idempotent(self):
        spec = self.spec(lambda *args: self.events.append("asap"))
        for _ in range(3):
            self.adapter.register([spec])
            self.adapter.apply()
        self.target.original(1)
        self.assertEqual(self.events.count("asap"), 1)
        self.assertEqual(len(self.registry._hooks[spec.target]), 1)

    def test_signature_drift_is_rejected_before_registration(self):
        spec = self.spec(lambda *args: None, params=("unexpected",))
        with self.assertRaisesRegex(AsapInstallError, "expected leading parameters"):
            self.adapter.register([spec])
        self.assertFalse(self.registry._hooks)

    def test_registration_failure_rolls_back_own_pending_callbacks(self):
        spec = self.spec(lambda *args: None)
        original_register = self.registry.register

        def failed_register(*args):
            original_register(*args)
            raise RuntimeError("injected registration failure")

        with patch.object(self.registry, "register", side_effect=failed_register):
            with self.assertRaisesRegex(AsapInstallError, "could not register"):
                self.adapter.register([spec])
        self.assertFalse(self.registry._hooks)


if __name__ == "__main__":
    unittest.main()
