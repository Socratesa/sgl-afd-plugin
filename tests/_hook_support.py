"""Load real upstream hook code without importing its GPU dependencies."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


def _source_directory() -> Path:
    configured = os.environ.get("ASAP_TEST_SGLANG_SOURCE")
    if configured:
        root = Path(configured)
        candidates = [root / "python/sglang/srt/plugins", root / "sglang/srt/plugins"]
    else:
        spec = importlib.util.find_spec("sglang")
        candidates = [Path(p) / "srt/plugins" for p in (spec.submodule_search_locations or ())] if spec else []
    for candidate in candidates:
        if all(path.is_file() for path in (
            candidate / "hook_registry.py", candidate / "__init__.py", candidate.parent / "environ.py"
        )):
            return candidate
    if configured:
        raise RuntimeError(f"ASAP_TEST_SGLANG_SOURCE has no hook registry and loader: {configured}")
    raise unittest.SkipTest("set ASAP_TEST_SGLANG_SOURCE to a supported SGLang checkout")


class RegistrySandbox:
    def module(self, name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, _, attr = name.rpartition(".")
            setattr(self.module(parent), attr, module)
        return module

    def load(self, name: str, path: Path) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if "." in name:
            parent, _, attr = name.rpartition(".")
            setattr(self.module(parent), attr, module)
        spec.loader.exec_module(module)
        return module


@contextmanager
def registry_sandbox():
    """Yield isolated real registry/loader modules and a synthetic-module helper.

    The environment module, plugin loader and hook registry are read verbatim
    from the selected SGLang source. Only parent package imports are isolated.
    """
    directory = _source_directory()
    prefixes = ("sglang", "asap_hook_test")
    plugin_names = {"sgl_asap_plugin.hooks", "sgl_asap_plugin.compat"}

    def selected(name):
        return name in plugin_names or any(name == p or name.startswith(p + ".") for p in prefixes)

    saved = {name: module for name, module in sys.modules.items() if selected(name)}
    package = sys.modules.get("sgl_asap_plugin")
    saved_attrs = {name: getattr(package, name) for name in ("hooks", "compat") if hasattr(package, name)}
    for name in saved:
        del sys.modules[name]
    if package:
        for name in ("hooks", "compat"):
            package.__dict__.pop(name, None)
    sandbox = RegistrySandbox()
    try:
        sandbox.module("sglang.srt.plugins")
        sandbox.load("sglang.srt.environ", directory.parent / "environ.py")
        sandbox.registry = sandbox.load("sglang.srt.plugins.hook_registry", directory / "hook_registry.py")
        sandbox.loader = sandbox.load("sglang.srt.plugins", directory / "__init__.py")
        sandbox.loader.hook_registry = sandbox.registry
        yield sandbox
    finally:
        for name in list(sys.modules):
            if selected(name):
                del sys.modules[name]
        sys.modules.update(saved)
        package = sys.modules.get("sgl_asap_plugin")
        if package:
            for name in ("hooks", "compat"):
                package.__dict__.pop(name, None)
            package.__dict__.update(saved_attrs)
