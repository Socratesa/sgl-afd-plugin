"""Compatibility with the sglang internals the hooks attach to.

The plugin patches private sglang symbols. sglang's loader only logs a failure when a patch
cannot be applied, so a renamed function or changed signature would silently turn a hook
off. Every hook therefore declares its target and the leading parameter names it expects:
``preflight`` checks them before registration, ``ensure_installed`` checks that the
patches were actually applied.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import pkgutil

from sgl_asap_plugin.errors import AsapInstallError

logger = logging.getLogger("sgl_asap_plugin")

TESTED_SGLANG = ("0.5.22",)


def sglang_version():
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return None


def warn_if_untested() -> None:
    version = sglang_version()
    if version not in TESTED_SGLANG:
        logger.warning("[ASAP] sglang %s is not in the tested set %s; relying on hook preflight",
                       version, TESTED_SGLANG)


def _resolve(target: str):
    owner_path, _, attr = target.rpartition(".")
    owner = pkgutil.resolve_name(owner_path)
    raw = inspect.getattr_static(owner, attr)
    return raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw


def _problem(spec):
    try:
        fn = _resolve(spec.target)
    except Exception as exc:  # ImportError / AttributeError / ValueError from resolve_name
        return f"{spec.target}: cannot resolve ({type(exc).__name__}: {exc})"
    if not callable(fn):
        return f"{spec.target}: not callable"
    if spec.params:
        try:
            names = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError) as exc:
            return f"{spec.target}: signature unreadable ({exc})"
        if names[: len(spec.params)] != list(spec.params):
            return (f"{spec.target}: expected leading parameters {list(spec.params)}, "
                    f"found {names[: len(spec.params)]}")
    return None


def preflight(specs) -> None:
    """Raise AsapInstallError listing every spec whose target is missing or changed."""
    problems = [p for p in (_problem(s) for s in specs) if p]
    if problems:
        raise AsapInstallError(
            f"[ASAP] hook targets do not match sglang {sglang_version()}:\n  - " + "\n  - ".join(problems))


def missing_patches(specs) -> list:
    from sglang.srt.plugins.hook_registry import HookRegistry

    return sorted({s.target for s in specs} - HookRegistry._patched)


def ensure_installed(specs) -> None:
    missing = missing_patches(specs)
    if missing:
        raise AsapInstallError(
            "[ASAP] hooks registered but not applied (look for 'Failed to apply hooks' in the log):\n  - "
            + "\n  - ".join(missing))
