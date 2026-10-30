"""SGLang hook compatibility boundary.

The registry records intent separately from the callable it installs. We check
both: queued conflicts before applying hooks and actual wrapper closures after
application. ``__wrapped__`` is signature metadata, not an execution guarantee:
SGLang's REPLACE wrapper retains it while bypassing the original callable.

Registration and application follow SGLang's single-threaded startup contract.
Runtime mutation after startup is unsupported; the next check detects drift.
"""

from __future__ import annotations

import functools
import inspect
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from sglang.srt.plugins import hook_registry as _upstream

from sgl_asap_plugin.config import AsapInstallError

HookType = _upstream.HookType


class _Spec(Protocol):
    target: str
    type: Any
    params: tuple[str, ...]
    fn: Callable


@dataclass(frozen=True)
class _Entry:
    spec: _Spec
    original: Callable
    initial: Callable


def _resolve(target: str) -> Callable:
    owner_path, separator, attr = target.rpartition(".")
    if not separator:
        raise ValueError("a hook target must include its owner")
    raw = inspect.getattr_static(pkgutil.resolve_name(owner_path), attr)
    return raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw


def _error(problems: Iterable[str]) -> AsapInstallError:
    return AsapInstallError("[ASAP] hook compatibility check failed:\n  - " + "\n  - ".join(problems))


class HookAdapter:
    """Own ASAP registrations and validate their effective execution paths.

    Explicit preflight/register/ensure calls raise ``AsapInstallError``. The
    apply guard instead delegates failures to ``on_failure`` so the caller can
    abort an opted-in launch or keep an unrelated SGLang launch running.
    """

    def __init__(self, on_failure: Callable[[AsapInstallError], None]):
        self._on_failure = on_failure
        self._registry = _upstream.HookRegistry
        self._entries: dict[str, _Entry] = {}
        self._codes: dict[Any, Any] | None = None
        self._guard: Callable | None = None

    def _wrapper_codes(self) -> dict[Any, Any]:
        if self._codes is None:
            try:
                sentinel = lambda *args, **kwargs: None
                codes = {}
                for kind in HookType:
                    wrapper = _upstream._wrap_fn(sentinel, sentinel, kind)
                    expected = {"hook"} if kind == HookType.REPLACE else {"hook", "original_fn"}
                    if set(wrapper.__code__.co_freevars) != expected:
                        raise ValueError(f"unexpected {kind.value} wrapper closure")
                    codes[wrapper.__code__] = kind
                if len(codes) != len(HookType):
                    raise ValueError("hook kinds do not have distinct wrapper implementations")
                if not isinstance(self._registry._hooks, dict) or not isinstance(self._registry._patched, set):
                    raise ValueError("unexpected hook registry storage")
                self._codes = codes
            except Exception as exc:
                raise _error([f"unsupported SGLang HookRegistry: {type(exc).__name__}: {exc}"]) from exc
        return self._codes

    def _chain(self, function: Callable) -> tuple[list[tuple[Any, Callable]], Callable | None]:
        codes = self._wrapper_codes()
        chain = []
        seen = set()
        while getattr(function, "__code__", None) in codes:
            if id(function) in seen:
                raise ValueError("cyclic hook execution chain")
            seen.add(id(function))
            kind = codes[function.__code__]
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            hook = cells["hook"].cell_contents
            chain.append((kind, hook))
            if kind == HookType.REPLACE:
                return chain, None
            function = cells["original_fn"].cell_contents
        if hasattr(function, "__wrapped__"):
            raise ValueError("unrecognized wrapper; execution through __wrapped__ is not guaranteed")
        return chain, function

    def _queued_problems(self, specs: Iterable[_Spec]) -> list[str]:
        problems = []
        self._wrapper_codes()
        for spec in specs:
            for target, records in self._registry._hooks.items():
                if spec.target.startswith(target + ".") and any(record[0] == HookType.REPLACE for record in records):
                    problems.append(f"{spec.target}: conflicts with replacement of parent {target}")
            own_count = 0
            for record in self._registry._hooks.get(spec.target, ()):
                if len(record) != 3:
                    problems.append(f"{spec.target}: unsupported registration record")
                    continue
                kind, hook, _source = record
                if hook is spec.fn and kind == spec.type:
                    own_count += 1
                elif kind in (HookType.REPLACE, HookType.AROUND):
                    problems.append(f"{spec.target}: conflicts with another {kind.value} callback")
            if own_count > 1:
                problems.append(f"{spec.target}: ASAP callback registered more than once")
            if own_count and spec.target not in self._entries:
                problems.append(f"{spec.target}: ASAP callback was registered outside this adapter")
        return problems

    def preflight(self, specs: Iterable[_Spec]) -> None:
        specs = list(specs)
        problems = self._queued_problems(specs)
        for spec in specs:
            try:
                function = _resolve(spec.target)
                names = tuple(inspect.signature(function).parameters)
                if names[:len(spec.params)] != tuple(spec.params):
                    problems.append(f"{spec.target}: expected leading parameters {list(spec.params)}, "
                                    f"found {list(names[:len(spec.params)])}")
                chain, _terminal = self._chain(function)
                for kind, hook in chain:
                    if kind in (HookType.REPLACE, HookType.AROUND) and hook is not spec.fn:
                        problems.append(f"{spec.target}: another {kind.value} callback is already applied")
                if spec.target in self._registry._patched and spec.target not in self._entries:
                    problems.append(f"{spec.target}: target was already applied; late installation is unsupported")
            except Exception as exc:
                problems.append(f"{spec.target}: cannot inspect ({type(exc).__name__}: {exc})")
        if problems:
            raise _error(problems)

    def register(self, specs: Iterable[_Spec]) -> None:
        specs = list(specs)
        targets = [spec.target for spec in specs]
        if len(targets) != len(set(targets)):
            raise _error(["a registration batch contains duplicate targets"])
        for spec in specs:
            existing = self._entries.get(spec.target)
            if existing and (existing.spec.fn is not spec.fn or existing.spec.type != spec.type):
                raise _error([f"{spec.target}: ASAP registration changed during startup"])
        self.preflight(specs)
        new = [spec for spec in specs if spec.target not in self._entries]
        initial = {spec.target: _resolve(spec.target) for spec in new}
        originals = {target: self._chain(function)[1] for target, function in initial.items()}
        self._install_guard()
        added = []
        try:
            for spec in new:
                self._registry.register(spec.target, spec.fn, spec.type)
                self._entries[spec.target] = _Entry(spec, originals[spec.target], initial[spec.target])
                added.append(spec)
        except Exception as exc:
            for spec in new:
                self._remove_queued(spec)
            for spec in added:
                self._entries.pop(spec.target, None)
            raise _error([f"could not register hooks: {type(exc).__name__}: {exc}"]) from exc

    def _remove_queued(self, spec: _Spec) -> None:
        records = self._registry._hooks.get(spec.target)
        if records is not None:
            records[:] = [record for record in records if not (record[0] == spec.type and record[1] is spec.fn)]
            if not records:
                self._registry._hooks.pop(spec.target, None)

    def _drop_unapplied(self) -> None:
        for target, entry in list(self._entries.items()):
            if target not in self._registry._patched:
                self._remove_queued(entry.spec)
                del self._entries[target]

    def _install_guard(self) -> None:
        if self._guard is not None:
            if getattr(self._registry.apply_hooks, "__func__", None) is not self._guard:
                raise _error(["HookRegistry.apply_hooks was replaced after ASAP registration"])
            return
        original = self._registry.apply_hooks

        @functools.wraps(original)
        def guarded_apply(_registry):
            try:
                problems = self._queued_problems(entry.spec for entry in self._entries.values())
                for target, entry in self._entries.items():
                    if target not in self._registry._patched:
                        try:
                            if _resolve(target) is not entry.initial:
                                problems.append(f"{target}: callable changed before hook application")
                        except Exception as exc:
                            problems.append(f"{target}: cannot resolve before hook application ({exc})")
                if problems:
                    raise _error(problems)
            except AsapInstallError as exc:
                self._on_failure(exc)
                self._drop_unapplied()
            original()
            try:
                self.ensure(entry.spec for entry in self._entries.values())
            except AsapInstallError as exc:
                self._on_failure(exc)
                self._drop_unapplied()

        self._guard = guarded_apply
        self._registry.apply_hooks = classmethod(guarded_apply)

    def apply(self) -> None:
        self._install_guard()
        self._registry.apply_hooks()

    def _installation_problems(self, specs: Iterable[_Spec]) -> list[str]:
        specs = list(specs)
        problems = self._queued_problems(specs)
        if self._guard is not None and getattr(self._registry.apply_hooks, "__func__", None) is not self._guard:
            problems.append("HookRegistry.apply_hooks was replaced after ASAP registration")
        for spec in specs:
            entry = self._entries.get(spec.target)
            if entry is None:
                problems.append(f"{spec.target}: callback is not owned by this adapter")
                continue
            try:
                chain, terminal = self._chain(_resolve(spec.target))
                matches = [(kind, hook) for kind, hook in chain if hook is spec.fn and kind == spec.type]
                if len(matches) != 1:
                    problems.append(f"{spec.target}: ASAP callback is not present exactly once in the execution chain")
                for kind, hook in chain:
                    if kind in (HookType.REPLACE, HookType.AROUND) and hook is not spec.fn:
                        problems.append(f"{spec.target}: execution is controlled by another {kind.value} callback")
                if terminal is not None and terminal is not entry.original:
                    problems.append(f"{spec.target}: original callable changed after registration")
            except Exception as exc:
                problems.append(f"{spec.target}: cannot verify execution chain ({type(exc).__name__}: {exc})")
        return problems

    def missing(self, specs: Iterable[_Spec]) -> list[str]:
        return sorted(spec.target for spec in specs if self._installation_problems([spec]))

    def ensure(self, specs: Iterable[_Spec]) -> None:
        problems = self._installation_problems(specs)
        if problems:
            raise _error(problems)
