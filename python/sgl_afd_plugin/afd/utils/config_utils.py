from __future__ import annotations

import dataclasses
import json
import logging
import os
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

A = Annotated


@dataclass(frozen=True)
class Arg:
    """CLI metadata for one field."""

    help: str = ""
    choices: list | None = None
    aliases: list | None = None
    cli_name: str | None = None
    nargs: str | None = None
    type_parser: Callable | None = None
    required: bool | None = None
    action: Any | None = None
    action_kwargs: dict | None = None
    const: Any | None = None
    no_cli: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unwrap(hint: Any) -> tuple[Any, Arg]:
    """A[T, Arg(...)] / A[T, "help"] -> (T, Arg)."""
    if get_origin(hint) is not Annotated:
        return hint, Arg()
    base, *meta = get_args(hint)
    arg = meta[0] if meta else Arg()
    return base, arg if isinstance(arg, Arg) else Arg(help=arg)


def _scalar_type(tp: Any) -> type:
    return tp if tp in (str, int, float) else str


def _classify(tp: Any) -> tuple[str, Any, Any, list]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(tp) if a is not type(None)]
        if len(inner) != 1:
            return "scalar", str, None, []
        tp = inner[0]
        origin = get_origin(tp)

    if tp is bool:
        return "bool", None, None, []
    if tp in (int, float, str):
        return "scalar", tp, None, []

    if origin is Literal:
        values = list(get_args(tp))
        return "scalar", _scalar_type(type(values[0])), None, values
    container = origin or tp
    if container in (list, tuple, set, frozenset):
        elems = [a for a in get_args(tp) if a is not Ellipsis]
        return "seq", (_scalar_type(elems[0]) if elems else str), container, []
    return "json", None, None, []


class _Field:
    def __init__(self, f: Any, hint: Any, prefix: str) -> None:
        self.field = f.name
        self.dest = f"{prefix}_{f.name}"
        self.env = f"{prefix.upper()}_{f.name.upper()}"
        self._default, self._default_factory = f.default, f.default_factory

        tp, arg = _unwrap(hint)
        self.cli = arg.cli_name or f"--{prefix}-{f.name.replace('_', '-')}"
        self.aliases = list(arg.aliases or ())
        self.help = arg.help or f"{prefix} {f.name}"
        self.nargs = arg.nargs
        self.no_cli = arg.no_cli
        self.type_parser = arg.type_parser
        self.required = arg.required is True or (arg.required is None and self.default is dataclasses.MISSING)
        self.action = arg.action
        self.action_kwargs = arg.action_kwargs or {}
        self.const = arg.const

        self.kind, self.type_, self.container, literal = _classify(tp)
        self.choices = arg.choices or literal

    @property
    def default(self) -> Any:
        if self._default is not dataclasses.MISSING:
            return self._default
        if self._default_factory is not dataclasses.MISSING:
            return self._default_factory()
        return dataclasses.MISSING

    def cli_kwargs(self) -> dict:
        kwargs = {} if self.default is dataclasses.MISSING else {"default": self.default}
        if self.action is not None:
            return kwargs | {"action": self.action} | self.action_kwargs
        kwargs["required"] = self.required
        if self.kind == "bool":
            return kwargs | {"action": "store_true"}
        kwargs["type"] = self.parser()
        kwargs["metavar"] = self.field.upper()
        if self.kind == "seq" and self.type_parser is None:
            kwargs["nargs"] = self.nargs or "+"
        elif self.nargs:
            kwargs["nargs"] = self.nargs
        if self.choices:
            kwargs["choices"] = self.choices
        if self.const is not None:
            kwargs["const"] = self.const
        return kwargs

    def parser(self) -> Any:
        return self.type_parser or (json.loads if self.kind == "json" else self.type_)

    def dump(self, value: Any) -> str:
        if self.kind == "bool":
            return "1" if value else "0"
        if self.kind == "seq":
            return ",".join(map(str, value))
        if self.kind == "json":
            return json.dumps(value)
        return str(value)

    def resolve(self, raw: str | None) -> Any:
        if raw is None:
            default = self.default
            if default is dataclasses.MISSING:
                raise ValueError(f"{self.cli}: no value (env {self.env} unset)")
            return default
        try:
            if self.kind == "bool":
                return raw.strip().lower() in frozenset({"1", "true", "yes", "on"})
            if self.kind == "seq":
                return self.container(self.type_(p) for p in raw.split(",") if p.strip())
            if self.kind == "json":
                return json.loads(raw)
            return self.type_(raw)
        except Exception as exc:
            raise ValueError(f"{self.env}={raw!r} invalid: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PluginConfig:
    def __init__(self, cls: type, *, prefix: str) -> None:
        self._cls = cls
        self._prefix = prefix
        self._instance = None

        hints = get_type_hints(cls, include_extras=True)
        self._fields = tuple(
            _Field(f, hints[f.name], prefix) for f in dataclasses.fields(cls)
        )

    def read(self) -> Any:
        if self._instance is None:
            self._instance = self._cls(
                **{fl.field: fl.resolve(os.environ.get(fl.env)) for fl in self._fields}
            )
            logger.info(f"{self._prefix}_config={dataclasses.asdict(self._instance)}")
        return self._instance

    def add_cli_args(self, parser: Any) -> None:
        for fl in self._fields:
            if fl.no_cli:
                continue
            parser.add_argument(
                fl.cli,
                *fl.aliases,
                dest=fl.dest,
                help=fl.help,
                **fl.cli_kwargs(),
            )

    def export_env(self, namespace: Any) -> None:
        """Copy parsed values onto os.environ so subprocesses inherit them."""
        for fl in self._fields:
            value = getattr(namespace, fl.dest, None)
            if value is not None:
                os.environ[fl.env] = fl.dump(value)


def plugin_config(cls: type, *, prefix: str) -> PluginConfig:
    """Build a config group. Its CLI/env hooks live in srt/server_args.py."""
    return PluginConfig(cls, prefix=prefix)
