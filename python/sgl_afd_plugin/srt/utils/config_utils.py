from __future__ import annotations

import dataclasses
import json
import logging
import os
import types
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


def _unwrap(hint: Any) -> tuple[Any, Arg]:
    """A[T, Arg(...)] / A[T, "help"] -> (T, Arg)."""
    if get_origin(hint) is not Annotated:
        return hint, Arg()
    base, *meta = get_args(hint)
    text = next((m for m in meta if isinstance(m, str)), "")
    arg = next((m for m in meta if isinstance(m, Arg)), Arg())
    return base, dataclasses.replace(arg, help=arg.help or text)


def _classify(ann: Any) -> tuple[str, Any, Any, list]:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(ann) if a is not type(None)]
        ann = inner[0] if inner else str
        origin = get_origin(ann)

    if ann is bool:
        return "bool", None, None, []

    if origin is Literal:
        values = list(get_args(ann))
        return "scalar", (type(values[0]) if values else str), None, values

    container = origin or ann
    if container in (list, tuple, set, frozenset):
        elems = [a for a in get_args(ann) if a is not Ellipsis]
        return "seq", (elems[0] if elems else str), container, []
    if container is dict:
        return "json", None, None, []
    if container in (int, float, str):
        return "scalar", container, None, []
    return "json", None, None, []


class _Field:
    def __init__(self, f: Any, hint: Any, prefix: str) -> None:
        self.field = f.name
        self.dest = f"{prefix}_{f.name}"
        self.env = f"{prefix.upper()}_{f.name.upper()}"
        self.default, self.default_factory = f.default, f.default_factory

        base, arg = _unwrap(hint)
        self.cli = arg.cli_name or f"--{prefix}-{f.name.replace('_', '-')}"
        self.aliases = list(arg.aliases or ())
        self.nargs = arg.nargs
        self.help = arg.help or f"{prefix} {f.name}"

        self.kind, self.type_, self.container, literal = _classify(base)
        self.choices = arg.choices or literal

    @property
    def required(self) -> bool:
        return self.default is self.default_factory is dataclasses.MISSING

    def cli_default(self) -> Any:
        if self.default is not dataclasses.MISSING:
            return self.default
        if self.default_factory is not dataclasses.MISSING:
            return self.default_factory()
        return None

    def cli_kwargs(self) -> dict:
        kwargs = {"required": self.required}
        if not self.required:
            kwargs["default"] = self.cli_default()
        if self.kind == "bool":
            return kwargs | {"action": "store_true"}
        kwargs["type"] = json.loads if self.kind == "json" else self.type_
        kwargs["metavar"] = self.field.upper()
        if self.kind == "seq" or self.nargs:
            kwargs["nargs"] = self.nargs or "+"
        if self.choices:
            kwargs["choices"] = self.choices
        return kwargs

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
            return self.cli_default()
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

    def install(self) -> None:
        from sglang.srt.plugins.hook_registry import HookType, plugin_hook

        @plugin_hook(
            "sglang.srt.server_args.ServerArgs.add_cli_args", type=HookType.AFTER
        )
        def add_args(result, parser):
            for fl in self._fields:
                parser.add_argument(
                    fl.cli,
                    *fl.aliases,
                    dest=fl.dest,
                    help=fl.help,
                    **fl.cli_kwargs(),
                )

        @plugin_hook(
            "sglang.srt.server_args.ServerArgs.from_cli_args", type=HookType.AROUND
        )
        def bridge(original_fn, cls, namespace):
            for fl in self._fields:
                value = getattr(namespace, fl.dest, None)
                if value is not None:
                    os.environ[fl.env] = fl.dump(value)
            return original_fn(cls, namespace)


def plugin_config(cls: type, *, prefix: str):
    """Install CLI hooks and return a getter for the config."""
    cfg = PluginConfig(cls, prefix=prefix)
    cfg.install()

    def get() -> Any:
        return cfg.read()

    get.__name__ = get.__qualname__ = f"get_{prefix}"
    get.__doc__ = "Get sglang Attention-FFN Disaggregation (AFD) plugin config."
    return get
