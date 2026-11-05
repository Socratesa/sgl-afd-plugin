"""The plugin's side of ``sglang.srt.server_args``.

``add_cli_args`` and ``from_cli_args`` hook the ServerArgs methods of the same name:
the ``--asap-*`` flags are parsed with SGLang's own, and on ``--asap-role attn|ffn``
the role's hooks are registered before ServerArgs is resolved. The launcher exports
the flags as ``ASAP_*`` environment variables (upper-cased field names), which the
processes it starts inherit; ``get_asap_args()`` reads them back there.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from typing import List, Optional, Tuple

import msgspec
from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass

from sgl_asap_plugin.srt.arg_groups.fields.asap import Asap
from sgl_asap_plugin.transport.layout import Geometry

logger = logging.getLogger(__name__)

ENV_ROLE = "ASAP_ROLE"
TESTED_SGLANG = ("0.5.22",)

_asap_args: Optional[Asap] = None


def add_cli_args(result, parser) -> None:
    """AFTER ``ServerArgs.add_cli_args``."""
    add_cli_args_from_dataclass(parser.add_argument_group("ASAP"), Asap)


def from_cli_args(cls, args) -> None:
    """BEFORE ``ServerArgs.from_cli_args``: on --asap-role attn|ffn, register the role's
    hooks before ServerArgs resolution runs, and export the flags."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    from sgl_asap_plugin import hooks

    asap_args = asap_args_from_cli(args)
    set_asap_args(asap_args)
    if asap_args.asap_role == "null":
        return None
    export_env(asap_args)
    hooks.register_role_hooks(asap_args)
    HookRegistry.apply_hooks()
    hooks.check_hooks_applied()
    version = importlib.metadata.version("sglang")
    if version not in TESTED_SGLANG:
        logger.warning(f"ASAP is tested with sglang {TESTED_SGLANG}, not {version}")
    return None


def asap_args_from_cli(args) -> Asap:
    """The --asap-* values of a parsed argparse namespace."""
    defaults = Asap()
    given = {
        f.name: getattr(args, f.name)
        for f in msgspec.structs.fields(Asap)
        if getattr(args, f.name, None) != getattr(defaults, f.name)
    }
    if given and given.get("asap_role", "null") == "null":
        flags = ", ".join("--" + name.replace("_", "-") for name in sorted(given))
        raise ValueError(f"{flags} given without --asap-role attn|ffn")
    return Asap(**given)


def asap_args_from_env() -> Asap:
    values = {
        f.name: os.environ[f.name.upper()]
        for f in msgspec.structs.fields(Asap)
        if f.name.upper() in os.environ
    }
    return msgspec.convert(values, Asap, strict=False)


def export_env(args: Asap) -> None:
    for field in msgspec.structs.fields(args):
        os.environ.pop(field.name.upper(), None)
        value = getattr(args, field.name)
        if value is not None:
            os.environ[field.name.upper()] = str(value)


def set_asap_args(args: Asap) -> None:
    global _asap_args
    _asap_args = args


def get_asap_args() -> Asap:
    """This process's ASAP arguments; role "null" outside an ASAP deployment."""
    global _asap_args
    if _asap_args is None:
        _asap_args = asap_args_from_env()
    return _asap_args


def is_afd_attn(args: Asap) -> bool:
    """An attention role whose routed experts run on the FFN ranks."""
    return args.asap_role == "attn" and args.asap_ffn_size is not None


def bootstrap_endpoints(args: Asap) -> List[Tuple[str, int]]:
    """(host, port) of every FFN rank."""
    host, _, port = args.asap_bootstrap_addr.rpartition(":")
    return [(host, int(port) + f) for f in range(args.asap_ffn_size)]


def transport_geometry(args: Asap, hidden: int, topk: int) -> Geometry:
    """The buffer layout both roles agree on (``transport/layout.py``)."""
    return Geometry(
        num_a=args.asap_attn_dp_size * args.asap_attn_tp_size,
        group=args.asap_attn_tp_size,
        slots=args.asap_max_micro_batches,
        n_max=args.asap_max_dispatch_tokens,
        hidden=hidden,
        topk=topk,
        num_f=args.asap_ffn_size,
    )
