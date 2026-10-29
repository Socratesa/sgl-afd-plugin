"""Hook table.

Every sglang symbol the plugin attaches to is declared here as a ``HookSpec`` together with
the parameters it is expected to have (see ``compat``). Two tiers:

* ``base``: always on, cheap. Adds the ``--asap-*`` flags and watches ``ServerArgs.from_cli_args``.
* ``role``: only in an ASAP deployment (``--asap-role`` / ``ASAP_ROLE``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sgl_asap_plugin import compat

BASE = "base"
ROLE = "role"


@dataclass(frozen=True)
class HookSpec:
    target: str  # dotted path of the sglang function/method
    fn: Callable  # hook body, signature depends on ``type`` (see HookRegistry.register)
    type: HookType
    params: Tuple[str, ...]  # expected leading parameter names of the target
    tier: str


_registered: Dict[str, HookSpec] = {}


def all_specs(tier: Optional[str] = None) -> List[HookSpec]:
    from sgl_asap_plugin.hooks import cli, scheduler, validate

    specs = [*cli.SPECS, *validate.SPECS, *scheduler.SPECS]
    return [s for s in specs if tier is None or s.tier == tier]


def register_tier(tier: str) -> List[HookSpec]:
    """Preflight, then register every not-yet-registered spec of ``tier`` (idempotent).

    Registration only queues the hooks; sglang's loader (or ``HookRegistry.apply_hooks``)
    patches the targets afterwards.
    """
    todo = [s for s in all_specs(tier) if s.target not in _registered]
    compat.preflight(todo)
    for spec in todo:
        HookRegistry.register(spec.target, spec.fn, spec.type)
        _registered[spec.target] = spec
    return todo


def active_specs() -> List[HookSpec]:
    return list(_registered.values())
