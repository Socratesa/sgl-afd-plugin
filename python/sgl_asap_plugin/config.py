"""Plugin configuration: ``--asap-*`` CLI flags, ``ASAP_*`` environment round trip, syntax checks.

``ServerArgs`` is a sealed struct that cannot carry plugin fields, so the parsed config is
exported to ``ASAP_*`` environment variables, which every child process (DPC, schedulers)
inherits. Checks against the resolved ServerArgs live in ``validation.py``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.transport.protocol import MAX_TOKENS

ROLES = ("attn", "ffn")
ENV_ROLE = "ASAP_ROLE"
# below 10000: this host hands ports 10000-61000 to outgoing connections, which can take a fixed port first
DEFAULT_RDV_ADDR = "127.0.0.1:9600"
DEFAULT_N_MAX = 16384
DEFAULT_SLOTS = 2
MAX_N_MAX = MAX_TOKENS  # the transport doorbell's n_tokens field is 17 bits wide
MAX_SLOTS = 8

_ADDR_RE = re.compile(r"^[A-Za-z0-9_.\-]+:(\d{1,5})$")
# dataclass field -> argparse dest (the role is handled separately)
_CLI_DESTS = {
    "rdv_addr": "asap_rdv_addr",
    "n_max": "asap_n_max",
    "slots": "asap_slots",
    "num_replicas": "asap_num_replicas",
    "replica_tp": "asap_replica_tp",
    "ffn_ranks": "asap_ffn_ranks",
}
_INT_FIELDS = ("n_max", "slots", "num_replicas", "replica_tp", "ffn_ranks")


@dataclass(frozen=True)
class AsapConfig:
    role: str
    rdv_addr: str = DEFAULT_RDV_ADDR  # F listens here, A connects (handshake, PR-03)
    n_max: int = DEFAULT_N_MAX  # max tokens per micro-batch per TP row (arena sizing)
    slots: int = DEFAULT_SLOTS  # in-flight micro-batch slots per A rank (dual-batch needs 2)
    num_replicas: Optional[int] = None  # D, the number of A replicas (required by F)
    replica_tp: Optional[int] = None  # T, TP width of one A replica (required by F)
    ffn_ranks: Optional[int] = None  # E, number of F ranks (required by A; on F it must equal --ep-size)

    def __post_init__(self):
        problems = []
        if self.role not in ROLES:
            problems.append(f"role must be one of {ROLES}, got {self.role!r}")
        match = _ADDR_RE.match(self.rdv_addr or "")
        if not match or not 1 <= int(match.group(1)) <= 65535:
            problems.append(f"rdv_addr must look like HOST:PORT, got {self.rdv_addr!r}")
        if not 1 <= self.n_max <= MAX_N_MAX:
            problems.append(f"n_max must be in [1, {MAX_N_MAX}], got {self.n_max}")
        if not 1 <= self.slots <= MAX_SLOTS:
            problems.append(f"slots must be in [1, {MAX_SLOTS}], got {self.slots}")
        for name in ("num_replicas", "replica_tp", "ffn_ranks"):
            value = getattr(self, name)
            if value is not None and value < 1:
                problems.append(f"{name} must be >= 1, got {value}")
            if self.role == "ffn" and value is None and name != "ffn_ranks":
                problems.append(f"{name} is required for the ffn role")
        if problems:
            raise AsapConfigError("[ASAP] invalid config: " + "; ".join(problems))

    def to_env(self) -> dict:
        env = {ENV_ROLE: self.role}
        for field in _CLI_DESTS:
            value = getattr(self, field)
            if value is not None:
                env["ASAP_" + field.upper()] = str(value)
        return env

    def export_env(self, environ: Optional[dict] = None) -> None:
        """Replace stale ASAP_* settings in ``environ`` (default os.environ) with this config."""
        environ = os.environ if environ is None else environ
        for field in _CLI_DESTS:
            environ.pop("ASAP_" + field.upper(), None)
        environ.update(self.to_env())

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> Optional["AsapConfig"]:
        environ = os.environ if environ is None else environ
        role = environ.get(ENV_ROLE)
        if not role:
            return None
        kwargs = {}
        for field in _CLI_DESTS:
            raw = environ.get("ASAP_" + field.upper())
            if raw is None:
                continue
            try:
                kwargs[field] = int(raw) if field in _INT_FIELDS else raw
            except ValueError:
                raise AsapConfigError(f"[ASAP] ASAP_{field.upper()} must be an integer, got {raw!r}") from None
        return cls(role=role, **kwargs)

    @classmethod
    def from_namespace(cls, ns) -> Optional["AsapConfig"]:
        """Build from a parsed argparse namespace; None when no ``--asap-*`` option was given."""
        role = getattr(ns, "asap_role", None)
        given = {f: getattr(ns, d) for f, d in _CLI_DESTS.items() if getattr(ns, d, None) is not None}
        if role is None:
            if given:
                raise AsapConfigError("[ASAP] --asap-* options were given without --asap-role {attn,ffn}")
            return None
        return cls(role=role, **given)


def add_cli_args(parser) -> None:
    """Register the ``--asap-*`` flags. All defaults are None so that "unset" is detectable."""
    group = parser.add_argument_group("ASAP (prefill Attention-FFN disaggregation)")
    group.add_argument("--asap-role", choices=ROLES, default=None,
                       help="Role of this launch. 'attn' runs on sglang serve (plain --dp-size D replicas); "
                            "'ffn' is started with python -m sgl_asap_plugin.ffn_serve.")
    group.add_argument("--asap-rdv-addr", default=None, metavar="HOST:PORT",
                       help=f"Rendezvous address of the F service (default {DEFAULT_RDV_ADDR}).")
    group.add_argument("--asap-n-max", type=int, default=None,
                       help=f"Max tokens per micro-batch per TP row; sizes the arenas (default {DEFAULT_N_MAX}).")
    group.add_argument("--asap-slots", type=int, default=None,
                       help=f"In-flight micro-batch slots per A rank (default {DEFAULT_SLOTS}).")
    group.add_argument("--asap-num-replicas", type=int, default=None,
                       help="D, number of A replicas. Required by F; on A it must equal --dp-size.")
    group.add_argument("--asap-replica-tp", type=int, default=None,
                       help="T, TP width of one A replica. Required by F; on A it must equal --tp-size.")
    group.add_argument("--asap-ffn-ranks", type=int, default=None,
                       help="E, number of F ranks. Required by A; on F it must equal --ep-size. "
                            "F rank f listens on the --asap-rdv-addr port + f.")
