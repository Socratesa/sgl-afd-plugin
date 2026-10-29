"""ASAP deployment configuration: the ``--asap-*`` flags, their ``ASAP_*`` environment form, and the rules a
deployment must satisfy, checked against sglang's resolved ServerArgs.

Topology. A (attention role): D replicas of T ranks, plain data parallelism (``sglang serve --dp-size D
--tp-size T``). Each replica is its own torch.distributed world, so replicas never wait for each other: no
attention DP, no per-step all-gather, no idle batches. F (expert role, ``python -m sgl_asap_plugin.ffn``): E
ranks of pure expert parallelism that never talk to each other. A rank ``a = replica * T + tp_rank``; F rank f
owns the logical experts ``[f * P, (f + 1) * P)`` with ``P = ceil(n_experts / E)`` (ids past n_experts are
padding that nobody routes to). F rank f listens on the ``--asap-rdv-addr`` port + f.

ServerArgs is a sealed struct that cannot carry plugin fields, so the launcher exports the parsed config to
``ASAP_*`` environment variables, which every child process (DP controller, schedulers) inherits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

from sgl_asap_plugin.transport.layout import MAX_TOKENS, Geometry


class AsapConfigError(ValueError):
    """The requested deployment is not supported; the message says what to change."""


class AsapInstallError(RuntimeError):
    """The plugin's hooks cannot be attached to this sglang build."""


ROLES = ("attn", "ffn")
ENV_ROLE = "ASAP_ROLE"
# below 10000: this host hands ports 10000-61000 to outgoing connections, which may take a fixed port first
DEFAULT_RDV_ADDR = "127.0.0.1:9600"
DEFAULT_N_MAX = 16384
DEFAULT_SLOTS = 2
MAX_SLOTS = 8
_ADDR_RE = re.compile(r"^[A-Za-z0-9_.\-]+:(\d{1,5})$")
_INT_FIELDS = ("n_max", "slots", "num_replicas", "replica_tp", "ffn_ranks")
_FIELDS = ("rdv_addr",) + _INT_FIELDS  # dataclass field <-> --asap-<field> <-> ASAP_<FIELD>


@dataclass(frozen=True)
class AsapConfig:
    """The --asap-* flags of one launch; __post_init__ reports every invalid field at once."""
    role: str
    rdv_addr: str = DEFAULT_RDV_ADDR  # F rank 0's handshake address; rank f uses port + f
    n_max: int = DEFAULT_N_MAX  # max tokens per message (one micro-batch of one TP rank): sizes the arenas
    slots: int = DEFAULT_SLOTS  # micro-batches in flight per A rank (the synchronous forward uses 1)
    num_replicas: Optional[int] = None  # D. Required on F; on A it defaults to --dp-size
    replica_tp: Optional[int] = None  # T. Required on F; on A it defaults to --tp-size
    ffn_ranks: Optional[int] = None  # E. On A it switches AFD on; on F it defaults to --ep-size

    def __post_init__(self):
        problems = []
        if self.role not in ROLES:
            problems.append(f"role must be one of {ROLES}, got {self.role!r}")
        match = _ADDR_RE.match(self.rdv_addr or "")
        if not match or not 1 <= int(match.group(1)) <= 65535:
            problems.append(f"rdv_addr must look like HOST:PORT, got {self.rdv_addr!r}")
        if not 1 <= self.n_max <= MAX_TOKENS:
            problems.append(f"n_max must be in [1, {MAX_TOKENS}] (the doorbell's token field), got {self.n_max}")
        if not 1 <= self.slots <= MAX_SLOTS:
            problems.append(f"slots must be in [1, {MAX_SLOTS}], got {self.slots}")
        for name in ("num_replicas", "replica_tp", "ffn_ranks"):
            value = getattr(self, name)
            if value is not None and value < 1:
                problems.append(f"{name} must be >= 1, got {value}")
        if self.role == "ffn" and (self.num_replicas is None or self.replica_tp is None):
            problems.append("the ffn role needs --asap-num-replicas D and --asap-replica-tp T (they size its arena)")
        if problems:
            raise AsapConfigError("[ASAP] invalid config: " + "; ".join(problems))

    @property
    def afd(self) -> bool:
        """An attention role whose routed experts live on the F service."""
        return self.role == "attn" and self.ffn_ranks is not None

    def endpoints(self) -> List[Tuple[str, int]]:
        """Handshake address of every F rank: the rdv port + f."""
        host, _, port = self.rdv_addr.rpartition(":")
        return [(host, int(port) + f) for f in range(self.ffn_ranks)]

    def geometry(self, hidden: int, topk: int) -> Geometry:
        """The transport geometry; needs num_replicas, replica_tp and ffn_ranks to be known."""
        return Geometry(num_a=self.num_replicas * self.replica_tp, group=self.replica_tp, slots=self.slots,
                        n_max=self.n_max, hidden=hidden, topk=topk, num_f=self.ffn_ranks)

    def with_defaults(self, **values) -> "AsapConfig":
        """A copy whose unset (None) fields take ``values``."""
        given = {f: getattr(self, f) for f in ("role",) + _FIELDS}
        given.update({k: v for k, v in values.items() if given.get(k) is None})
        return AsapConfig(**given)

    # ---- environment round trip ----
    def export_env(self, environ: Optional[dict] = None) -> None:
        """Replace every ASAP_* setting in ``environ`` (default os.environ) with this config."""
        environ = os.environ if environ is None else environ
        for name in _FIELDS:
            environ.pop("ASAP_" + name.upper(), None)
            if getattr(self, name) is not None:
                environ["ASAP_" + name.upper()] = str(getattr(self, name))
        environ[ENV_ROLE] = self.role

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> Optional["AsapConfig"]:
        """The config the launcher exported (export_env); None outside an ASAP deployment."""
        environ = os.environ if environ is None else environ
        if not environ.get(ENV_ROLE):
            return None
        kwargs = {}
        for name in _FIELDS:
            raw = environ.get("ASAP_" + name.upper())
            if raw is not None:
                try:
                    kwargs[name] = int(raw) if name in _INT_FIELDS else raw
                except ValueError:
                    raise AsapConfigError(f"[ASAP] ASAP_{name.upper()} must be an integer, got {raw!r}") from None
        return cls(role=environ[ENV_ROLE], **kwargs)

    @classmethod
    def from_namespace(cls, ns) -> Optional["AsapConfig"]:
        """From a parsed argparse namespace; None when no ``--asap-*`` option was given."""
        given = {f: getattr(ns, "asap_" + f) for f in _FIELDS if getattr(ns, "asap_" + f, None) is not None}
        role = getattr(ns, "asap_role", None)
        if role is None:
            if given:
                raise AsapConfigError("[ASAP] --asap-* options were given without --asap-role {attn,ffn}")
            return None
        return cls(role=role, **given)


def add_cli_args(parser) -> None:
    """Register the ``--asap-*`` flags. Every default is None so that "not given" stays detectable."""
    group = parser.add_argument_group("ASAP (prefill Attention-FFN disaggregation)")
    group.add_argument("--asap-role", choices=ROLES, default=None,
                       help="'attn' runs under sglang serve (--dp-size D plain replicas); "
                            "'ffn' is the expert service: python -m sgl_asap_plugin.ffn")
    group.add_argument("--asap-rdv-addr", default=None, metavar="HOST:PORT",
                       help=f"Handshake address of F rank 0; rank f listens on port + f (default {DEFAULT_RDV_ADDR}).")
    group.add_argument("--asap-n-max", type=int, default=None,
                       help=f"Max tokens per message; sizes the arenas, must be >= --chunked-prefill-size "
                            f"(default {DEFAULT_N_MAX}).")
    group.add_argument("--asap-slots", type=int, default=None,
                       help=f"Micro-batches in flight per A rank (default {DEFAULT_SLOTS}).")
    group.add_argument("--asap-num-replicas", type=int, default=None,
                       help="D, the number of A replicas. Required on F; on A it defaults to --dp-size.")
    group.add_argument("--asap-replica-tp", type=int, default=None,
                       help="T, the TP width of one A replica. Required on F; on A it defaults to --tp-size.")
    group.add_argument("--asap-ffn-ranks", type=int, default=None,
                       help="E, the number of F ranks. On A it turns AFD on (no local experts); "
                            "on F it defaults to --ep-size.")


# ---- deployment rules, against sglang's resolved ServerArgs (``resolving_view``) ----
def _reader(view, problems):
    """getattr that reports a field sglang no longer has (version drift) instead of crashing."""

    def get(name):
        try:
            return getattr(view, name)
        except AttributeError:
            problems.append(f"sglang no longer exposes server arg {name!r} (version drift?)")
            return None

    return get


def _common(get, problems) -> None:
    """Rules of both roles: one node, no PP, no a2a backend, no EPLB."""
    if get("pp_size") not in (None, 1):
        problems.append(f"pipeline parallelism is not supported (--pp-size 1, got {get('pp_size')})")
    if get("nnodes") not in (None, 1):
        problems.append(f"the transport is CUDA IPC, so A and F share one node (--nnodes 1, got {get('nnodes')})")
    if get("moe_a2a_backend") not in (None, "none"):
        problems.append(f"--moe-a2a-backend must be none (experts are served by the F service), "
                        f"got {get('moe_a2a_backend')!r}")
    if get("enable_eplb"):
        problems.append("--enable-eplb is not supported")


def check_attn(view, cfg: AsapConfig) -> None:
    """Rules of the attention role. Raises AsapConfigError listing every problem at once."""
    problems = []
    get = _reader(view, problems)
    _common(get, problems)
    if get("attn_dp_size") not in (None, 1) or get("enable_dp_attention"):
        problems.append("attention data parallelism couples the replicas through per-step collectives; "
                        "use plain replicas instead: --dp-size D")
    if get("disaggregation_mode") == "decode":
        problems.append("ASAP serves the prefill phase only; --disaggregation-mode decode is not allowed")
    dp, tp = get("dp_size"), get("tp_size")
    if cfg.num_replicas is not None and dp is not None and cfg.num_replicas != dp:
        problems.append(f"--asap-num-replicas {cfg.num_replicas} must equal --dp-size {dp}")
    if cfg.replica_tp is not None and tp is not None and cfg.replica_tp != tp:
        problems.append(f"--asap-replica-tp {cfg.replica_tp} must equal --tp-size {tp}")
    if cfg.afd:
        if get("ep_size") not in (None, 1):
            problems.append(f"--ep-size must be 1 on the attention role (experts live on F), got {get('ep_size')}")
        if tp not in (None, 1):
            # T > 1 would send each TP rank's row slice and fold the routed output into the shared expert's
            # all-reduce; the transport already groups T rows into one F job, the A side does not do it yet.
            problems.append(f"AFD replicas support --tp-size 1 only for now, got {tp}")
        if get("speculative_algorithm") is not None:
            problems.append("speculative decoding is not supported (the A role drops the MTP weights)")
        if get("expert_distribution_recorder_mode") is not None:
            problems.append("--expert-distribution-recorder-mode needs local experts")
        # one MoE layer of one forward goes to F as one message of at most n_max tokens
        for name in ("chunked_prefill_size", "max_prefill_tokens"):
            value = get(name)
            if value is None or value <= 0 or value > cfg.n_max:
                problems.append(f"--{name.replace('_', '-')} ({value}) must be in [1, --asap-n-max {cfg.n_max}]")
    if problems:
        raise AsapConfigError("[ASAP] invalid server args for the attn role:\n  - " + "\n  - ".join(problems))


def check_ffn(view, cfg: AsapConfig) -> None:
    """Rules of the expert service."""
    problems = []
    get = _reader(view, problems)
    _common(get, problems)
    tp, ep = get("tp_size"), get("ep_size")
    if tp != ep:
        problems.append(f"the F service is pure expert parallelism: --ep-size must equal --tp-size (E), "
                        f"got tp={tp} ep={ep}")
    if cfg.ffn_ranks is not None and cfg.ffn_ranks != ep:
        problems.append(f"--asap-ffn-ranks {cfg.ffn_ranks} must equal --ep-size {ep}")
    if get("dp_size") not in (None, 1) or get("attn_dp_size") not in (None, 1) or get("enable_dp_attention"):
        problems.append("the F service has no data-parallel dimension (--dp-size 1, no attention DP)")
    if get("disaggregation_mode") not in (None, "null"):
        problems.append(f"--disaggregation-mode must be null on the F service, got {get('disaggregation_mode')!r}")
    if problems:
        raise AsapConfigError("[ASAP] invalid server args for the ffn role:\n  - " + "\n  - ".join(problems))


# ---- the model, read without torch ----
@dataclass(frozen=True)
class MoeShape:
    """The MoE dimensions both roles size their buffers with."""
    n_experts: int
    hidden: int
    topk: int
    layers: int


def read_moe_shape(model_path: str, override_json: Optional[str] = "{}") -> MoeShape:
    """MoE dimensions from config.json plus --json-model-override-args."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        cfg.update(json.loads(override_json or "{}"))
        cfg = cfg.get("text_config", cfg)
        n_experts = next(cfg[k] for k in ("n_routed_experts", "num_experts", "num_local_experts") if k in cfg)
        return MoeShape(n_experts, cfg["hidden_size"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"])
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise AsapConfigError(f"[ASAP] cannot read the MoE shape from {model_path}/config.json: {exc!r}") from exc


def experts_per_rank(n_experts: int, ffn_ranks: int) -> int:
    """P = ceil(n_experts / E): F rank f owns [f * P, (f + 1) * P); ids >= n_experts are padding."""
    if not 1 <= ffn_ranks <= n_experts:
        raise AsapConfigError(f"[ASAP] {ffn_ranks} F ranks for {n_experts} experts")
    return -(-n_experts // ffn_ranks)
