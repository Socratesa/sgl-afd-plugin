"""Resolution and validation of the ``asap`` arguments (cf. ``pd_disaggregation_hook.py``).

``handle_asap`` runs inside ServerArgs resolution and declares what a role needs;
``check_asap_server_args`` runs before SGLang's own ``check_server_args`` on the
resolved arguments, so that a rule of the deployment is reported as such;
``check_two_batch_overlap`` replaces SGLang's check of the same name on an attention
rank whose experts run on the FFN ranks.
"""

from __future__ import annotations

import logging
import re

import msgspec
from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

from sgl_asap_plugin.srt.arg_groups.fields.asap import Asap
from sgl_asap_plugin.srt.server_args import (
    export_env,
    get_asap_args,
    is_afd_attn,
    set_asap_args,
)
from sgl_asap_plugin.transport.layout import MAX_TOKENS

logger = logging.getLogger(__name__)

MAX_MICRO_BATCHES = 8
_ADDR_RE = re.compile(r"^[A-Za-z0-9_.\-]+:(\d{1,5})$")


def handle_asap(server_args, previous) -> None:
    """Wraps the ``handle_cuda_graph_config`` resolution step, which reads what is
    declared here."""
    args = get_asap_args()
    if args.asap_role != "null":
        # The shared experts stay with the gate on the attention ranks.
        declare_resolution(
            server_args, "handle_asap", disable_shared_experts_fusion=True
        )
    if is_afd_attn(args):
        # The waits on the FFN ranks are enqueued with fixed sequence numbers,
        # which a graph replay would reuse.
        declare_resolution(server_args, "handle_asap", disable_cuda_graph=True)
    previous(server_args)


def check_asap_server_args(server_args) -> None:
    """BEFORE ``check_server_args``. Fills the values that follow from the resolved
    arguments and exports them to the processes the launcher starts."""
    from sgl_asap_plugin import hooks

    cfg = resolving_view(server_args)
    args = get_asap_args()
    check_asap_deployment(cfg, args)
    if args.asap_role == "attn":
        args = msgspec.structs.replace(
            args,
            asap_attn_dp_size=args.asap_attn_dp_size or cfg.dp_size,
            asap_attn_tp_size=cfg.tp_size,
        )
    else:
        args = msgspec.structs.replace(args, asap_ffn_size=cfg.ep_size)
    set_asap_args(args)
    export_env(args)
    logger.info(
        f"[ASAP] server args ok: role={args.asap_role} afd={is_afd_attn(args)} "
        f"hooks={hooks.hooks_applied()}"
    )


def check_two_batch_overlap(server_args) -> None:
    """REPLACE of ``validation_hook.check_two_batch_overlap`` on an attention rank
    whose routed experts run on the FFN ranks. Without an EP all-to-all, SGLang's
    two-batch overlap hides the attention-DP gather and scatter and needs attention
    DP. An ASAP replica hides the FFN ranks' work instead
    (``srt/batch_overlap/operations_strategy.py``) and needs none."""


def check_asap_deployment(cfg, args: Asap) -> None:
    """The rules of an ASAP deployment, against the resolved server arguments."""
    _check_values(args)
    if cfg.pp_size != 1:
        raise ValueError("ASAP does not support pipeline parallelism (--pp-size 1)")
    if cfg.nnodes != 1:
        raise ValueError(
            "Each ASAP role runs on one node (--nnodes 1); the attention and FFN roles "
            "can be on different GB200 trays with --asap-transport-mem fabric"
        )
    if cfg.moe_a2a_backend != "none":
        raise ValueError("ASAP replaces the MoE all-to-all: use --moe-a2a-backend none")
    if cfg.enable_eplb or cfg.ep_num_redundant_experts:
        raise ValueError(
            "ASAP does not support --enable-eplb or --ep-num-redundant-experts"
        )
    if cfg.speculative_algorithm is not None:
        raise ValueError("ASAP does not support speculative decoding")
    if cfg.startup_weight_load_mode == "overlap":
        raise ValueError("ASAP does not support --startup-weight-load-mode overlap")
    if args.asap_role == "attn":
        _check_attn_role(cfg, args)
    else:
        _check_ffn_role(cfg, args)


def _check_values(args: Asap) -> None:
    match = _ADDR_RE.match(args.asap_bootstrap_addr)
    if not match or not 1 <= int(match.group(1)) <= 65535:
        raise ValueError(
            f"--asap-bootstrap-addr must be HOST:PORT, got {args.asap_bootstrap_addr!r}"
        )
    if not 1 <= args.asap_max_dispatch_tokens <= MAX_TOKENS:
        raise ValueError(
            f"--asap-max-dispatch-tokens must be in [1, {MAX_TOKENS}], "
            f"got {args.asap_max_dispatch_tokens}"
        )
    if not 1 <= args.asap_max_micro_batches <= MAX_MICRO_BATCHES:
        raise ValueError(
            f"--asap-max-micro-batches must be in [1, {MAX_MICRO_BATCHES}], "
            f"got {args.asap_max_micro_batches}"
        )
    if not 0 <= args.asap_ffn_cuda_graph_max_tokens <= args.asap_max_dispatch_tokens:
        raise ValueError(
            "--asap-ffn-cuda-graph-max-tokens must be in "
            f"[0, --asap-max-dispatch-tokens {args.asap_max_dispatch_tokens}], "
            f"got {args.asap_ffn_cuda_graph_max_tokens}"
        )
    if args.asap_min_batch_tokens < 0 or args.asap_max_wait_ms < 0:
        raise ValueError(
            "--asap-min-batch-tokens and --asap-max-wait-ms must be >= 0, got "
            f"{args.asap_min_batch_tokens} and {args.asap_max_wait_ms}"
        )
    if args.asap_tbo_min_tokens < 0:
        raise ValueError(
            f"--asap-tbo-min-tokens must be >= 0, got {args.asap_tbo_min_tokens}"
        )
    for name in ("asap_ffn_size", "asap_attn_dp_size", "asap_attn_tp_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1, got {value}")
    if args.asap_role == "ffn" and None in (
        args.asap_attn_dp_size,
        args.asap_attn_tp_size,
    ):
        raise ValueError(
            "--asap-role ffn needs --asap-attn-dp-size and --asap-attn-tp-size "
            "(they size its receive buffers)"
        )


def _check_attn_role(cfg, args: Asap) -> None:
    if cfg.enable_dp_attention or cfg.attn_dp_size != 1:
        raise ValueError(
            "ASAP's attention replicas are plain data parallelism: use --dp-size "
            "without --enable-dp-attention"
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError("ASAP serves the prefill phase: --disaggregation-mode decode")
    total, base = args.asap_attn_dp_size or cfg.dp_size, args.asap_attn_base_replica
    if not 0 <= base <= total - cfg.dp_size:
        raise ValueError(
            f"The replicas of this instance, [{base}, {base + cfg.dp_size}) "
            f"(--asap-attn-base-replica {base}, --dp-size {cfg.dp_size}), must lie "
            f"within the --asap-attn-dp-size {total} replicas of the deployment"
        )
    if args.asap_attn_tp_size not in (None, cfg.tp_size):
        raise ValueError(
            f"--asap-attn-tp-size {args.asap_attn_tp_size} != --tp-size {cfg.tp_size}"
        )
    if not is_afd_attn(args):
        return
    if cfg.ep_size != 1:
        raise ValueError("The routed experts run on the FFN ranks: use --ep-size 1")
    if cfg.tp_size != 1:
        raise ValueError("ASAP attention replicas support --tp-size 1 only")
    if cfg.expert_distribution_recorder_mode is not None:
        raise ValueError("--expert-distribution-recorder-mode needs local experts")
    if cfg.enable_two_batch_overlap and args.asap_max_micro_batches < 2:
        raise ValueError(
            "--enable-two-batch-overlap keeps two micro-batches in flight: "
            "--asap-max-micro-batches must be >= 2"
        )
    for name in ("chunked_prefill_size", "max_prefill_tokens"):
        value = getattr(cfg, name)
        if value is None or not 0 < value <= args.asap_max_dispatch_tokens:
            raise ValueError(
                f"--{name.replace('_', '-')} ({value}) must be in "
                f"[1, --asap-max-dispatch-tokens {args.asap_max_dispatch_tokens}]: "
                "one dispatch carries one MoE layer of one forward"
            )


def _check_ffn_role(cfg, args: Asap) -> None:
    if cfg.ep_size != cfg.tp_size:
        raise ValueError(
            "The FFN ranks are pure expert parallelism: --ep-size must equal "
            f"--tp-size, got {cfg.ep_size} and {cfg.tp_size}"
        )
    if args.asap_ffn_size not in (None, cfg.ep_size):
        raise ValueError(
            f"--asap-ffn-size {args.asap_ffn_size} != --ep-size {cfg.ep_size}"
        )
    if cfg.dp_size != 1 or cfg.attn_dp_size != 1 or cfg.enable_dp_attention:
        raise ValueError("The FFN role has no data parallelism: --dp-size 1")
    if cfg.disaggregation_mode != "null":
        raise ValueError("The FFN role needs --disaggregation-mode null")
