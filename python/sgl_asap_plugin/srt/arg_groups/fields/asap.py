"""Namespace ``asap``: the ``--asap-*`` server arguments.

ServerArgs has a fixed set of fields, so the plugin declares its own namespace the
way SGLang declares one (``sglang.srt.arg_groups.fields``), and ``server_args.py``
adds it to the parser next to ServerArgs' flags.
"""

from typing import Literal, Optional

import msgspec
from sglang.srt.arg_groups.arg_utils import A


class Asap(msgspec.Struct):
    """Namespace ``asap``."""

    asap_role: A[
        Literal["null", "attn", "ffn"],
        "Role in an ASAP deployment (prefill attention-FFN disaggregation). 'attn': "
        "plain data-parallel replicas (--dp-size); with --asap-ffn-size their routed "
        "experts run on the FFN ranks. 'ffn': the FFN ranks (--tp-size E --ep-size E).",
    ] = "null"
    asap_ffn_size: A[
        Optional[int],
        "Number of FFN ranks. On the attention role it moves the routed experts to the "
        "FFN ranks; without it the experts stay local. Defaults to --ep-size on the "
        "FFN role.",
    ] = None
    asap_bootstrap_addr: A[
        str,
        "HOST:PORT where FFN rank 0 accepts the attention ranks; FFN rank f listens on "
        "PORT + f.",
    ] = "127.0.0.1:9600"
    asap_max_dispatch_tokens: A[
        int,
        "Max tokens of one dispatch (one MoE layer of one forward); sizes the transfer "
        "buffers. On the attention role --chunked-prefill-size and "
        "--max-prefill-tokens must not exceed it.",
    ] = 16384
    asap_max_micro_batches: A[
        int,
        "Micro-batches in flight per attention rank, i.e. receive slots per attention "
        "rank on every FFN rank.",
    ] = 2
    asap_min_batch_tokens: A[
        int,
        "Attention role: hold a prefill batch until the waiting requests have at least "
        "this many tokens to compute (prompt less the cached prefix) or the oldest has "
        "waited --asap-max-wait-ms "
        "(length-aware batching; 0: off). Bigger batches make bigger FFN jobs.",
    ] = 0
    asap_max_wait_ms: A[
        float,
        "Attention role: longest wait of a request for --asap-min-batch-tokens.",
    ] = 50.0
    asap_tbo_min_tokens: A[
        int,
        "Attention role with --enable-two-batch-overlap: split only prefill batches of "
        "at least this many tokens. A small batch is bound by the CPU launching its "
        "kernels, which a split doubles.",
    ] = 6144
    asap_ffn_cuda_graph_max_tokens: A[
        int,
        "FFN role: capture every layer's expert call in CUDA graphs for jobs of up to "
        "this many tokens (power-of-two buckets; 0: eager). It removes the per-job "
        "launch cost, which dominates small jobs.",
    ] = 0
    asap_attn_dp_size: A[
        Optional[int],
        "--dp-size of the attention role. Required on the FFN role.",
    ] = None
    asap_attn_tp_size: A[
        Optional[int],
        "--tp-size of the attention role. Required on the FFN role.",
    ] = None
