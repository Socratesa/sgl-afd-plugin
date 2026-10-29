"""The attention (A) role in AFD mode: every DeepSeek MoE layer keeps its gate, top-k and shared expert, and its
routed experts run on the F service. Hooks (hooks.py) make four changes to a stock sglang model:

1. ``get_moe_impl_class`` returns RemoteExperts, a FusedMoE stand-in without parameters;
2. the checkpoint iterators skip routed-expert tensors unread (checkpoint.py);
3. ``DeepseekV2MoE.forward`` runs ``moe_forward`` below for layers holding RemoteExperts;
4. after ``ModelRunner.load_model`` (weights loaded, KV pool not yet sized), ``on_model_loaded`` connects this
   rank to every F rank, so the result arena is allocated before sglang hands free memory to the KV pool.

Contract with F: A sends the hidden states, the logical top-k ids and the FINAL routing weights (top-k already
multiplied by routed_scaling_factor); F rank f returns sum over k with ids[k] in f of w[k] * expert_ids[k](x),
and A adds the E partial sums. F needs no knowledge of the scale and A needs none of F's quantization.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from sgl_asap_plugin.config import AsapConfig, AsapConfigError

logger = logging.getLogger("sgl_asap_plugin")

ENV_CONNECT_TIMEOUT = "ASAP_CONNECT_TIMEOUT_S"  # how long A waits for F to come up (default 1800 s: F loads ~40 GB)
ENV_WAIT_TIMEOUT = "ASAP_WAIT_TIMEOUT_S"  # one wait for F longer than this poisons the channel (default 60 s)
SLOT = 0  # the synchronous forward has one message in flight; dual-batch will use slot = micro-batch index


class RemoteExperts(nn.Module):
    """Stand-in for sglang's FusedMoE on an A rank: no weights, the experts are on F.

    DeepseekV2MoE.__init__ builds it through get_moe_impl_class(...)(...) and reads only
    ``should_fuse_routed_scaling_factor_in_topk``, which configures its TopK. It is True here, so TopK folds
    routed_scaling_factor into the weights (stock DSV4 on Hopper does the same: the MXFP4 FlashInfer method
    declares fuse_routed_scaling_factor_in_topk = True). ``quant_method = None`` sends a MoEOutput consumer
    (maybe_fuse_routed_scale_and_shared_add) to its plain ``routed += shared`` branch, right for final weights.
    """

    should_fuse_routed_scaling_factor_in_topk = True
    quant_method = None

    def __init__(self, num_experts, num_fused_shared_experts, top_k, hidden_size, intermediate_size, layer_id,
                 **_unused):
        super().__init__()
        if num_fused_shared_experts:
            raise AsapConfigError("[ASAP] fused shared experts would move the shared expert to F; keep "
                                  "shared-experts fusion off (the DSV4 default)")
        self.num_experts, self.top_k, self.hidden_size, self.layer_id = num_experts, top_k, hidden_size, layer_id
        self.num_local_experts = 0
        self.moe_runner_config = SimpleNamespace(inplace=False)  # read by stock code paths that never run here
        self.client: Optional["AttnClient"] = None  # set by on_model_loaded

    def forward(self, *_args, **_kwargs):
        raise RuntimeError("[ASAP] RemoteExperts has no weights: DeepseekV2MoE.forward must be the AFD one "
                           "(hook on sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward)")


class AttnClient:
    """This rank's AfdChannel, plus tensor views of the result segments it returns."""

    def __init__(self, channel, device: int):
        """Wrap a connected AfdChannel; ``device`` is this rank's GPU ordinal."""
        self.channel, self.device, self.g = channel, device, channel.g
        self._segments = {}  # result address -> [n_max, hidden] bf16 view

    def send(self, layer_id: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor):
        """Enqueue one layer's message on the current stream; x / ids / w stay alive by stream order."""
        if x.dtype != torch.bfloat16:
            raise TypeError(f"[ASAP] the transport carries bf16 activations, got {x.dtype}")
        x, ids, w = x.contiguous(), ids.to(torch.int32).contiguous(), w.to(torch.float32).contiguous()
        return self.channel.send(layer_id, SLOT, x.data_ptr(), ids.data_ptr(), w.data_ptr(), x.shape[0],
                                 torch.cuda.current_stream().cuda_stream)

    def recv(self, ticket) -> torch.Tensor:
        """Enqueue the waits for every F rank and return the sum of their partial outputs as a new tensor (the
        segments are overwritten by the next reply, so nothing may keep a view of them)."""
        ptrs = self.channel.wait(ticket, torch.cuda.current_stream().cuda_stream)
        parts = [self._segment(p)[:ticket.n_tokens] for p in ptrs]
        if len(parts) == 1:
            return parts[0].clone()
        acc = parts[0].float()
        for p in parts[1:]:
            acc += p
        return acc.to(torch.bfloat16)  # one rounding for the E-way sum

    def exchange(self, layer_id: int, x, ids, w) -> torch.Tensor:
        """send + recv in pieces of at most n_max rows, for the rare forward that is larger than one message."""
        m = self.g.n_max
        return torch.cat([self.recv(self.send(layer_id, x[i:i + m], ids[i:i + m], w[i:i + m]))
                          for i in range(0, x.shape[0], m)])

    def _segment(self, ptr: int) -> torch.Tensor:
        """[n_max, hidden] bf16 view of the result segment at ``ptr``, built once."""
        view = self._segments.get(ptr)
        if view is None:
            from sgl_asap_plugin.transport.cuda import DeviceBuffer

            g = self.g
            raw = torch.as_tensor(DeviceBuffer(ptr, g.n_max * g.hidden * 2), device=f"cuda:{self.device}")
            view = self._segments[ptr] = raw.view(torch.bfloat16).view(g.n_max, g.hidden)
        return view


def moe_forward(original, moe, hidden_states, forward_batch=None, gemm_output_zero_allocator=None, input_ids=None,
                input_ids_global=None, skip_shared_experts=False, return_moe_output=False):
    """AROUND hook of DeepseekV2MoE.forward. Layers with local experts run the stock code.

    For remote experts it is stock forward_normal (deepseek_v2.py) with the routed experts replaced by a message
    to F, reordered so the shared expert runs on A's GPU while F computes:
        gate, top-k        stock modules; hash layers route by input_ids_global
        send               x, ids, final weights to every F rank (enqueued; the CPU moves on)
        shared expert      overlaps F's work
        recv               stream waits for every F rank, then the sum of their partial outputs
        merge              stock tail: MoEOutput.get_merged, post_experts_all_reduce, replicated shared add
    Everything is enqueued on the current stream; the CPU never waits for F.
    """
    from sglang.srt.layers.moe.utils import post_experts_all_reduce
    from sglang.srt.models.deepseek_v2 import MoEOutput

    experts = moe.experts
    if not isinstance(experts, RemoteExperts):
        return original(moe, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids, input_ids_global,
                        skip_shared_experts, return_moe_output)
    n = hidden_states.shape[0]
    shared = None
    if n == 0:  # this rank has no tokens in this forward: nothing is sent, F never waits for a rank
        routed = hidden_states.new_empty((0, experts.hidden_size))
    else:
        router_logits = moe.gate(hidden_states, gemm_output_zero_allocator)
        topk = moe.topk(hidden_states, router_logits,
                        num_token_non_padded=forward_batch.moe_num_token_non_padded() if forward_batch else None,
                        expert_location_dispatch_info=None,
                        **({"input_ids": input_ids_global} if getattr(moe, "is_hash", False) else {}))
        if not hasattr(topk, "topk_ids"):
            raise RuntimeError(f"[ASAP] layer {moe.layer_id}: top-k produced {type(topk).__name__}, but F needs "
                               "explicit ids and weights (a STANDARD top-k output)")
        client = experts.client
        if n <= client.g.n_max:
            ticket = client.send(moe.layer_id, hidden_states, topk.topk_ids, topk.topk_weights)
            if not skip_shared_experts:
                shared = moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
            routed = client.recv(ticket)
        else:  # only a mixed prefill+decode batch: chunked-prefill-size and max-prefill-tokens <= n_max at launch
            if not skip_shared_experts:
                shared = moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
            routed = client.exchange(moe.layer_id, hidden_states, topk.topk_ids, topk.topk_weights)
    out = MoEOutput(routed=routed, shared=shared, experts=experts, routed_scaling_factor=moe.routed_scaling_factor,
                    shared_is_replicated=moe._shared_expert_tp1)
    if return_moe_output:
        return out
    merged = out.get_merged()
    if moe.reduce_results:
        merged = post_experts_all_reduce(merged)  # a no-op for --tp-size 1
    if shared is not None and moe._shared_expert_tp1 and moe._adds_replicated_output():
        merged += shared
    return merged


def remote_experts_of(model) -> List[RemoteExperts]:
    """Every RemoteExperts of the model, in layer order."""
    return sorted((m for m in model.modules() if isinstance(m, RemoteExperts)), key=lambda m: m.layer_id)


def model_fingerprint(stand_ins: List[RemoteExperts]) -> dict:
    """What F must serve for this model; F compares it field by field in the handshake."""
    first = stand_ins[0]
    return {"hidden": first.hidden_size, "topk": first.top_k, "n_experts": first.num_experts,
            "moe_layers": [m.layer_id for m in stand_ins]}


def on_model_loaded(model_runner, hooks_applied: str) -> None:
    """AFTER hook body of ModelRunner.load_model: log this rank's identity; in AFD mode connect it to F."""
    from sglang.srt.runtime_context import get_parallel, get_server_args

    from sgl_asap_plugin.transport.channel import AfdChannel

    if getattr(model_runner, "draft_model_idx", None) is not None:
        return  # a speculative draft model (rejected at launch in AFD mode)
    import torch.distributed as dist

    cfg, parallel, args = AsapConfig.from_env(), get_parallel(), get_server_args()
    replica, tp_rank, gpu = parallel.dp_rank or 0, parallel.tp_rank, model_runner.gpu_id
    world = dist.get_world_size() if dist.is_initialized() else None  # plain DP: the replica's TP size, not D*T
    logger.info("[ASAP] attn rank ready: replica=%d tp_rank=%d world_size=%s gpu=%d pid=%d hooks=%s afd=%s", replica,
                tp_rank, world, gpu, os.getpid(), hooks_applied, cfg.afd)
    if not cfg.afd:
        logger.warning("[ASAP] --asap-ffn-ranks is not set: experts stay local, this is plain data parallelism")
        return
    stand_ins = remote_experts_of(model_runner.model)
    if not stand_ins:
        raise AsapConfigError("[ASAP] AFD mode, but the model has no MoE layer built through get_moe_impl_class")
    cfg = cfg.with_defaults(num_replicas=args.dp_size, replica_tp=args.tp_size)
    fingerprint = model_fingerprint(stand_ins)
    channel = AfdChannel(replica * cfg.replica_tp + tp_rank, cfg.geometry(fingerprint["hidden"], fingerprint["topk"]),
                         cfg.endpoints(), device=gpu, hello={"model": fingerprint},
                         timeout_s=float(os.environ.get(ENV_WAIT_TIMEOUT, "60")),
                         connect_timeout_s=float(os.environ.get(ENV_CONNECT_TIMEOUT, "1800"))).connect()
    client = AttnClient(channel, gpu)
    for m in stand_ins:
        m.client = client
    served = [{k: r.get(k) for k in ("f", "experts", "quant_method")} for r in channel.replies]
    logger.info("[ASAP] A rank %d connected to %d F rank(s): %s; geometry %s", channel.a, len(served), served,
                channel.g)
