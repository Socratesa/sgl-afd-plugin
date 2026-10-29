"""AFD replacement of ``DeepseekV2MoE.forward`` on an A rank.

Same steps as sglang's ``forward_normal``, with the routed experts moved to the F service:

    gate + top-k (logical expert ids, final weights)   -> stays here, same modules as stock
    send x, ids, w to every F rank                      -> async, on the current stream
    shared expert                                       -> overlaps with the F ranks' work
    wait for the F ranks, sum their partial outputs     -> stream wait, no host blocking
    merge routed + shared exactly like the stock fused-MoE branch

Everything is enqueued on the current stream; the host never waits for F.
"""

from __future__ import annotations

from sgl_asap_plugin.model.remote_experts import AfdRemoteExperts


def moe_forward(original, self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None, input_ids=None,
                input_ids_global=None, skip_shared_experts=False, return_moe_output=False):
    """AROUND hook of DeepseekV2MoE.forward: layers without the stand-in experts keep the stock path."""
    if not isinstance(self.experts, AfdRemoteExperts):
        return original(self, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids, input_ids_global,
                        skip_shared_experts, return_moe_output)
    return remote_moe_forward(self, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids_global,
                              skip_shared_experts)


def valid_rows(forward_batch, n: int) -> int:
    """Rows that carry tokens. Padding rows (CUDA graph buckets) are not sent; stock marks them separately."""
    if forward_batch is None:
        return n
    non_padded = forward_batch.moe_num_token_non_padded()
    return n if non_padded is None else min(n, int(non_padded))


def merge_routed_and_shared(moe, routed, shared, tp_size=1):
    """Stock ``maybe_fuse_routed_scale_and_shared_add`` for the fused MXFP4/Marlin/ExpertPack branch, then the
    stock tail of forward_normal (post-experts reduce, replicated shared expert).

    With one rank per replica the post-experts reduce is a no-op and is skipped; a wider replica would have to
    reduce the shared expert before adding the (already complete) routed output, see remote_moe_forward."""
    already_scaled = moe.experts.should_fuse_routed_scaling_factor_in_topk
    tp1_shared = moe._shared_expert_tp1
    sharded = None if tp1_shared else shared
    if sharded is not None:
        merged = sharded.add_(routed, alpha=1.0 if already_scaled else moe.routed_scaling_factor)
    elif already_scaled:
        merged = routed
    else:
        merged = routed.mul_(moe.routed_scaling_factor)
    if moe.reduce_results and tp_size > 1:
        from sglang.srt.layers.moe.utils import post_experts_all_reduce

        merged = post_experts_all_reduce(merged)
    if shared is not None and tp1_shared and moe._adds_replicated_output():
        merged += shared
    return merged


def remote_moe_forward(moe, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids_global,
                       skip_shared_experts):
    import torch

    from sgl_asap_plugin.model.client import get_client

    n = hidden_states.shape[0]
    if n == 0:
        return torch.empty_like(hidden_states)  # a rank with no tokens sends nothing (F never waits for it)
    client = get_client()
    if client.geometry.group != 1:
        raise NotImplementedError("[ASAP] replicas with --tp-size > 1 are not implemented yet; use --tp-size 1")
    rows = valid_rows(forward_batch, n)
    num_token_non_padded = forward_batch.moe_num_token_non_padded() if forward_batch is not None else None

    router_logits = moe.gate(hidden_states, gemm_output_zero_allocator)
    topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
    topk = moe.topk(hidden_states, router_logits, num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=None, **topk_kwargs)
    x = hidden_states if rows == n else hidden_states[:rows]
    ids = topk.topk_ids[:rows].to(torch.int32).contiguous()
    w = topk.topk_weights[:rows].to(torch.float32).contiguous()
    ticket = client.send(moe.layer_id, x.contiguous(), ids, w, rows)

    shared = None if skip_shared_experts else moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator)

    routed = client.combine(ticket, rows)
    if rows != n:  # padding rows get a zero routed output, like a token routed nowhere
        routed = torch.cat([routed, routed.new_zeros((n - rows, routed.shape[1]))])
    return merge_routed_and_shared(moe, routed, shared, tp_size=client.geometry.group)
