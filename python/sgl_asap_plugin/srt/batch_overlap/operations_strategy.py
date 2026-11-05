"""Two-batch overlap on an attention rank whose routed experts run on the FFN ranks.

``_compute_moe_deepseek_v4_layer_operations_strategy_tbo`` (AROUND) gives every
DeepSeek-V4 layer of such a rank one stage boundary, right after a micro-batch has
sent its MoE input to the FFN ranks. SGLang's TBO engine issues the stages of the
two micro-batches alternately, so the attention rank runs micro-batch b's attention
of a layer while the FFN ranks work on a's experts of it, then waits for a's answer
and starts a's next layer while the FFN ranks work on b (ASAP, Fig. 8):

    a: attn L, send L | b: attn L, send L | a: wait L, attn L+1, send L+1 | b: wait L ...

The two halves of the transport take different streams. A send (the copies into the
FFN ranks' receive buffers, then the doorbells) runs on a stream of its own: on the
compute stream its copies would hold up the other micro-batch's attention. A wait
stays on the compute stream, which reaches it after the other micro-batch's
attention. On the send stream it would queue behind the other micro-batch's next
send, which cannot start before the attention that produces its input, so an answer
that has arrived would wait for those copies. Micro-batch i uses transport slot i.
"""

from __future__ import annotations


def _compute_moe_deepseek_v4_layer_operations_strategy_tbo(
    original, layer, forward_mode
):
    """AROUND ``operations_strategy._compute_moe_deepseek_v4_layer_operations_strategy_tbo``:
    the prefill operations of a layer whose experts are ``AsapMoE``."""
    from sglang.srt.batch_overlap import operations
    from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from sgl_asap_plugin.srt.layers.moe.token_dispatcher.asap import AsapDispatcher

    experts = getattr(layer.mlp, "experts", None)
    if forward_mode != ForwardMode.EXTEND or not isinstance(
        getattr(experts, "dispatcher", None), AsapDispatcher
    ):
        return original(layer, forward_mode)
    moe = AsapMoeOperations(layer.mlp)
    return OperationsStrategy(
        operations=[
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn,
            layer.op_mhc_post_attn_pre_mlp,
            moe.op_asap_dispatch,
            operations.YieldOperation(),
            moe.op_asap_combine,
            layer.op_mhc_postprocess,
        ],
        tbo_delta_stages=0,
    )


class AsapMoeOperations:
    """The MoE block (``DeepseekV2MoE``) of one layer as two TBO operations, with the
    arithmetic of ``DeepseekV2MoE.forward_normal``."""

    def __init__(self, mlp):
        self.mlp = mlp

    def op_asap_dispatch(self, state):
        """Router and top-k, the send to the FFN ranks, then the shared experts, which
        run while the FFN ranks compute."""
        mlp, forward_batch = self.mlp, state.forward_batch
        hidden_states = state.pop("hidden_states_mlp_input")
        if hidden_states.shape[0] > 0:
            router_logits = mlp.gate(hidden_states)
            # Hash layers route on the token ids of the micro-batch.
            kwargs = (
                {"input_ids": forward_batch.input_ids}
                if getattr(mlp, "is_hash", False)
                else {}
            )
            topk_output = mlp.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.moe_num_token_non_padded(),
                expert_location_dispatch_info=None,
                **kwargs,
            )
        else:
            topk_output = mlp.topk.empty_topk_output(
                hidden_states.device, layer_id=mlp.layer_id
            )
        state.asap_dispatch = mlp.experts.dispatcher.dispatch(
            hidden_states, topk_output, micro_batch=state.tbo_subbatch_index
        )
        state.asap_shared = mlp._forward_shared_experts(hidden_states)

    def op_asap_combine(self, state):
        """Wait for the FFN ranks' answer and add the shared experts."""
        from sglang.srt.models.deepseek_v2 import (
            maybe_fuse_routed_scale_and_shared_add,
            post_experts_all_reduce,
        )

        mlp = self.mlp
        routed = mlp.experts.dispatcher.combine(state.pop("asap_dispatch"))
        shared = state.pop("asap_shared")
        output = maybe_fuse_routed_scale_and_shared_add(
            mlp.experts,
            routed,
            None if mlp._shared_expert_tp1 else shared,
            mlp.routed_scaling_factor,
        )
        if mlp.reduce_results:
            output = post_experts_all_reduce(output)
        if (
            shared is not None
            and mlp._shared_expert_tp1
            and mlp._adds_replicated_output()
        ):
            output += shared
        state.hidden_states_mlp_output = output
