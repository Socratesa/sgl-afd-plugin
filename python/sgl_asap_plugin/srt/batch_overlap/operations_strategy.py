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

ERNIE 5.5 has no TBO operations in SGLang: ``OperationsStrategy.init_new_tbo`` (AROUND)
adds them, with the same stage boundary. Its layers, MoE and dense, are one operation
chain per micro-batch (``Ernie55LayerOperations``); the dense layers have no boundary
and run inside the stage of the layer next to them.
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


def init_new_tbo(original, layers, forward_mode):
    """AROUND ``OperationsStrategy.init_new_tbo``, which names the layer classes SGLang
    has TBO operations for and raises NotImplementedError for any other: the operations
    of an ERNIE 5.5 prefill."""
    from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
    from sglang.srt.models.ernie55 import Ernie5_5DecoderLayer

    if not isinstance(layers[0], Ernie5_5DecoderLayer):
        return original(layers, forward_mode)
    return OperationsStrategy.concat(
        [
            _compute_ernie55_layer_operations_strategy_tbo(layer, forward_mode)
            for layer in layers
        ]
    )


def _compute_ernie55_layer_operations_strategy_tbo(layer, forward_mode):
    from sglang.srt.batch_overlap import operations
    from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.ernie55 import Ernie5_5Moe

    from sgl_asap_plugin.srt.layers.moe.token_dispatcher.asap import AsapDispatcher

    if forward_mode != ForwardMode.EXTEND:
        raise NotImplementedError(
            f"ERNIE 5.5 TBO only supports prefill (EXTEND), got {forward_mode=}"
        )
    ops = Ernie55LayerOperations(layer)
    attention = [ops.op_prepare_attn, ops.op_attn, ops.op_post_attn_pre_mlp]
    if not isinstance(layer.mlp, Ernie5_5Moe):
        ffn = [ops.op_dense_mlp]
    elif isinstance(layer.mlp.experts.dispatcher, AsapDispatcher):
        ffn = [ops.op_asap_dispatch, operations.YieldOperation(), ops.op_asap_combine]
    else:
        raise NotImplementedError(
            "ERNIE 5.5 TBO needs the routed experts on the FFN ranks (--asap-ffn-size)"
        )
    return OperationsStrategy(
        operations=[*attention, *ffn, ops.op_postprocess], tbo_delta_stages=0
    )


class Ernie55LayerOperations:
    """One ``Ernie5_5DecoderLayer`` as TBO operations, with the arithmetic of
    ``Ernie5_5DecoderLayer.forward`` and, for a MoE layer, of ``Ernie5_5Moe.forward``
    split at the routed experts. Attention is a single operation: an attention module
    keeps its output gate on itself from the q projection until the gate is taken, and
    a second micro-batch entering the module in between would clobber it (SGLang's
    reason for refusing TBO for ERNIE). The only stage boundary is after the send."""

    def __init__(self, layer):
        self.layer = layer

    def op_prepare_attn(
        self,
        state,
        positions,
        hidden_states,
        forward_batch,
        tbo_subbatch_index,
        zero_allocator,
        **kwargs,
    ):
        layer = self.layer
        x, post, comb = layer._hc_pre(
            hidden_states,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.input_layernorm,
        )
        state.update(
            dict(
                positions=positions,
                forward_batch=forward_batch,
                tbo_subbatch_index=tbo_subbatch_index,
                zero_allocator=zero_allocator,
                attn_input=x,
                attn_residual=(hidden_states, post, comb),
            )
        )

    def op_attn(self, state):
        from sglang.srt.models.ernie55 import Ernie5_5MlaAttention

        layer, x = self.layer, state.pop("attn_input")
        if isinstance(layer.self_attn, Ernie5_5MlaAttention):
            x = layer.self_attn(
                state.positions, x, state.forward_batch, state.zero_allocator
            )
        else:
            x = layer.self_attn(x, state.positions, state.forward_batch)
        state.attn_output = x

    def op_post_attn_pre_mlp(self, state):
        layer = self.layer
        hidden_states, post, comb = state.pop("attn_residual")
        hidden_states = layer._hc_post(
            state.pop("attn_output"), hidden_states, post, comb
        )
        x, post, comb = layer._hc_pre(
            hidden_states,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.post_attention_layernorm,
        )
        state.hidden_states_mlp_input = x
        state.ffn_residual = (hidden_states, post, comb)

    def op_dense_mlp(self, state):
        state.hidden_states_mlp_output = self.layer._forward_ffn(
            state.pop("hidden_states_mlp_input"), state.forward_batch
        )

    def op_asap_dispatch(self, state):
        """Router and top-k, the latent projection, the send to the FFN ranks, then the
        shared experts, which run while the FFN ranks compute."""
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        mlp, x = self.layer.mlp, state.pop("hidden_states_mlp_input")
        if x.shape[0] == 0:
            # As Ernie5_5Moe.forward: nothing to route, nothing sent.
            state.asap_dispatch, state.asap_shared = None, x
            return
        topk_output = mlp.topk(x, linear_bf16_fp32(x, mlp.gate_weight))
        latent, _ = mlp.fc1_latent_proj(x)
        state.asap_dispatch = mlp.experts.dispatcher.dispatch(
            latent, topk_output, micro_batch=state.tbo_subbatch_index
        )
        state.asap_shared = mlp.shared_experts(x)

    def op_asap_combine(self, state):
        """Wait for the FFN ranks' answer, project it back and add the shared experts."""
        mlp = self.layer.mlp
        dispatch, shared = state.pop("asap_dispatch"), state.pop("asap_shared")
        if dispatch is None:
            state.hidden_states_mlp_output = shared
            return
        routed = mlp.experts.dispatcher.combine(dispatch)
        routed, _ = mlp.fc2_latent_proj(mlp.latent_norm(routed))
        state.hidden_states_mlp_output = routed + shared

    def op_postprocess(self, state):
        layer = self.layer
        hidden_states, post, comb = state.pop("ffn_residual")
        hidden_states = layer._hc_post(
            state.pop("hidden_states_mlp_output"), hidden_states, post, comb
        )
        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
            zero_allocator=state.zero_allocator,
        )
        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
                "zero_allocator",
            }
        )
        return output
