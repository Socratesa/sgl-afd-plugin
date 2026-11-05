"""Shared experts between ASAP dispatch and combine, without changing SGLang.

Install one persistent pre-combine hook per MoE block after model loading. It
computes shared experts on the current compute stream, before the remote wait.
The stock forward's later shared-expert call consumes that result. Its routing,
scaling, output merge and return_moe_output behavior remain SGLang's own.
"""

from __future__ import annotations


def install_shared_experts_overlap(model) -> int:
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    from sgl_asap_plugin.srt.layers.moe.ep_moe.layer import AsapMoE

    count = 0
    for mlp in model.modules():
        if (
            isinstance(mlp, DeepseekV2MoE)
            and isinstance(mlp.experts, AsapMoE)
            and hasattr(mlp, "shared_experts")
        ):
            if mlp.experts.shared_experts_overlap is None:
                SharedExpertsOverlap(mlp)
            count += 1
    return count


class SharedExpertsOverlap:
    """One layer's permanent hooks and temporary normal-forward state.

    A scheduler executes normal forwards serially; they never yield to TBO.
    TBO calls the shared helper outside this scope and follows its original path.
    Clear all tensor references on both successful and failed normal forwards.
    """

    def __init__(self, mlp):
        if mlp.experts.moe_runner_config.inplace:
            raise ValueError("ASAP shared overlap requires out-of-place experts")
        self._forward_normal = mlp.forward_normal
        self._forward_shared = mlp._forward_shared_experts
        self.active = False
        self._hidden_states = None
        self._allocator = None
        self.pre_quant_input = None
        self._shared_output = None
        self._shared_ready = False
        # Use the stock deferred-shared path, not its temporary two-stream hooks.
        mlp._fuse_shared_experts_inside_sbo = False
        self._handle = mlp.experts.dispatcher.register_pre_combine_hook(
            self.pre_combine
        )
        mlp.experts.shared_experts_overlap = self
        mlp.forward_normal = self.forward_normal
        mlp._forward_shared_experts = self.forward_shared_experts

    def forward_normal(
        self,
        hidden_states,
        gemm_output_zero_allocator=None,
        input_ids=None,
        input_ids_global=None,
        skip_shared_experts=False,
        num_token_non_padded=None,
        use_vision_topk=False,
        return_moe_output=False,
    ):
        if self.active:
            raise RuntimeError("Concurrent normal forwards of the same ASAP MoE")
        self.active = True
        if not skip_shared_experts and hidden_states.shape[0] > 0:
            self._hidden_states = hidden_states
            self._allocator = gemm_output_zero_allocator
        try:
            return self._forward_normal(
                hidden_states,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global=input_ids_global,
                skip_shared_experts=skip_shared_experts,
                num_token_non_padded=num_token_non_padded,
                use_vision_topk=use_vision_topk,
                return_moe_output=return_moe_output,
            )
        finally:
            self.active = False
            self._hidden_states = None
            self._allocator = None
            self.pre_quant_input = None
            self._shared_output = None
            self._shared_ready = False

    def pre_combine(self, dispatcher, combine_input):
        if self._hidden_states is not None:
            try:
                self._shared_output = self._forward_shared(
                    self._hidden_states,
                    self._allocator,
                    pre_quant_input=self.pre_quant_input,
                )
            except BaseException:
                # No receive wait will be queued if this hook fails. Protect
                # send sources before unwinding releases dispatch's keepalive.
                if combine_input.keepalive:
                    from sgl_asap_plugin.srt.layers.moe.token_dispatcher.asap import (
                        AsapBuffer,
                    )

                    send_stream = AsapBuffer.get().send_stream
                    for tensor in combine_input.keepalive:
                        tensor.record_stream(send_stream)
                raise
            self._shared_ready = True

    def forward_shared_experts(
        self, hidden_states, gemm_output_zero_allocator=None, pre_quant_input=None
    ):
        if self._shared_ready and hidden_states is self._hidden_states:
            output = self._shared_output
            self._shared_output = None
            self._shared_ready = False
            return output
        return self._forward_shared(
            hidden_states,
            gemm_output_zero_allocator,
            pre_quant_input=pre_quant_input,
        )
