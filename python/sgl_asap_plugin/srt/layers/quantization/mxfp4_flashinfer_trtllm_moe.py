"""SiTU on the ``flashinfer_mxfp4`` experts of an FFN rank.

SGLang's ``Mxfp4FlashinferTrtllmMoEMethod`` runs the MXFP4 experts of a checkpoint whose
``quantization_config`` is fp8 (the layout of DeepSeek-V4 and of the ERNIE 5.5 FP4
checkpoints) on FlashInfer's TRT-LLM MoE kernels, with SwiGLU hardwired: it calls the
kernel with FlashInfer's default activation and never reads
``moe_runner_config.activation``. ERNIE 5.5's experts use SiTU, so the kernel would
compute the wrong function without raising. FlashInfer's kernels do implement SiTU (SGLang
uses it for Kimi-K3, through ``Mxfp4MoEMethod``); this hook calls them for it.

The weights need no change. The method already lays them out as the SiTU kernel reads
them: rows interleaved as (up_i, gate_i) pairs, which is also what ``Mxfp4MoEMethod``
builds for Kimi-K3's non-interleaved gate/up. The kernel computes
``beta * tanh(x0 / beta) * alpha * tanh(x1 / alpha) * sigmoid(x1)`` with x0 the up and x1
the gate row, so ``gemm1_alpha`` is SiTU's ``beta`` (``MoeRunnerConfig.gemm1_alpha``) and
``gemm1_beta`` its ``linear_beta`` (``MoeRunnerConfig.gemm1_clamp_limit``).
"""

from __future__ import annotations

import torch


def apply(original, self, layer, dispatch_output):
    """AROUND ``Mxfp4FlashinferTrtllmMoEMethod.apply``: SiTU layers run the SiTU kernel
    on a standard top-k output; every other layer is untouched."""
    config = self.moe_runner_config
    if config.activation != "situ":
        return original(self, layer, dispatch_output)

    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        routed_hidden_size,
    )
    from sglang.srt.utils.triton_utils import next_power_of_2

    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise NotImplementedError(
            "SiTU on flashinfer_mxfp4 takes a standard top-k output"
        )
    if self.flashinfer_mxfp4_moe_precision != "default":
        raise NotImplementedError(
            "SiTU on flashinfer_mxfp4 runs MXFP8 activations, not "
            f"flashinfer_mxfp4_moe_precision={self.flashinfer_mxfp4_moe_precision}"
        )

    alpha, beta = _situ_constants(layer, config)
    w13, w2 = layer.w13_weight, layer.w2_weight
    intermediate_size = w2.shape[2] * 2 if w2.dtype == torch.uint8 else w2.shape[2]
    hidden_size = routed_hidden_size(layer)
    x_quant, x_scale = self.quantize_routed_input(
        dispatch_output.hidden_states, hidden_size
    )

    num_tokens = x_quant.shape[0]
    # the kernel writes its result to `output`: return that, not the FFI return value
    output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=x_quant.device
    )
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(topk_output.topk_ids, topk_output.topk_weights),
        routing_bias=None,
        hidden_states=x_quant,
        hidden_states_scale=x_scale,
        gemm1_weights=w13,
        gemm1_weights_scale=layer.w13_weight_scale_inv,
        gemm1_bias=None,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=layer.w2_weight_scale_inv,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=layer.num_experts,
        top_k=topk_output.topk_ids.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
        local_num_experts=layer.num_local_experts,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.TopK),
        activation_type=ActivationType.Situ.value,
        tune_max_num_tokens=next_power_of_2(num_tokens),
        output=output,
        do_finalize=True,
        enable_pdl=trtllm_moe_enable_pdl(num_tokens),
    )
    return StandardCombineInput(hidden_states=output)


def _situ_constants(layer, config):
    """The per-expert (alpha, beta) tensors of the kernel: SiTU's ``beta`` and
    ``linear_beta``, made once per layer (the first call is outside any CUDA graph
    capture: the FFN rank runs every layer eagerly before capturing it)."""
    constants = getattr(layer, "_asap_situ_constants", None)
    if constants is None:
        if config.gemm1_alpha is None or config.gemm1_clamp_limit is None:
            raise ValueError(
                "SiTU needs the layer's gemm1_alpha (beta) and gemm1_clamp_limit (linear_beta)"
            )
        device = layer.w13_weight.device
        constants = tuple(
            torch.full(
                (layer.num_local_experts,), value, dtype=torch.float32, device=device
            )
            for value in (config.gemm1_alpha, config.gemm1_clamp_limit)
        )
        layer._asap_situ_constants = constants
    return constants
