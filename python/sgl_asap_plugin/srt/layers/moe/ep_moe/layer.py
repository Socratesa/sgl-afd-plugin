"""The MoE layer of an attention rank whose routed experts run on the FFN ranks.

``get_moe_impl_class`` replaces SGLang's function of the same name on such a rank, so
every model that builds its routed experts through it gets ``AsapMoE``. SGLang's MoE
modules are imported when a layer is built, not with this module: the hook is
registered in every process SGLang starts, most of which never build a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


def get_moe_impl_class(quant_config=None, *, context=None):
    """REPLACE of ``sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class``."""
    return AsapMoE


class AsapMoE(torch.nn.Module):
    """A FusedMoE without expert weights: the model keeps its gate, top-k and shared
    experts, and ``forward`` hands the routed experts to the FFN ranks through
    ``AsapDispatcher``.

    It has the FusedMoE attributes that model code reads.
    ``should_fuse_routed_scaling_factor_in_topk`` is True, so the top-k weights it
    sends already include routed_scaling_factor (as the MXFP4 FlashInfer methods have
    it colocated) and the FFN ranks never scale. ``quant_method`` is None, so
    ``maybe_fuse_routed_scale_and_shared_add`` just adds the shared output.
    """

    should_fuse_routed_scaling_factor_in_topk = True
    quant_method = None

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        from sgl_asap_plugin.srt.layers.moe.token_dispatcher.asap import (
            AsapDispatcher,
        )

        super().__init__()
        if num_fused_shared_experts:
            raise ValueError(
                "ASAP keeps the shared experts on the attention ranks: "
                "disable shared experts fusion"
            )
        self.num_experts = num_experts
        self.num_local_experts = 0
        self.num_fused_shared_experts = 0
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=0,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=params_dtype,
            inplace=False,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.dispatcher = AsapDispatcher(layer_id)
        self.shared_experts_overlap = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        pre_quant_input=None,
    ) -> torch.Tensor:
        # The FFN ranks run the experts between dispatch and combine.
        overlap = self.shared_experts_overlap
        if overlap is not None and overlap.active:
            overlap.pre_quant_input = pre_quant_input
        dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)
        return self.dispatcher.combine(dispatch_output)
