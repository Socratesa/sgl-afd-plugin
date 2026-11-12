"""ERNIE 5.5 Flash on an ASAP FFN rank.

The attention ranks run SGLang's ``Ernie5_5_MoeForCausalLM``: every MoE
layer's ``Ernie5_5Moe`` keeps its router, latent projections and shared expert and
gets ``AsapMoE`` for the routed experts, which run in the ``moe_latent_size`` space.
An FFN rank builds only those routed experts: the ``mlp.experts`` that
``Ernie5_5Moe`` creates, with the same arguments, under the same parameter names
(``model.layers.{i}.mlp.experts.*``, the layers from ``first_k_dense_replace`` on).
The checkpoint therefore loads through SGLang's own expert loader and ``FusedMoE``
weight loaders, and the quant method prepares the experts as it does colocated.

SGLang has no two-batch overlap for ERNIE: ``forward`` below hooks
``Ernie5_5Model.forward`` of an attention rank to run the layers of a split prefill
batch as the operations of ``srt/batch_overlap/operations_strategy.py``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.configs.ernie55 import Ernie5_5_MoeConfig
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models import ernie55
from torch import nn


class Ernie5_5Moe(nn.Module):
    """The routed experts of ``Ernie5_5Moe``: no router, latent projections or shared
    expert. The top-k weights the attention ranks send already include
    ``per_expert_scale`` (``multi_router_topk``), as colocated."""

    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        beta, linear_beta = ernie55._situ_betas(config)
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.moe_latent_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="situ",
            gemm1_alpha=beta,
            gemm1_clamp_limit=linear_beta,
            reduce_results=False,
        )


class Ernie5_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp = Ernie5_5Moe(
            config, layer_id, quant_config, prefix=add_prefix("mlp", prefix)
        )


class Ernie5_5Model(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Keyed by layer id: the dense layers below first_k_dense_replace have none.
        self.layers = nn.ModuleDict(
            {
                str(i): Ernie5_5DecoderLayer(
                    config, i, quant_config, prefix=add_prefix(f"layers.{i}", prefix)
                )
                for i in range(config.first_k_dense_replace, config.num_hidden_layers)
            }
        )


class Ernie5_5_MoeForCausalLM(nn.Module):
    """``Ernie5_5_MoeForCausalLM`` as an FFN rank builds it."""

    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        config.validate_checkpoint_fields()
        self.config = config
        self.quant_config = quant_config
        self.model = Ernie5_5Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.start_layer, self.end_layer = 0, config.num_hidden_layers

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "An FFN rank calls the FusedMoE layers of its jobs directly (FfnScheduler)"
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """The routed-expert branch of ``Ernie5_5WeightLoader``. The FFN rank reads
        only its own experts (``read_routed_experts_only``)."""
        ExpertWeightLoader(self).load_all(weights)


class ExpertWeightLoader(ernie55.Ernie5_5WeightLoader):
    """``Ernie5_5WeightLoader`` with its expert loader and checks only: a tensor that
    is not a routed expert has no parameter here and is skipped, as the full model
    skips the ones it does not hold."""

    def __init__(self, model: Ernie5_5_MoeForCausalLM):
        self._init_tables(model, layers=model.model.layers, mla_layers=())

    def load(self, name: str, tensor: torch.Tensor) -> None:
        layer_id, _, suffix = name.removeprefix("model.layers.").partition(".")
        if (
            name.startswith("model.layers.")
            and layer_id in self.layers
            and suffix.startswith("block_sparse_moe.experts.")
        ):
            self._load_expert(f"model.layers.{layer_id}.", suffix, tensor)


EntryClass = [Ernie5_5_MoeForCausalLM]


def forward(
    original,
    self,
    input_ids,
    positions,
    forward_batch,
    input_embeds=None,
    pp_proxy_tensors=None,
):
    """AROUND ``Ernie5_5Model.forward`` on an attention rank whose experts run on the
    FFN ranks. A prefill batch that was split into two micro-batches
    (``--enable-two-batch-overlap``) runs its layers as overlapped operations, as
    ``DeepseekV4Model._forward_layers_tbo`` does for DeepSeek-V4; any other batch takes
    SGLang's path. The embedding and the head after the layers are SGLang's own
    (``Ernie5_5Model.forward`` with one pipeline stage and no DSpark capture)."""
    if not _can_run_tbo(self, forward_batch):
        return original(
            self, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
    from sglang.srt.layers.layernorm import exact_rms_norm

    hidden_states = (
        self.embed_tokens(input_ids) if input_embeds is None else input_embeds
    )
    hidden_states = (
        hidden_states.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
    )
    hidden_states = _forward_layers_tbo(self, positions, hidden_states, forward_batch)
    pre_hc_head = hidden_states.flatten(1)
    rows = ernie55.logits_rows(forward_batch) if hidden_states.shape[0] > 0 else None
    hidden_states = self.hc_head(hidden_states if rows is None else hidden_states[rows])
    hidden_states = exact_rms_norm(hidden_states, self.norm)
    if rows is not None:
        hidden_states = ernie55.scatter_rows(hidden_states, rows, pre_hc_head.shape[0])
    return hidden_states, pre_hc_head, []


def _can_run_tbo(model, forward_batch) -> bool:
    """The gate of ``DeepseekV4Model._can_run_tbo`` without its path check. SGLang
    overlaps only an EP all-to-all or the attention-DP gather; the transfer to the FFN
    ranks, which this overlaps, is neither."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.moe import is_tbo_enabled

    return (
        is_tbo_enabled()
        and forward_batch.can_run_tbo
        and forward_batch.tbo_children is not None
        and forward_batch.global_forward_mode is not None
        # Target-verify also reports is_extend(); only a real prefill is overlapped.
        and forward_batch.global_forward_mode.is_extend_without_speculative()
        and not is_cp_active(forward_batch)
        and model.pp_group.world_size == 1
        and not model.dspark_layers_to_capture
    )


def _forward_layers_tbo(model, positions, hidden_states, forward_batch):
    import torch
    from sglang.srt.batch_overlap.operations import execute_overlapped_operations
    from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
    from sglang.srt.batch_overlap.two_batch_overlap import (
        _model_forward_filter_inputs,
        _model_forward_tbo_merge_key,
    )
    from sglang.srt.utils import BumpAllocator

    layers = [model.layers[i] for i in range(model.start_layer, model.end_layer)]
    strategy = OperationsStrategy.init_new_tbo(
        layers, forward_batch.global_forward_mode
    )
    # Sized as for a whole batch, twice over: both micro-batches take from it.
    zero_allocator = BumpAllocator(
        buffer_size=2 * 2 * len(layers),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    inputs_arr = [
        dict(
            **_model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=None,
                positions=positions,
                output_forward_batch=child,
                tbo_subbatch_index=index,
            ),
            zero_allocator=zero_allocator,
        )
        for index, child in enumerate(forward_batch.tbo_children)
    ]
    outputs = execute_overlapped_operations(
        inputs_arr=inputs_arr,
        operations_arr=[strategy.operations] * 2,
        delta_stages=[0, strategy.tbo_delta_stages],
    )
    return _model_forward_tbo_merge_key(
        outputs[0], outputs[1], "hidden_states", hidden_states.shape[0]
    )
