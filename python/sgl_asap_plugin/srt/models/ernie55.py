"""ERNIE 5.5 Flash on an ASAP FFN rank.

The attention ranks run SGLang's ``Ernie5_5_MoeForCausalLM`` unchanged: every MoE
layer's ``Ernie5_5Moe`` keeps its router, latent projections and shared expert and
gets ``AsapMoE`` for the routed experts, which run in the ``moe_latent_size`` space.
An FFN rank builds only those routed experts: the ``mlp.experts`` that
``Ernie5_5Moe`` creates, with the same arguments, under the same parameter names
(``model.layers.{i}.mlp.experts.*``, the layers from ``first_k_dense_replace`` on).
The checkpoint therefore loads through SGLang's own expert loader and ``FusedMoE``
weight loaders, and the quant method prepares the experts as it does colocated.
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
