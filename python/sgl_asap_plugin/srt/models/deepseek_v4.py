"""DeepSeek-V4 on an ASAP FFN rank.

The attention ranks run SGLang's ``DeepseekV4ForCausalLM`` unchanged: every decoder
layer's ``DeepseekV2MoE`` keeps its gate, top-k and shared experts and gets
``AsapMoE`` for the routed experts. An FFN rank builds only those routed experts:
the ``mlp.experts`` that ``DeepseekV2MoE`` creates, with the same arguments, under
the same parameter names (``model.layers.{i}.mlp.experts.*``). The checkpoint
therefore loads through SGLang's own name mapping and ``FusedMoE`` weight loaders,
and the quant method prepares the experts as it does colocated.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.models import deepseek_v4
from sglang.srt.utils import add_prefix
from torch import nn

logger = logging.getLogger(__name__)


class DeepseekV4MoE(nn.Module):
    """The routed experts of ``DeepseekV2MoE``: no gate, top-k or shared experts.

    The attention ranks send top-k weights that already include
    routed_scaling_factor (``AsapMoE``), so the experts are built without it: a
    runner that scales its output itself (humming, marlin) must not scale again.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            num_fused_shared_experts=0,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            routed_scaling_factor=None,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=add_prefix("experts", prefix),
        )


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp = DeepseekV4MoE(
            config, layer_id, quant_config, prefix=add_prefix("mlp", prefix)
        )


class DeepseekV4Model(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DeepseekV4DecoderLayer(
                config, i, quant_config, prefix=add_prefix(f"layers.{i}", prefix)
            )
            for i in range(config.num_hidden_layers)
        )


class DeepseekV4ForCausalLM(nn.Module):
    """``DeepseekV4ForCausalLM`` as an FFN rank builds it."""

    # What SGLang reads from the model class, the same as for the full model.
    get_hf_to_sglang_mapper = deepseek_v4.DeepseekV4ForCausalLM.get_hf_to_sglang_mapper
    get_model_config_for_expert_location = (
        deepseek_v4.DeepseekV4ForCausalLM.get_model_config_for_expert_location
    )

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.start_layer, self.end_layer = 0, config.num_hidden_layers

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "An FFN rank calls the FusedMoE layers of its jobs directly (FfnScheduler)"
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """The routed-expert branch of ``DeepseekV4ForCausalLM.load_weights``. The FFN
        rank reads only its own experts (``read_routed_experts_only``); a tensor
        without a parameter here is skipped, as the full model skips the ones it does
        not hold."""
        params = dict(self.named_parameters())
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        remap = deepseek_v4.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        loaded = set()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for name, loaded_weight in weights:
                name = remap(name, num_hidden_layers=self.config.num_hidden_layers)
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    resolved_name = name.replace(weight_name, param_name)
                    if resolved_name not in params:
                        continue
                    param = params[resolved_name]
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=should_async_load(loaded_weight),
                        func=param.weight_loader,
                        func_args=(param, loaded_weight, resolved_name),
                        func_kwargs={"shard_id": shard_id, "expert_id": expert_id},
                    )
                    loaded.add(resolved_name)
                    break
            for future in concurrent.futures.as_completed(futures):
                future.result()
        unloaded = params.keys() - loaded
        if unloaded:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded}"
            )


EntryClass = [DeepseekV4ForCausalLM]
