"""A-side stand-in for sglang's FusedMoE: no weights, the routed experts run on the F service.

``DeepseekV2MoE.__init__`` builds ``self.experts`` through ``get_moe_impl_class(quant_config)(...)``
and reads ``should_fuse_routed_scaling_factor_in_topk`` from it to configure its TopK, so the
stand-in must carry the value the real experts would have. That value is a property of the F
side's quantization method, so it is asked from the F service (``describe``) the first time a
stand-in is built. The forward of ``DeepseekV2MoE`` itself is replaced (``afd_forward``);
calling this module directly is a bug.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional

import torch.nn as nn

# Quantization methods whose routed-scale handling matches afd_forward's combine (stock
# maybe_fuse_routed_scale_and_shared_add with `fused=True`).
FUSED_QUANT_METHODS = ("Mxfp4FlashinferCutlassMoEMethod", "Mxfp4FlashinferTrtllmMoEMethod", "Mxfp4MarlinMoEMethod",
                       "ExpertPackMoEMethod")

_description: Optional[Dict] = None
_layers: Dict[int, dict] = {}


def set_description(desc: Dict) -> None:
    """Install the F service's description (tests; production fetches it lazily)."""
    global _description
    _description = desc


def ffn_description() -> Dict:
    global _description
    if _description is None:
        from sgl_asap_plugin.model.client import describe_ffn

        _description = describe_ffn()
    return _description


def model_dims() -> Dict:
    """What the stand-ins saw: hidden size, top-k and the MoE layer ids of this model."""
    if not _layers:
        raise RuntimeError("[ASAP] no AfdRemoteExperts was built in this process")
    first = next(iter(_layers.values()))
    return {"hidden": first["hidden"], "topk": first["topk"], "layers": sorted(_layers)}


class AfdRemoteExperts(nn.Module):
    def __init__(self, num_experts, num_fused_shared_experts, top_k, hidden_size, intermediate_size, layer_id,
                 quant_config=None, routed_scaling_factor=None, routing_method_type=None, swiglu_limit=None,
                 prefix="", **_unused):
        super().__init__()
        if num_fused_shared_experts:
            raise RuntimeError("[ASAP] fused shared experts are not supported by the AFD experts stand-in")
        desc = ffn_description()
        if desc["quant_method"] not in FUSED_QUANT_METHODS:
            raise RuntimeError(f"[ASAP] the F service uses {desc['quant_method']}, whose routed-scale handling "
                               f"afd_forward does not reproduce (supported: {FUSED_QUANT_METHODS})")
        self.layer_id, self.num_experts, self.top_k, self.hidden_size = layer_id, num_experts, top_k, hidden_size
        self.routed_scaling_factor = routed_scaling_factor
        # the value the real experts have: F's quantization method decides it
        self.should_fuse_routed_scaling_factor_in_topk = bool(desc["fuse_routed_scaling_in_topk"])
        self.num_local_experts = 0  # nothing is stored here
        self.moe_runner_config = SimpleNamespace(inplace=False)
        _layers[layer_id] = {"hidden": hidden_size, "topk": top_k, "experts": num_experts, "prefix": prefix}

    def forward(self, *_args, **_kwargs):
        raise RuntimeError("[ASAP] AfdRemoteExperts has no weights; DeepseekV2MoE.forward must be replaced by "
                           "the AFD forward (hook sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward)")
