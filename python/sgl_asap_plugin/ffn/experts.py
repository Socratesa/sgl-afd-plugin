"""The routed experts of one F rank, built and loaded the way sglang builds them.

Each MoE layer is a stock ``FusedMoE`` constructed with the arguments ``DeepseekV2MoE`` passes, so the
quantization method (flashinfer MXFP4 W4A16 on Hopper), the weight layout and the kernels are exactly
the colocated model's. This rank is expert-parallel rank ``f`` of ``E`` and owns the experts
``[lo, hi)``; it never talks to the other F ranks, and the topk ids it receives are global.

Differences from running inside ``ModelRunner``, all deliberate:

* no process groups: the ranks do not communicate, so only the published rank bundle is used and the
  TP group is replaced by a stub (it is read only to allocate "symmetric memory", which stays off);
* the checkpoint is read by tensor name, only this rank's experts of the layers that exist;
* the forward calls dispatch / run_moe_core / combine directly, skipping ``FusedMoE.forward_impl``
  (which needs the group and a forward context, and would all-reduce).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sgl_asap_plugin.layout import ExpertPartition
from sgl_asap_plugin.model import weights as W

logger = logging.getLogger("sgl_asap_plugin")

_PROJ = {"gate_proj": ("w13_", "w1"), "up_proj": ("w13_", "w3"), "down_proj": ("w2_", "w2")}
_HF_NAME = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
                      r"(weight|weight_scale_inv)$")


class _NoGroup:
    """Stands in for the TP group: F ranks never communicate and symmetric memory is not used."""


@dataclass
class LoadStats:
    tensors: int
    seconds: float
    post_seconds: float
    weight_bytes: int


def init_f_process(server_args, f: int, gpu_id: int) -> None:
    """Publish this process's place (EP rank f of E, no process groups) and the global kernel configs."""
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish

    torch.cuda.set_device(gpu_id)
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=f, gpu_id=gpu_id))
    get_parallel().override_permanently(tp_group=_NoGroup())
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_bf16_gemm_config()
    initialize_fp4_gemm_config()


class ExpertStack:
    def __init__(self, server_args, partition: ExpertPartition, f: int, gpu_id: int, layers: Optional[int] = None):
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.utils import RoutingMethodType
        from sglang.srt.model_loader.loader import _get_quantization_config

        self.f, self.gpu_id, self.partition = f, gpu_id, partition
        self.model_config = ModelConfig.from_server_args(server_args)
        cfg = self.model_config.hf_config
        self.hf = cfg
        self.model_path = self.model_config.model_path
        self.num_layers = cfg.num_hidden_layers if layers is None else layers
        self.lo, self.hi = partition.real_range(f)
        quant_config = _get_quantization_config(self.model_config, LoadConfig(load_format="auto", tp_rank=f))
        t0 = time.time()
        with torch.device(f"cuda:{gpu_id}"):
            self.layers: List = [
                FusedMoE(num_experts=partition.padded, num_fused_shared_experts=0, top_k=cfg.num_experts_per_tok,
                         hidden_size=cfg.hidden_size, intermediate_size=cfg.moe_intermediate_size, layer_id=i,
                         quant_config=quant_config, routed_scaling_factor=cfg.routed_scaling_factor,
                         routing_method_type=getattr(cfg, "routing_method_type", RoutingMethodType.DeepSeekV3),
                         swiglu_limit=getattr(cfg, "swiglu_limit", None), prefix=f"model.layers.{i}.mlp.experts")
                for i in range(self.num_layers)]
        logger.info("[ASAP] F rank %d built %d expert layers in %.1fs (%d of %d experts each, %s)", f,
                    self.num_layers, time.time() - t0, self.layers[0].num_local_experts, partition.padded,
                    type(self.layers[0].quant_method).__name__)

    # ---- facts the A side needs ----
    @property
    def quant_method(self) -> str:
        return type(self.layers[0].quant_method).__name__

    @property
    def fuse_routed_scaling_in_topk(self) -> bool:
        return bool(self.layers[0].should_fuse_routed_scaling_factor_in_topk)

    def describe(self) -> dict:
        cfg = self.hf
        return {"quant_method": self.quant_method, "fuse_routed_scaling_in_topk": self.fuse_routed_scaling_in_topk,
                "model": {"hidden": cfg.hidden_size, "topk": cfg.num_experts_per_tok,
                          "layers": list(range(self.num_layers)), "n_experts": cfg.n_routed_experts,
                          "routed_scaling_factor": cfg.routed_scaling_factor},
                "ep": {"rank": self.f, "experts": [self.lo, self.hi]}}

    # ---- weights ----
    def load(self, workers: int = 4) -> LoadStats:
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        # parameter objects by name; dropped before the repack so the pre-repack tensors can be freed
        params: Dict[str, torch.nn.Parameter] = {
            f"model.layers.{i}.mlp.experts.{n}": p for i, layer in enumerate(self.layers)
            for n, p in layer.named_parameters()}
        keep = W.expert_selector(self.num_layers, self.lo, self.hi)
        files, expected = W.select_files(self.model_path, keep)
        want = self.num_layers * (self.hi - self.lo) * 6  # w1, w2, w3, each with its scale
        if expected != want:
            raise RuntimeError(f"[ASAP] the checkpoint holds {expected} tensors for experts [{self.lo}, {self.hi}) of "
                               f"{self.num_layers} layers, expected {want}")
        t0, loaded = time.time(), 0
        for raw, tensor in W.filtered_safetensors_iterator(files, keep, max_workers=workers):
            name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                raw, is_nextn=False, num_hidden_layers=self.num_layers)
            m = _HF_NAME.match(name)
            if m is None:
                raise RuntimeError(f"[ASAP] unexpected expert tensor name {raw!r} (remapped to {name!r})")
            layer, expert, proj, suffix = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            prefix, shard = _PROJ[proj]
            key = f"model.layers.{layer}.mlp.experts.{prefix}{suffix}"
            param = params[key]
            param.weight_loader(param, tensor, key, shard_id=shard, expert_id=expert)
            loaded += 1
        torch.cuda.synchronize(self.gpu_id)
        t1 = time.time()
        del params, param
        for layer in self.layers:
            layer.quant_method.process_weights_after_loading(layer)
        torch.cuda.synchronize(self.gpu_id)
        torch.cuda.empty_cache()
        stats = LoadStats(loaded, t1 - t0, time.time() - t1, torch.cuda.memory_allocated(self.gpu_id))
        if loaded != want:
            raise RuntimeError(f"[ASAP] loaded {loaded} expert tensors, expected {want}")
        logger.info("[ASAP] F rank %d loaded %d tensors in %.1fs (+%.1fs repack), %.2f GB on the GPU", self.f, loaded,
                    stats.seconds, stats.post_seconds, stats.weight_bytes / 1e9)
        return stats

    # ---- compute ----
    @torch.no_grad()
    def forward(self, layer_id: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Partial output of this rank's experts for rows ``x``: topk ids are global, others contribute zero."""
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        layer = self.layers[layer_id]
        topk = StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=None)
        dispatched = layer.dispatcher.dispatch(hidden_states=x, topk_output=topk)
        combined = layer.dispatcher.combine(combine_input=layer.run_moe_core(dispatch_output=dispatched))
        return combined[..., :x.shape[-1]].contiguous()

    def warmup(self, n_max: int) -> List[int]:
        """Run every token bucket once (kernels are tuned/compiled per next_power_of_2(n)), layer 0 for all
        buckets and every layer for a small one, so no request pays a first-call cost."""
        cfg, device = self.hf, f"cuda:{self.gpu_id}"
        buckets, n = [], 1
        while n < n_max:
            buckets.append(n)
            n *= 2
        buckets.append(n_max)
        gen = torch.Generator(device=device).manual_seed(0)
        for n in buckets:
            x = torch.randn(n, cfg.hidden_size, device=device, dtype=torch.bfloat16, generator=gen)
            ids = torch.rand(n, self.partition.n_experts, device=device, generator=gen).topk(
                cfg.num_experts_per_tok, dim=1).indices.to(torch.int32)
            w = torch.rand(n, cfg.num_experts_per_tok, device=device, generator=gen, dtype=torch.float32)
            for layer_id in ([0] if n > 256 else range(self.num_layers)):
                self.forward(layer_id, x, ids, w)
        torch.cuda.synchronize(self.gpu_id)
        return buckets
