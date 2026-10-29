"""AFD-tier hooks (attention role with --asap-ffn-ranks): the routed experts run on the F service.

* ``get_moe_impl_class`` returns a weightless stand-in, so no expert tensor is ever allocated.
* ``DeepseekV2MoE.forward`` is wrapped; layers holding the stand-in take the AFD path.
* The two safetensors iterators skip routed-expert tensors before reading them (about 147 GB
  of the 166.9 GB checkpoint), so an A rank loads only what it keeps.

Bodies import torch-dependent code lazily: hooks are declared in every process.
"""

from sglang.srt.plugins.hook_registry import HookType

from sgl_asap_plugin.hooks import AFD, HookSpec


def _remote_experts_class(quant_config):
    from sgl_asap_plugin.model.remote_experts import AfdRemoteExperts

    return AfdRemoteExperts


def _around_moe_forward(original, *args, **kwargs):
    from sgl_asap_plugin.model.afd_forward import moe_forward

    return moe_forward(original, *args, **kwargs)


def _multi_thread_iterator(hf_weights_files, max_workers, disable_mmap=False, prefetch=False, prefetch_num_threads=4,
                           drop_cache_after_load=False):
    from sgl_asap_plugin.model.weights import attn_weights_iterator

    return attn_weights_iterator(hf_weights_files, max_workers, disable_mmap, prefetch, prefetch_num_threads,
                                 drop_cache_after_load)


def _single_thread_iterator(hf_weights_files, disable_mmap=False, prefetch=False, prefetch_num_threads=4,
                            drop_cache_after_load=False):
    from sgl_asap_plugin.model.weights import attn_weights_iterator

    return attn_weights_iterator(hf_weights_files, 1, disable_mmap, prefetch, prefetch_num_threads,
                                 drop_cache_after_load)


_WEIGHT_UTILS = "sglang.srt.model_loader.weight_utils."

SPECS = [
    HookSpec("sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class", _remote_experts_class, HookType.REPLACE,
             ("quant_config",), AFD),
    HookSpec("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward", _around_moe_forward, HookType.AROUND,
             ("self", "hidden_states", "forward_batch", "gemm_output_zero_allocator", "input_ids",
              "input_ids_global", "skip_shared_experts", "return_moe_output"), AFD),
    HookSpec(_WEIGHT_UTILS + "buffered_multi_thread_safetensors_weights_iterator", _multi_thread_iterator,
             HookType.REPLACE, ("hf_weights_files", "max_workers", "disable_mmap", "prefetch",
                                "prefetch_num_threads", "drop_cache_after_load"), AFD),
    HookSpec(_WEIGHT_UTILS + "safetensors_weights_iterator", _single_thread_iterator, HookType.REPLACE,
             ("hf_weights_files", "disable_mmap", "prefetch", "prefetch_num_threads", "drop_cache_after_load"), AFD),
]
