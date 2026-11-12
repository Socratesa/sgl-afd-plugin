"""Models with ASAP support, one module per family as in ``sglang.srt.models``.

* ``deepseek_v4.py``: ``DeepseekV4ForCausalLM`` (DeepSeek-V4-Flash)
* ``ernie55.py``: ``Ernie5_5_MoeForCausalLM`` (ERNIE 5.5 Flash)

A module defines, under the architecture name of the checkpoint (``EntryClass``), the
model an FFN rank builds: the routed experts of every MoE layer and nothing else. The
FFN role registers this package in SGLang's ``ModelRegistry`` with
``overwrite=True``, so SGLang's loader builds that class instead of the full model.
The attention role runs SGLang's own class; its MoE layers get ``AsapMoE`` from
``get_moe_impl_class``. Both roles refuse an architecture that has no module here.
"""

from typing import Dict


def asap_model_classes() -> Dict[str, type]:
    """Architecture -> the model an FFN rank builds."""
    from sglang.srt.models.registry import import_model_classes

    return import_model_classes(__name__, strict=True)


def check_asap_model(architecture: str) -> None:
    supported = asap_model_classes()
    if architecture not in supported:
        raise ValueError(
            f"ASAP has no model for {architecture}; supported: {sorted(supported)} "
            "(sgl_asap_plugin/srt/models)"
        )


def register_ffn_models() -> None:
    """On the FFN role: SGLang's loader builds the classes of this package."""
    from sglang.srt.models.registry import ModelRegistry

    ModelRegistry.register(__name__, overwrite=True, strict=True)
