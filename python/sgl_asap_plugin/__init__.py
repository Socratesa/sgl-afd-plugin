"""SGLang plugin: ASAP-style prefill Attention-FFN disaggregation.

sglang calls ``register()`` in every process it starts (main, DPC, scheduler,
detokenizer), so importing this package must stay cheap: no torch, no CUDA.
"""

__version__ = "0.1.0"


def register() -> None:
    """Entry point of the ``sglang.srt.plugins`` group."""
    from sgl_asap_plugin.activation import register_plugin

    register_plugin()
