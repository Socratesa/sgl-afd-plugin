"""SGLang plugin: ASAP-style prefill Attention-FFN disaggregation (see README.md).

sglang calls register() in every process it starts (launcher, DP controller, schedulers, detokenizer), so
importing this package stays cheap: no torch, no CUDA.
"""

__version__ = "0.2.0"


def register() -> None:
    """Entry point of the ``sglang.srt.plugins`` group."""
    from sgl_asap_plugin.hooks import register_plugin

    register_plugin()
