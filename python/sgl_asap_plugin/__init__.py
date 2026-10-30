"""SGLang plugin: ASAP-style prefill Attention-FFN disaggregation (see README.md).

sglang calls register() in every process it starts (launcher, DP controller, schedulers, detokenizer), so
importing this package stays cheap: no torch, no CUDA.
"""

__version__ = "0.2.0"


def _activation_requested() -> bool:
    """Detect explicit opt-in before the upstream CLI hooks can run.

    Config-file options are handled by the parsed namespace once the base hooks
    are installed. An empty/malformed ASAP option is still an explicit request.
    """
    import os
    import sys

    return bool(os.environ.get("ASAP_ROLE")) or any(arg.startswith("--asap-") for arg in sys.argv[1:])


def register() -> None:
    """Entry point of the ``sglang.srt.plugins`` group."""
    try:
        from sgl_asap_plugin.hooks import register_plugin

        register_plugin()
    except Exception as exc:
        # The upstream loader catches Exception around each entry point. An
        # explicit deployment must not continue after even an import failure.
        if _activation_requested():
            raise SystemExit(f"[ASAP] plugin explicitly requested but cannot start: {type(exc).__name__}: {exc}") from exc
        import logging

        logging.getLogger("sgl_asap_plugin").warning(
            "[ASAP] plugin unavailable; continuing without ASAP: %s: %s", type(exc).__name__, exc
        )
