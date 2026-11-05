"""ASAP-style prefill Attention-FFN disaggregation for SGLang.

The package mirrors ``sglang``: every module under ``srt/`` sits at the path of the
SGLang module it hooks or extends, ``hooks.py`` lists all hooks, ``srt/models/`` the
supported models, and ``transport/`` is the communication library between the two
roles (independent of SGLang and torch, as DeepEP is).

Attention role (``--asap-role attn --asap-ffn-size E``): plain data-parallel replicas.
Each MoE layer keeps its gate, top-k and shared experts; its routed experts are an
``AsapMoE``, whose ``AsapDispatcher`` sends the tokens to the FFN ranks and sums
their outputs.

FFN role (``--asap-role ffn --tp-size E --ep-size E``): scheduler ranks that build the
routed experts of the model (``srt/models/``) and serve them to every attention rank.

SGLang calls ``register()`` in every process it starts, so importing this package
stays cheap.
"""

import os

__version__ = "0.4.0"


def register() -> None:
    """Entry point of the ``sglang.srt.plugins`` group."""
    from sgl_asap_plugin import hooks
    from sgl_asap_plugin.srt.server_args import ENV_ROLE, get_asap_args

    hooks.register_base_hooks()
    if os.environ.get(ENV_ROLE, "null") != "null":  # started by an ASAP launcher
        try:
            args = get_asap_args()
        except ValueError as e:
            # SGLang's loader logs and skips a plugin that raises an Exception.
            raise SystemExit(f"ASAP: invalid ASAP_* environment: {e}") from e
        hooks.register_role_hooks(args)
