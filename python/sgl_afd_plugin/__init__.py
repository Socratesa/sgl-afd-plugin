"""Sglang AFD Plugin Entry Point."""

import os
import warnings

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


SGLANG_AFD_PLUGIN_ENABLED = bool(int(os.getenv("SGLANG_AFD_PLUGIN_ENABLED", "0")))


# Register the plugin with the SGLang runtime.
# Call HookRegistry.register() to register hooks.
def register():
    # Imported for its side effect: this import registers the CLI hooks.
    from sgl_afd_plugin.srt.config import get_config    # noqa: F401

    from sgl_afd_plugin.srt.arg_groups.serving_hook import install as ag_install
    from sgl_afd_plugin.srt.entrypoints.engine import install as eng_install
    from sgl_afd_plugin.srt.managers.data_parallel_controller import (
        install as dpc_install,
    )

    ag_install()
    eng_install()
    dpc_install()


if not SGLANG_AFD_PLUGIN_ENABLED:
    # Skip registration if not enabled.
    register = lambda: warnings.warn(
        "SGLang AFD plugin is disabled by default. "
        "Set SGLANG_AFD_PLUGIN_ENABLED=1 to enable."
    )
