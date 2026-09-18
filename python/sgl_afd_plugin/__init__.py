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

    pass


if not SGLANG_AFD_PLUGIN_ENABLED:
    # Skip registration if not enabled.
    register = lambda: warnings.warn(
        "SGLang AFD plugin is disabled by default. "
        "Set SGLANG_AFD_PLUGIN_ENABLED=1 to enable."
    )