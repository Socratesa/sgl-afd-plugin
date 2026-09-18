"""Sglang AFD Plugin Entry Point."""

import os as _os
import warnings

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_plugin_enabled = lambda: bool(int(_os.getenv("SGLANG_AFD_PLUGIN_ENABLED", "0")))

# Register the plugin with the SGLang runtime.
# Call HookRegistry.register() to register hooks.
def register():
    if not _plugin_enabled():
        warnings.warn(
            "SGLang AFD plugin is disabled by default. "
            "Set SGLANG_AFD_PLUGIN_ENABLED=1 to enable."
        )
        return

    # Imported for its side effect: this import registers the CLI hooks.
    from sgl_afd_plugin.srt.config import get_config    # noqa: F401

    pass
