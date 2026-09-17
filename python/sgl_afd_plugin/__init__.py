"""Sglang AFD Plugin Entry Point."""

# Imported for its side effect: this import registers the CLI hooks.
from sgl_afd_plugin.srt.config import get_config    # noqa: F401

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


# Register the plugin with the SGLang runtime.
# Call HookRegistry.register() to register hooks.
def register():
    pass
