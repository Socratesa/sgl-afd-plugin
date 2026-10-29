"""Role-tier hook: validate the resolved server args of an ASAP deployment."""

import logging

from sglang.srt.plugins.hook_registry import HookType

from sgl_asap_plugin.hooks import ROLE, HookSpec

logger = logging.getLogger("sgl_asap_plugin")


def _before_check_server_args(server_args):
    """Runs before sglang's own checks, so a coupled configuration is reported in ASAP terms."""
    from sglang.srt.arg_groups.overrides import resolving_view

    from sgl_asap_plugin import compat, hooks, validation
    from sgl_asap_plugin.config import AsapConfig

    asap = AsapConfig.from_env()
    if asap is None:
        return None
    specs = hooks.active_specs()
    compat.ensure_installed(specs)  # a hook that failed to apply must stop the launch
    validation.check_for_role(resolving_view(server_args), asap)
    patched = len(specs) - len(compat.missing_patches(specs))
    logger.info("[ASAP] server args ok: role=%s hooks=%d/%d sglang=%s",
                asap.role, patched, len(specs), compat.sglang_version())
    return None


SPECS = [
    HookSpec("sglang.srt.arg_groups.validation_hook.check_server_args", _before_check_server_args,
             HookType.BEFORE, ("server_args",), ROLE),
]
