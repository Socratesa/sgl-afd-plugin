"""When and how the plugin turns itself on.

* base tier (every process, cheap): ``--asap-*`` flags and a watcher on ``ServerArgs.from_cli_args``.
* role tier (ASAP deployments only): validation and per-scheduler runtime hooks.

The role tier is activated in the launcher from the parsed CLI (before ServerArgs
resolution runs) and in every child process from the inherited ``ASAP_ROLE`` variable.
Once a user has opted in, failures are loud (SystemExit / exception), never silently
ignored the way sglang's plugin loader treats ordinary plugin errors.
"""

from __future__ import annotations

import logging
import os

from sgl_asap_plugin import compat, hooks
from sgl_asap_plugin.config import ENV_ROLE, AsapConfig
from sgl_asap_plugin.errors import AsapConfigError, AsapInstallError

logger = logging.getLogger("sgl_asap_plugin")


def register_plugin() -> None:
    """Entry point body; sglang's loader applies the registered hooks afterwards."""
    try:
        hooks.register_tier(hooks.BASE)
    except AsapInstallError as exc:
        # Not opted in yet: never break plain sglang because of version drift.
        logger.warning("%s\n[ASAP] the plugin stays disabled in this process.", exc)
        return
    if os.environ.get(ENV_ROLE):
        try:
            _activate_role(AsapConfig.from_env(), apply=False)
        except (AsapConfigError, AsapInstallError) as exc:
            # SystemExit is not an Exception: sglang's loader does not swallow it.
            raise SystemExit(f"[ASAP] {ENV_ROLE}={os.environ[ENV_ROLE]} but the plugin cannot start: {exc}")
        compat.warn_if_untested()


def activate_from_namespace(ns) -> None:
    """Body of the ``ServerArgs.from_cli_args`` BEFORE hook (launcher process)."""
    config = AsapConfig.from_namespace(ns)
    if config is None:
        return
    _activate_role(config, apply=True)
    config.export_env()  # children inherit the config
    compat.warn_if_untested()


def _activate_role(config: AsapConfig, *, apply: bool) -> None:
    if config.role != "attn":
        raise AsapConfigError(f"[ASAP] role {config.role!r} cannot run under sglang serve; "
                              "start the expert service with: python -m sgl_asap_plugin.ffn_serve")
    hooks.register_tier(hooks.ROLE)
    if config.ffn_ranks is not None:  # AFD mode: experts are remote
        hooks.register_tier(hooks.AFD)
    if apply:
        from sglang.srt.plugins.hook_registry import HookRegistry

        HookRegistry.apply_hooks()
        compat.ensure_installed(hooks.active_specs())
