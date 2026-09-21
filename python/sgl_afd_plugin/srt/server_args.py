"""AFD overrides for SGLang's ServerArgs."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sgl_afd_plugin.afd.config import AFD_CONFIG


def _after_add_cli_args(result, parser):
    AFD_CONFIG.add_cli_args(parser)


def _around_from_cli_args(original_fn, *args):
    """``args`` is ``(cls, namespace)``.

    Env must be written before ServerArgs is constructed, so the spawned schedulers
    inherit it. ``args[-1]`` is the namespace either way.
    """
    AFD_CONFIG.export_env(args[-1])
    return original_fn(*args)


def install():
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        _after_add_cli_args,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.from_cli_args",
        _around_from_cli_args,
        HookType.AROUND,
    )
