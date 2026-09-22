"""AFD overrides for SGLang's ServerArgs."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sgl_afd_plugin.afd.config import _AFD


def _after_add_cli_args(result, parser):
    _AFD.add_cli_args(parser)
    return result


def _after_from_cli_args(result, cls, namespace):
    _AFD.export_env(namespace)
    return result


def install():
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        _after_add_cli_args,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.from_cli_args",
        _after_from_cli_args,
        HookType.AFTER,
    )
