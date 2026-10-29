"""Base-tier hooks: CLI flags and launcher-side activation."""

from sglang.srt.plugins.hook_registry import HookType

from sgl_asap_plugin.hooks import BASE, HookSpec


def _after_add_cli_args(result, parser):
    from sgl_asap_plugin import config

    config.add_cli_args(parser)
    return result


def _before_from_cli_args(cls, args):
    """Runs before ServerArgs resolution so role hooks can take part in it.

    Also covers ``--config file.yaml`` because it looks at the parsed namespace, not argv.
    """
    from sgl_asap_plugin import activation

    activation.activate_from_namespace(args)
    return None


SPECS = [
    HookSpec("sglang.srt.server_args.ServerArgs.add_cli_args", _after_add_cli_args,
             HookType.AFTER, ("parser",), BASE),
    HookSpec("sglang.srt.server_args.ServerArgs.from_cli_args", _before_from_cli_args,
             HookType.BEFORE, ("cls", "args"), BASE),
]
