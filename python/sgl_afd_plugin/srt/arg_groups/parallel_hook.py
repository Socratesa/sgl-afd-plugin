"""AFD overrides for SGLang's EPLB / dispatch resolution."""

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _after_handle_eplb_and_dispatch(result, server_args):
    cfg = resolving_view(server_args)
    if cfg.ep_dispatch_algorithm is None:
        declare_resolution(
            server_args,
            "afd.eplb_and_dispatch",
            ep_dispatch_algorithm=(
                "dynamic" if cfg.moe_a2a_backend == "none" else "static"
            ),
        )
    return result


def install():
    HookRegistry.register(
        "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch",
        _after_handle_eplb_and_dispatch,
        HookType.AFTER,
    )
