"""AFD overrides for SGLang's server-argument resolution."""

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


# AFD turns some DP slots into FFN slots that never serve requests. Only the dispatch
# paths that consult the controller's active-worker set stay correct under that, so the
# allowed set is pinned here.
#
# `follow_bootstrap_room` is not supported currently: it is not a load-balance policy but a
# prefill-addressing contract. The decode side independently recomputes
# `bootstrap_room % dp_size` (disaggregation/common/conn.py:1231) and fails the request on
# mismatch, so remapping the modulo base on the controller alone would break PD.
ALLOWED_LOAD_BALANCE_METHODS = ("round_robin", "total_requests", "total_tokens")


def _after_handle_load_balance_method(result, server_args):
    method = resolving_view(server_args).load_balance_method
    if method not in ALLOWED_LOAD_BALANCE_METHODS:
        raise ValueError(
            f"AFD only supports --load-balance-method "
            f"{'/'.join(ALLOWED_LOAD_BALANCE_METHODS)}, but the effective value is "
            f"{method!r}. Pass valid methods explicitly, or disable "
            f"AFD (unset SGLANG_AFD_PLUGIN_ENABLED)."
        )

    return result


def install():
    HookRegistry.register(
        "sglang.srt.arg_groups.serving_hook.handle_load_balance_method",
        _after_handle_load_balance_method,
        HookType.AFTER,
    )
