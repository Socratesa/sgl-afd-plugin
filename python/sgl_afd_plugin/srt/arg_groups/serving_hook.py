"""AFD overrides for SGLang's server-argument resolution."""

import logging

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)

# AFD turns half the DP slots into FFN slots that never serve requests. Only the dispatch
# paths that consult the controller's active-worker set stay correct under that, so the
# allowed set is pinned here.
#
# `total_requests` / `total_tokens` are in fact already covered -- the DPBudget.dispatch
# hook in srt/managers/data_parallel_controller.py masks FFN slots with sys.maxsize -- and
# can be added to this tuple once they are exercised end to end.
#
# `follow_bootstrap_room` cannot be fixed locally: it is not a load-balance policy but a
# prefill-addressing contract. The decode side independently recomputes
# `bootstrap_room % dp_size` (disaggregation/common/conn.py:1231) and fails the request on
# mismatch, so remapping the modulo base on the controller alone would break PD.
ALLOWED_LOAD_BALANCE_METHODS = ("round_robin",)


def _around_handle_load_balance_method(original_fn, server_args):
    """Reject any load_balance_method AFD cannot honour.

    Validates the *effective* value rather than the raw input. Native resolves `auto` to
    `follow_bootstrap_room` for PD prefill and to `round_robin` everywhere else
    (arg_groups/serving_hook.py:201-213), so `auto` has to be expanded before it can be
    judged -- hence original_fn runs first. That also keeps native's `disaggregation_mode`
    validation at :198 in place.

    `resolving_view` is a live read of the declaration stash
    (arg_groups/model_override_base.py:137-139), so it reflects what original_fn declared.
    """
    result = original_fn(server_args)

    method = resolving_view(server_args).load_balance_method
    if method not in ALLOWED_LOAD_BALANCE_METHODS:
        raise ValueError(
            f"AFD only supports --load-balance-method"
            f"{'/'.join(ALLOWED_LOAD_BALANCE_METHODS)}, but the effective value is "
            f"{method!r}. Pass --load-balance-method round_robin explicitly, or disable "
            f"AFD (unset SGLANG_AFD_PLUGIN_ENABLED)."
        )

    logger.info("[AFD][ServerArgs] load_balance_method=%s", method)
    return result


def install():
    HookRegistry.register(
        "sglang.srt.arg_groups.serving_hook.handle_load_balance_method",
        _around_handle_load_balance_method,
        HookType.AROUND,
    )
