"""AFD overrides for SGLang's server-argument validation."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.arg_groups.overrides import resolving_view


def _after_check_server_args(result, server_args):
    cfg = resolving_view(server_args)

    if not cfg.enable_dp_attention:
        raise ValueError(
            "AFD requires --enable-dp-attention: FFN slots are driven by the "
            "dp-attention MLP-sync idle batch, so without it an FFN slot never "
            "reaches run_batch()."
        )
    if cfg.dp_size <= 1:
        raise ValueError(
            f"AFD requires --dp-size > 1 (got {cfg.dp_size}): the MLP-sync "
            "all-gather and its idle batch are skipped at dp_size == 1, leaving "
            "FFN slots idle."
        )
    if cfg.nnodes <= 1:
        raise ValueError(
            f"AFD requires --nnodes > 1 (got {cfg.nnodes}): ATTN and FFN are "
            "started as separate launches (--afd-role is per launch) and must "
            "join one distributed world."
        )

    return result


def install():
    HookRegistry.register(
        "sglang.srt.arg_groups.validation_hook.check_server_args",
        _after_check_server_args,
        HookType.AFTER,
    )
