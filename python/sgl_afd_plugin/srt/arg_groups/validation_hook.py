"""AFD overrides for SGLang's server-argument validation."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.arg_groups.overrides import resolving_view

# Mirrors is_deepep_class_backend() (layers/moe/utils.py:608).
ALLOWED_MOE_A2A_BACKENDS = ("deepep", "deepep_v2", "mooncake", "mori", "pplx")


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
    if cfg.enable_eplb:
        raise ValueError(
            "AFD refuses --enable-eplb: EPLB reads physical_to_logical_map as a "
            "dense index, but the attention ranks' slots are -1."
        )
    if cfg.expert_distribution_recorder_mode is not None:
        raise ValueError(
            "AFD refuses --expert-distribution-recorder-mode="
            f"{cfg.expert_distribution_recorder_mode!r}: it scatters over "
            "physical_to_logical_map, whose attention slots are -1 "
            "(--enable-eplb sets this mode implicitly)."
        )
    if cfg.ep_dispatch_algorithm == "lp":
        raise ValueError(
            "AFD refuses --ep-dispatch-algorithm lp: its solver bincounts "
            "physical_to_logical_map, whose attention slots are -1."
        )
    if cfg.moe_a2a_backend not in ALLOWED_MOE_A2A_BACKENDS:
        raise ValueError(
            f"AFD requires a deepep-class --moe-a2a-backend (got "
            f"{cfg.moe_a2a_backend!r}): any other backend takes the model's "
            "non-a2a forward path, which never applies the logical->physical remap."
        )

    return result


def install():
    HookRegistry.register(
        "sglang.srt.arg_groups.validation_hook.check_server_args",
        _after_check_server_args,
        HookType.AFTER,
    )
