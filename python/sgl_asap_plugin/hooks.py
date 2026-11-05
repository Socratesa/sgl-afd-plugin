"""Every SGLang target the plugin hooks, per role.

A hook body lives in the module of this package at the path of the SGLang module it
hooks: ``sglang.srt.X.f`` is hooked by ``sgl_asap_plugin.srt.X.f``. Every process
gets the base hooks, which add the ``--asap-*`` flags. The launcher registers the
role's hooks once it has parsed them, before ServerArgs resolution; the processes it
starts register them from the exported ``ASAP_*`` environment. SGLang's plugin
loader only logs a hook that fails to apply, so the launcher checks that every hook
of the role was applied.
"""

from __future__ import annotations

from typing import List

from sglang.srt.plugins.hook_registry import HookRegistry, HookSource, HookType

from sgl_asap_plugin.srt.arg_groups.fields.asap import Asap

# What load_plugins() records for register(). The role hooks are registered later,
# from the ServerArgs hook, where SGLang would log their source as "unknown".
_SOURCE = HookSource(plugin_name="sgl_asap_plugin", dist_name="sgl-asap-plugin")
_WEIGHT_UTILS = "sglang.srt.model_loader.weight_utils."

# Targets registered in this process, in registration order.
_targets: List[str] = []
_resolution_hook_registered = False


def _register(target: str, hook, hook_type: HookType) -> None:
    if target not in _targets:
        HookRegistry.register(target, hook, hook_type, source=_SOURCE)
        _targets.append(target)


def register_base_hooks() -> None:
    from sgl_asap_plugin.srt import server_args

    _register(
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        server_args.add_cli_args,
        HookType.AFTER,
    )
    _register(
        "sglang.srt.server_args.ServerArgs.from_cli_args",
        server_args.from_cli_args,
        HookType.BEFORE,
    )


def register_role_hooks(args: Asap) -> None:
    global _resolution_hook_registered
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    from sgl_asap_plugin.srt.arg_groups import asap_hook
    from sgl_asap_plugin.srt.model_loader import weight_utils
    from sgl_asap_plugin.srt.server_args import is_afd_attn

    if not _resolution_hook_registered:
        register_resolution_hook("handle_cuda_graph_config")(asap_hook.handle_asap)
        _resolution_hook_registered = True
    _register(
        "sglang.srt.arg_groups.validation_hook.check_server_args",
        asap_hook.check_asap_server_args,
        HookType.BEFORE,
    )
    from sgl_asap_plugin.srt.utils import network

    _register(
        "sglang.srt.utils.network.get_free_port_below_ephemeral",
        network.get_free_port_below_ephemeral,
        HookType.REPLACE,
    )
    if args.asap_role == "attn":
        from sgl_asap_plugin.srt.model_executor import model_runner

        _register(
            "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
            model_runner.load_model,
            HookType.AFTER,
        )
        if not is_afd_attn(args):
            return
        from sgl_asap_plugin.srt.batch_overlap import operations_strategy
        from sgl_asap_plugin.srt.layers.moe.ep_moe import layer
        from sgl_asap_plugin.srt.managers import scheduler

        _register(
            "sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class",
            layer.get_moe_impl_class,
            HookType.REPLACE,
        )
        # Length-aware batching (--asap-min-batch-tokens); inert without the flag.
        _register(
            "sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill",
            scheduler.get_new_batch_prefill,
            HookType.AROUND,
        )
        # Two-batch overlap (--enable-two-batch-overlap); inert without the flag.
        _register(
            "sglang.srt.arg_groups.validation_hook.check_two_batch_overlap",
            asap_hook.check_two_batch_overlap,
            HookType.REPLACE,
        )
        _register(
            "sglang.srt.model_executor.model_runner.ModelRunner."
            "_prepare_eager_forward_batch",
            model_runner._prepare_eager_forward_batch,
            HookType.AFTER,
        )
        _register(
            "sglang.srt.batch_overlap.operations_strategy."
            "_compute_moe_deepseek_v4_layer_operations_strategy_tbo",
            operations_strategy._compute_moe_deepseek_v4_layer_operations_strategy_tbo,
            HookType.AROUND,
        )
        weight_utils.skip_routed_experts()
    else:
        from sgl_asap_plugin.srt.entrypoints import http_server
        from sgl_asap_plugin.srt.managers import scheduler

        _register(
            "sglang.srt.entrypoints.http_server.launch_server",
            http_server.launch_server,
            HookType.REPLACE,
        )
        _register(
            "sglang.srt.managers.scheduler.Scheduler",
            scheduler.FfnScheduler,
            HookType.REPLACE,
        )
    _register(
        _WEIGHT_UTILS + "buffered_multi_thread_safetensors_weights_iterator",
        weight_utils.buffered_multi_thread_safetensors_weights_iterator,
        HookType.AROUND,
    )
    _register(
        _WEIGHT_UTILS + "safetensors_weights_iterator",
        weight_utils.safetensors_weights_iterator,
        HookType.AROUND,
    )


def hooks_applied() -> str:
    """'applied/registered' hooks of this process, e.g. '7/7'."""
    applied = sum(target in HookRegistry._patched for target in _targets)
    return f"{applied}/{len(_targets)}"


def check_hooks_applied() -> None:
    missing = [t for t in _targets if t not in HookRegistry._patched]
    if missing:
        raise RuntimeError(
            f"ASAP hooks were not applied (see 'Failed to apply hooks' above): {missing}"
        )
