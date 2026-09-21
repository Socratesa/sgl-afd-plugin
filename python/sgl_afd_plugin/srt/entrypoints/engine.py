"""AFD overrides for SGLang's Engine."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sgl_afd_plugin.srt.utils.common import override_mod


def _around_launch_scheduler_processes(original_fn, cls, server_args, port_args,
                                       run_scheduler_process_func, **kwargs):
    from sgl_afd_plugin.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process
    )

    with override_mod(
        "sglang.srt.entrypoints.engine.run_data_parallel_controller_process",
        run_data_parallel_controller_process,
    ):
        result = original_fn(cls, server_args, port_args, run_scheduler_process_func, **kwargs)

    return result


def install():
    HookRegistry.register(
        "sglang.srt.entrypoints.engine.Engine._launch_scheduler_processes",
        _around_launch_scheduler_processes,
        HookType.AROUND
    )
