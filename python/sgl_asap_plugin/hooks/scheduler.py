"""Role-tier hook: record the ASAP identity of every scheduler process."""

from sglang.srt.plugins.hook_registry import HookType

from sgl_asap_plugin.hooks import ROLE, HookSpec


def _after_scheduler_init(result, self, server_args, port_args):
    """Scheduler.__init__ has returned: model loaded, distributed world up, event loop not started."""
    from sgl_asap_plugin import runtime

    runtime.init_scheduler_runtime(self)
    return None


SPECS = [
    HookSpec("sglang.srt.managers.scheduler.Scheduler.__init__", _after_scheduler_init,
             HookType.AFTER, ("self", "server_args", "port_args"), ROLE),
]
