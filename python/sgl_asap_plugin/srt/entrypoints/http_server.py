"""``launch_server`` of the FFN role.

``sglang serve --asap-role ffn --tp-size E --ep-size E`` starts what a node with
``--node-rank >= 1`` runs: E scheduler processes and a /health endpoint, no tokenizer
or HTTP API, since the FFN ranks receive no requests. Each rank is an
``FfnScheduler`` (``srt/managers/scheduler.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import tempfile

# Set by the launcher: the directory of the ranks' doorbell files, which the
# attention ranks map.
ENV_RUN_DIR = "ASAP_FFN_RUN_DIR"


def launch_server(server_args, *args, **kwargs) -> None:
    """REPLACE of ``http_server.launch_server`` on the FFN role: the steps of
    ``Engine._launch_subprocesses`` up to the scheduler launch, then what it does on a
    node with ``--node-rank >= 1``."""
    from sglang.srt.entrypoints.engine import Engine, _set_envs_and_config
    from sglang.srt.entrypoints.server_utils import launch_dummy_health_check_server
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import get_observability, get_serving, publish
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import configure_logger

    from sgl_asap_plugin.srt.server_args import get_asap_args

    configure_logger(server_args)
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    load_plugins()
    server_args.check_server_args()
    publish(server_args, role="tokenizer")
    port_args = PortArgs.init_new(server_args)

    run_dir = tempfile.mkdtemp(prefix="asap_ffn_", dir="/dev/shm")
    os.environ[ENV_RUN_DIR] = run_dir
    on_child_failure = signal.getsignal(signal.SIGQUIT)

    def remove_run_dir_and_exit(signum, frame):
        # SGLang's handler kills the whole tree, so the finally below never runs.
        shutil.rmtree(run_dir, ignore_errors=True)
        if callable(on_child_failure):
            on_child_failure(signum, frame)

    signal.signal(signal.SIGQUIT, remove_run_dir_and_exit)
    try:
        result, procs = Engine._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process
        )

        def stop(signum, frame):
            for proc in procs:
                if proc.is_alive():
                    os.kill(proc.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        result.wait_for_ready()
        for info in result.scheduler_infos:
            asap = info["asap"]
            print(
                f"[ASAP] ffn rank {asap['f']} ready: {json.dumps(asap, sort_keys=True)}",
                flush=True,
            )
        print(
            f"[ASAP] ffn ready: ranks={len(procs)} "
            f"bootstrap={get_asap_args().asap_bootstrap_addr}+rank",
            flush=True,
        )
        launch_dummy_health_check_server(
            get_serving().host, get_serving().port, get_observability().enable_metrics
        )
        for proc in procs:
            proc.join()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    failed = {proc.pid: proc.exitcode for proc in procs if proc.exitcode}
    print(
        "[ASAP] ffn stopped" + (f", exit codes {failed}" if failed else ""), flush=True
    )
    if failed:
        raise SystemExit(1)
