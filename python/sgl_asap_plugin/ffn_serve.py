"""Entry point of the F (expert) role: ``python -m sgl_asap_plugin.ffn_serve``.

It takes the same flags as ``sglang serve`` (``--model-path``, ``--tp-size E --ep-size E``,
``--quantization`` ...) plus ``--asap-*``, resolves them through sglang's own pipeline,
validates the deployment and computes the static layout (expert partition, arena sizes).
Then it starts one worker process per F rank (``ffn/worker.py``): each builds and loads its
share of the routed experts, listens on ``--asap-rdv-addr`` port + rank, and serves A's messages.
GPUs are ``--base-gpu-id + rank * --gpu-id-step``; the launcher prints ``[ASAP] ffn ready`` once
every rank is loaded, warmed up and listening.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from sgl_asap_plugin import config, layout, validation
from sgl_asap_plugin.errors import AsapConfigError


def build_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(prog="python -m sgl_asap_plugin.ffn_serve")
    ServerArgs.add_cli_args(parser)
    config.add_cli_args(parser)
    parser.add_argument("--asap-dry-run", action="store_true",
                        help="Validate, print the layout and exit without starting the worker.")
    return parser


def plan(ns) -> dict:
    """Resolve + validate the launch and return the static layout summary."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.server_args import ServerArgs

    ns.asap_role = ns.asap_role or "ffn"
    asap = config.AsapConfig.from_namespace(ns)
    if asap.role != "ffn":
        raise AsapConfigError("[ASAP] ffn_serve only runs the ffn role")
    view = resolving_view(ServerArgs.from_cli_args(ns))
    validation.check_ffn(view, asap)
    shape = layout.read_moe_shape(view.model_path, view.json_model_override_args)
    part = layout.partition_experts(shape.n_experts, view.ep_size)
    arena = layout.arena_plan(asap.num_replicas, asap.replica_tp, view.ep_size, asap.slots, asap.n_max,
                              shape.hidden, shape.topk)
    return {
        "role": "ffn",
        "ep_size": view.ep_size,
        "n_experts": shape.n_experts,
        "padded_experts": part.padded,
        "padding_experts": part.padding,
        "experts_per_rank": part.per_rank,
        "expert_ranges": [list(r) for r in part.ranges()],
        "moe": {"hidden": shape.hidden, "topk": shape.topk, "layers": shape.layers},
        "asap": {"rdv_addr": asap.rdv_addr, "num_replicas": asap.num_replicas, "replica_tp": asap.replica_tp,
                 "n_max": asap.n_max, "slots": asap.slots},
        "arena": {"bytes_per_token": arena.bytes_per_token, "region_bytes": arena.region_bytes,
                  "f_recv_arena_bytes": arena.f_recv_arena_bytes,
                  "a_result_arena_bytes": arena.a_result_arena_bytes},
    }


def serve(ns, summary: dict) -> int:
    """Start the F ranks, wait until they are all ready, then supervise them until SIGTERM / SIGINT."""
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    import time

    from sgl_asap_plugin.ffn.worker import worker_main

    ctx = mp.get_context("spawn")
    ranks = summary["ep_size"]
    base, step = getattr(ns, "base_gpu_id", 0) or 0, getattr(ns, "gpu_id_step", 1) or 1
    run_dir = tempfile.mkdtemp(prefix="asap_ffn_", dir="/dev/shm")
    stop = ctx.Event()
    quit_requested = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: quit_requested.set())
    workers = []
    for f in range(ranks):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker_main, args=(ns, f, base + f * step, run_dir, child, stop),
                           name=f"asap-ffn-{f}")
        proc.start()
        workers.append((proc, parent))
    status, ready = 0, {}
    deadline = time.time() + float(os.environ.get("ASAP_FFN_START_TIMEOUT_S", "3600"))
    while len(ready) < ranks and not quit_requested.is_set():
        for f_rank, (proc, conn) in enumerate(workers):
            try:
                msg = conn.recv() if conn.poll(0.2) else None
            except (EOFError, OSError):  # the worker died before it could report
                print(f"[ASAP] ffn rank {f_rank} exited unexpectedly (exit code {proc.exitcode})", file=sys.stderr,
                      flush=True)
                status = 1
                continue
            if msg is not None:
                kind, f, *rest = msg
                if kind == "ready":
                    ready[f] = rest
                    print(f"[ASAP] ffn rank {f} ready: {json.dumps(rest[1], sort_keys=True)}", flush=True)
                elif kind == "error":
                    print(f"[ASAP] ffn rank {f} failed:\n{rest[0]}", file=sys.stderr, flush=True)
                    status = 1
        if status or time.time() > deadline or any(not p.is_alive() and p.exitcode for p, _ in workers):
            status = status or 1
            break
    if status == 0 and len(ready) == ranks:
        print(f"[ASAP] ffn ready: ranks={ranks} rdv={summary['asap']['rdv_addr']}+rank "
              f"experts_per_rank={summary['experts_per_rank']} layers={summary['moe']['layers']}", flush=True)
        while not quit_requested.is_set():
            if any(not p.is_alive() for p, _ in workers):
                print("[ASAP] an ffn rank exited; shutting the service down", file=sys.stderr, flush=True)
                status = 1
                break
            time.sleep(0.5)
    stop.set()
    for proc, conn in workers:
        proc.join(timeout=30)
        try:
            while conn.poll():
                kind, f, *rest = conn.recv()
                if kind == "stats":
                    print(f"[ASAP] ffn rank {f} stats {json.dumps(rest[0], sort_keys=True)} "
                          f"layer_calls={sum(rest[1])}", flush=True)
        except (EOFError, OSError):
            pass
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    shutil.rmtree(run_dir, ignore_errors=True)
    print("[ASAP] ffn stopped", flush=True)
    return status


def main(argv=None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        summary = plan(ns)
    except AsapConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    print("[ASAP] ffn layout " + json.dumps(summary, sort_keys=True), flush=True)
    if ns.asap_dry_run:
        return 0
    return serve(ns, summary)


if __name__ == "__main__":
    sys.exit(main())
