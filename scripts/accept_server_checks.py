"""GPU checks of the M0 acceptance run: an A server with D=2 plain-DP replicas, plus a stock
attention-DP control that shows the independence check is able to fail."""

import os
import signal
import statistics
import sys
import time

from accept_lib import ARGS_OK_RE, TINY_OVERRIDE, Server, generate, server_env, start_server


def launch_argv(args, port, *flags):
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model, "--host", "127.0.0.1",
            "--port", str(port), "--load-format", "dummy", "--json-model-override-args", TINY_OVERRIDE,
            "--mem-fraction-static", "0.55", "--disable-cuda-graph", *flags]


def run_attn_server_checks(rep, args, hooks_expected):
    flags = ("--asap-role", "attn", "--dp-size", "2", "--tp-size", "1")
    make = lambda: Server("attn_dp2", launch_argv(args, args.port, *flags), server_env(args.gpus), args.out,
                          args.port)
    start = time.time()
    srv, attempts, ready = start_server(make, args.startup_timeout)
    try:
        rep.add("G0", "A server (D=2, T=1, DSV4 4-layer dummy) comes up",
                f"{time.time() - start:.0f}s, {attempts} attempt(s)" if ready else f"not ready, see {srv.log_path}",
                f"/health 200 within {args.startup_timeout}s (retry only on the EADDRINUSE port race)", ready)
        if not ready:
            return
        lines, args_ok = srv.ready_lines(), ARGS_OK_RE.findall(srv.log_text())
        hooks_seen = sorted({l["ok"] + "/" + l["total"] for l in lines})
        rep.add("G1", "hooks are live in the launcher and in every scheduler process",
                f"launcher={len(args_ok)} line(s), schedulers={len(lines)} line(s), hooks={hooks_seen}",
                f"launcher >= 1 line, 2 scheduler lines, all hooks={hooks_expected}/{hooks_expected}",
                len(lines) == 2 and bool(args_ok)
                and all(l["ok"] == l["total"] == str(hooks_expected) for l in lines)
                and all(int(a) == int(b) == hooks_expected for _, a, b in args_ok))
        rep.add("G2", "pure DP: every replica is its own distributed world",
                [(f"dp{l['dp']}", f"world_size={l['world']}", f"tp_size={l['tp_size']}", f"gpu={l['gpu']}")
                 for l in lines],
                "dp ranks {0,1}; world_size == tp_size == 1 (not D*T = 2)",
                {l["dp"] for l in lines} == {"0", "1"} and all(l["world"] == l["tp_size"] == "1" for l in lines))
        served = [generate(args.port, rank=r) for r in (0, 1) for _ in range(3)]
        served += [generate(args.port) for _ in range(4)]
        median = statistics.median(t for _, t in served)
        rep.add("G3", "replicas serve requests (3 per replica + 4 round-robin)",
                f"{sum(ok for ok, _ in served)}/{len(served)} ok, median {median * 1000:.0f} ms", "all ok",
                all(ok for ok, _ in served))
        check_independence(rep, args.port, {int(l["dp"]): int(l["pid"]) for l in lines})
    finally:
        srv.stop()


def check_independence(rep, port, pids):
    os.kill(pids[0], signal.SIGSTOP)
    try:
        live = [generate(port, rank=1, timeout=30) for _ in range(5)]
        stuck_ok, stuck_t = generate(port, rank=0, timeout=5)
    finally:
        os.kill(pids[0], signal.SIGCONT)
    resumed_ok, _ = generate(port, rank=0, timeout=60)
    slowest = max(t for _, t in live)
    stuck = "completed" if stuck_ok else f"no reply after {stuck_t:.0f}s"
    rep.add("G4", "replica independence: SIGSTOP replica 0, replica 1 keeps serving",
            f"replica 1: {sum(ok for ok, _ in live)}/5 ok, slowest {slowest:.2f}s; replica 0 request while "
            f"stopped: {stuck}; after SIGCONT: {'ok' if resumed_ok else 'failed'}",
            "replica 1 5/5 ok, slowest <= 10 s; stopped replica gives no reply (experiment is valid); "
            "replica 0 serves again after SIGCONT",
            all(ok for ok, _ in live) and slowest <= 10 and not stuck_ok and resumed_ok)


def run_control(rep, args):
    """Same experiment on stock attention-DP: the coupling the plugin removes must show up here."""
    port = args.port + 100
    flags = ("--tp-size", "2", "--dp-size", "2", "--enable-dp-attention")
    make = lambda: Server("control_attn_dp", launch_argv(args, port, *flags), server_env(args.control_gpus),
                          args.out, port)
    srv, attempts, ready = start_server(make, args.startup_timeout)
    try:
        if not ready:
            rep.add("N1", "control: stock attention-DP, SIGSTOP one rank",
                    f"control server did not start, see {srv.log_path}", "neither rank replies", None)
            return
        procs = srv.scheduler_procs()
        before = [generate(port, rank=r, timeout=60)[0] for r in (0, 1)]
        procs[0].send_signal(signal.SIGSTOP)
        try:
            after = [generate(port, rank=r, timeout=8)[0] for r in (0, 1)]
        finally:
            procs[0].send_signal(signal.SIGCONT)
        rep.add("N1", "control: stock attention-DP, SIGSTOP one rank (the check G4 must be able to fail)",
                f"before: {before}; after stopping one of {len(procs)} schedulers: {after}",
                "served before; neither rank replies afterwards (ranks are lock-stepped)",
                all(before) and not any(after))
    finally:
        srv.stop()
