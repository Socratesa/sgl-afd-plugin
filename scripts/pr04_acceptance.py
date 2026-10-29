#!/usr/bin/env python
"""PR-04 acceptance: the attention role in AFD mode, driven against a loopback F that answers zeros.

    python scripts/pr04_acceptance.py --gpus 2,3

GPU 1 of the list hosts the A rank, GPU 2 the loopback F (both processes see both GPUs, which the
CUDA IPC mapping needs). Real DeepSeek-V4-Flash weights are loaded; what is checked is the A side
only: hooks, the weights an A rank keeps, and that every MoE layer of a real forward pass goes
through the transport. Numerics against real experts are PR-05/PR-06.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accept_lib import ARGS_OK_RE, READY_RE, LogServer, Server, generate, server_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731"
UNIT_MODULES = ["test_afd_forward", "test_afd_pieces", "test_hooks", "test_config", "test_validation", "test_docs"]
LOAD_RE = re.compile(r"Load weight end\. elapsed=([\d.]+) s.*?mem usage=([\d.]+) GB")
F_STATS_RE = re.compile(r"\[loopback-f\] stats \('stats', 0, (\{.*?\}), (\[.*?\])\)")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, cid, title, measured, criterion, ok):
        self.rows.append({"id": cid, "title": title, "measured": str(measured), "criterion": criterion,
                          "status": "PASS" if ok else "FAIL"})
        print(f"[{self.rows[-1]['status']}] {cid} {title}: {measured}   (criterion: {criterion})", flush=True)

    def ok(self):
        return all(r["status"] == "PASS" for r in self.rows)


def run_units(rep):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "python"), os.path.join(REPO, "tests")]))
    env.pop("ASAP_ROLE", None)
    start = time.time()
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", *UNIT_MODULES, "-v"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    n = int(m.group(1)) if (m := re.search(r"Ran (\d+) tests?", p.stderr)) else 0
    bad = len(re.findall(r"\.\.\. (FAIL|ERROR|skipped)", p.stderr))
    rep.add("U1", "unit tests of the A side", f"{n} run, {bad} failed or skipped, {time.time() - start:.0f}s",
            ">= 44 run, 0 failed, 0 skipped", p.returncode == 0 and n >= 44 and bad == 0)


def afd_run(rep, tag, layers, gpus, args, mem_limit_gb, prompts):
    visible = gpus
    port, rdv = args.port, args.rdv_port
    out = os.path.join(args.out, tag)
    f_env = server_env(visible, real_weights=True)
    f_srv = LogServer(f"{tag}_loopback_f", [sys.executable, "-u", os.path.join(REPO, "tests", "loopback_f.py"),
                                            "--ranks", "1", "--gpu-base", "1", "--port", str(rdv), "--a-ranks", "1",
                                            "--layers", str(layers), "--n-max", "8192"], f_env, out,
                      r"\[loopback-f\] ready")
    f_env["PYTHONPATH"] = os.path.join(REPO, "python")
    f_srv.start()
    if not f_srv.wait_ready(300):
        f_srv.stop()
        rep.add(f"{tag}.0", "loopback F starts", f_srv.log_text()[-300:], "ready line", False)
        return
    argv = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--host", "127.0.0.1", "--port",
            str(port), "--dp-size", "1", "--tp-size", "1", "--base-gpu-id", "0", "--asap-role", "attn",
            "--asap-ffn-ranks", "1", "--asap-num-replicas", "1", "--asap-replica-tp", "1", "--asap-rdv-addr",
            f"127.0.0.1:{rdv}", "--asap-n-max", "8192", "--mem-fraction-static", "0.8", "--disable-cuda-graph",
            "--json-model-override-args", json.dumps({"num_hidden_layers": layers})]
    a_srv = Server(f"{tag}_attn", argv, server_env(visible, real_weights=True), out, port)
    a_srv.start()
    ready = a_srv.wait_ready(args.startup_timeout)
    log = a_srv.log_text()
    args_ok = ARGS_OK_RE.search(log)
    scheds = [m.groupdict() for m in READY_RE.finditer(log)]
    load = LOAD_RE.search(log)
    rep.add(f"{tag}.G0", f"A rank starts ({layers} layers, real weights)", f"ready={ready}",
            "/health 200 within the timeout", ready)
    rep.add(f"{tag}.G1", "AFD hooks applied", f"launcher {args_ok.groups() if args_ok else None}, scheduler "
            f"{[(s['ok'], s['total']) for s in scheds]}", "8/8 in the launcher and in the scheduler",
            bool(args_ok) and args_ok.group(2) == "8" and len(scheds) == 1 and (scheds[0]["ok"], scheds[0]["total"]) == ("8", "8"))
    mem = float(load.group(2)) if load else None
    rep.add(f"{tag}.G2", "weights an A rank keeps", f"{mem} GB in {load.group(1) if load else '?'} s",
            f"< {mem_limit_gb} GB (no routed expert is allocated or read)", mem is not None and mem < mem_limit_gb)
    results = [generate(port, text="hello world " * (n // 2), timeout=120) for n in prompts] if ready else []
    rep.add(f"{tag}.G3", "requests go through every MoE layer", f"prompts {prompts}: "
            f"{[(ok, round(t, 2)) for ok, t in results]}", "all succeed", bool(results) and all(ok for ok, _ in results))
    a_srv.stop()
    time.sleep(2)
    f_srv.stop()
    stats = F_STATS_RE.search(f_srv.log_text())
    seen = json.loads(stats.group(2)) if stats else []
    st = json.loads(stats.group(1).replace("'", '"')) if stats else {}
    rep.add(f"{tag}.G4", "F saw all layers, nothing lost", f"layers {len(seen)}/{layers}, jobs {st.get('jobs')}, "
            f"lost {st.get('lost_messages', 0)}", "every MoE layer seen, 0 lost messages",
            seen == list(range(layers)) and st.get("lost_messages", 0) == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--port", type=int, default=31030)
    ap.add_argument("--rdv-port", type=int, default=9640)
    ap.add_argument("--out", default="/tmp/asap_reports/pr04", help="logs and report (not on the shared volume)")
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--skip-units", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rep = Report()
    if not args.skip_units:
        run_units(rep)
    afd_run(rep, "tiny", 4, args.gpus, args, 6.0, [10, 500, 3000])
    afd_run(rep, "full", 43, args.gpus, args, 14.0, [10, 500, 3000, 6000])
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep.rows, f, indent=2)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("| ID | Check | Measured | Criterion | Result |\n|---|---|---|---|---|\n")
        f.writelines(f"| {r['id']} | {r['title']} | {r['measured']} | {r['criterion']} | {r['status']} |\n"
                     for r in rep.rows)
    print(f"\nPR-04 acceptance: {'PASS' if rep.ok() else 'FAIL'}  (report: {args.out}/report.md)")
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
