#!/usr/bin/env python
"""M0 acceptance: one command, one PASS/FAIL table, one JSON report.

    python scripts/m0_acceptance.py --gpus 0,1 --control-gpus 2,3

Needs the venv that has sglang and this plugin installed (uv pip install -e .). The GPU
checks start their own sglang servers (own process groups) and only ever signal those.
"""

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731"
MIN_TESTS = 43


class Report:
    def __init__(self):
        self.rows = []

    def add(self, cid, title, measured, criterion, ok):
        status = "PASS" if ok is True else "SKIP" if ok is None else "FAIL"
        self.rows.append({"id": cid, "title": title, "measured": str(measured), "criterion": criterion,
                          "status": status})
        print(f"[{status}] {cid} {title}: {measured}   (criterion: {criterion})", flush=True)

    def ok(self):
        return all(r["status"] != "FAIL" for r in self.rows)

    def save(self, out):
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump(self.rows, f, indent=2, ensure_ascii=False)
        lines = ["| ID | Check | Measured | Criterion | Result |", "|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['title']} | {r['measured']} | {r['criterion']} | {r['status']} |"
                  for r in self.rows]
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write("\n".join(lines) + "\n")


def run_unit_tests(rep, out):
    paths = [os.path.join(REPO, "python"), os.path.join(REPO, "tests")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    env.pop("ASAP_ROLE", None)
    start = time.time()
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    with open(os.path.join(out, "unit_tests.log"), "w") as f:
        f.write(p.stderr)
    ran = re.search(r"Ran (\d+) tests?", p.stderr)
    n = int(ran.group(1)) if ran else 0
    failed = len(re.findall(r"\.\.\. (FAIL|ERROR)$", p.stderr, re.M))
    skipped = len(re.findall(r"\.\.\. skipped", p.stderr, re.M))
    rep.add("U1", "unit + fresh-process hook tests",
            f"{n} run, {failed} failed, {skipped} skipped, {time.time() - start:.0f}s",
            f">= {MIN_TESTS} run, 0 failed, 0 skipped",
            p.returncode == 0 and n >= MIN_TESTS and failed == 0 and skipped == 0)


def run_install_checks(rep):
    found = [ep.value for ep in importlib.metadata.entry_points(group="sglang.srt.plugins")
             if ep.name == "sgl_asap_plugin"]
    rep.add("U2", "entry point discoverable by sglang", found or "missing", "sgl_asap_plugin:register",
            found == ["sgl_asap_plugin:register"])
    code = ("import sys,time; t=time.perf_counter(); import sgl_asap_plugin; "
            "print(time.perf_counter()-t, 'torch' in sys.modules, 'sglang' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()
    dt, torch_in, sglang_in = float(out[0]), out[1], out[2]
    rep.add("U3", "import stays cheap", f"{dt * 1000:.0f} ms, torch={torch_in}, sglang={sglang_in}",
            "< 500 ms, torch=False, sglang=False", dt < 0.5 and torch_in == "False" and sglang_in == "False")


def expected_hook_count():
    code = "from sgl_asap_plugin import hooks; print(len(hooks.all_specs()))"
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    return int(out.stdout.split()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpus", default="0,1", help="GPUs for the A server (D=2, T=1)")
    ap.add_argument("--control-gpus", default="2,3", help="GPUs for the attention-DP control (tp=2)")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--out", default=os.path.join(REPO, "m0_report"))
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rep = Report()
    if not args.skip_unit:
        run_unit_tests(rep, args.out)
    run_install_checks(rep)
    if not args.skip_gpu:
        from accept_server_checks import run_attn_server_checks, run_control

        run_attn_server_checks(rep, args, expected_hook_count())
        if not args.skip_control:
            run_control(rep, args)
    rep.save(args.out)
    print(f"\nM0 acceptance: {'PASS' if rep.ok() else 'FAIL'}  (report: {args.out}/report.md)")
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
