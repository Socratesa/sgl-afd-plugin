#!/usr/bin/env python
"""Acceptance runs of the ASAP plugin: one command per stage, a PASS/FAIL table, report.json and report.md.

    python scripts/acceptance.py m0   --gpus 0,1 --control-gpus 2,3   # skeleton: hooks, rules, plain-DP topology
    python scripts/acceptance.py pr03 --gpus 0,1,2,3                  # transport library
    python scripts/acceptance.py pr04 --gpus 0,1,2                    # A-side MoE forward vs synthetic F ranks
    python scripts/acceptance.py pr05 --gpus 0,1,2,3                  # F service vs a torch reference
    python scripts/acceptance.py e2e  --a-gpus 0,1 --f-gpus 4,5,6,7   # full model: PR-04/05 bring-up, PR-06, PR-07

Tests assert their own gates; this script checks complete stage coverage and structured suite results.
Exit codes: PASS=0, FAIL=1, PARTIAL=2. Logs and reports go to --out (default /tmp/asap_reports/<stage>).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import servers as sv  # noqa: E402
from acceptance_load import RequestSample, assess_load  # noqa: E402
from acceptance_metrics import compare  # noqa: E402
from acceptance_report import Report  # noqa: E402
from acceptance_runner import SuiteResult, run_suite  # noqa: E402


def run_tests(names: list[str], out: str, log_name: str, gpus: str | None = None,
              timeout_s: float = 3600) -> SuiteResult:
    """Preserve complete unittest outcomes, child exit status and a bounded deadline."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "python"), os.path.join(REPO, "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("ASAP_ROLE", None)
    if gpus:
        env["ASAP_TEST_GPUS"] = gpus
    return run_suite(names, cwd=os.path.join(REPO, "tests"), env=env,
                     log_path=os.path.join(out, log_name), timeout_s=timeout_s)


def add_mapped(rep, checks, suite: SuiteResult, measured=None, min_tests=None):
    """One row per (id, title, tests, criterion); then one row that every test ran, none skipped, all mapped."""
    measured = measured or {}
    status = suite.statuses
    resolved = set()
    for cid, title, tests, criterion in checks:
        results = []
        for name in tests:
            matches = [test_id for test_id in status if test_id == name or test_id.endswith("." + name)]
            if len(matches) == 1:
                resolved.add(matches[0])
                results.append(status[matches[0]])
            else:
                results.append("not run" if not matches else f"ambiguous test name: {matches}")
        ok = all(r == "ok" for r in results)
        text = measured.get(cid) or (f"{len(tests)} test(s) ok" if ok else
                                     ", ".join(f"{t}: {r}" for t, r in zip(tests, results) if r != "ok"))
        rep.add(cid, title, text, criterion, ok)
    if min_tests is not None:
        extra = sorted(set(status) - resolved)
        rep.add("U", "suite completion and mapping",
                f"{len(status)} outcomes in {suite.elapsed_s:.0f}s, exit {suite.returncode}, "
                f"unmapped {extra or 'none'}, problems {suite.problems}",
                f">= {min_tests} tests, each uniquely mapped, all pass, exit 0",
                suite.ok and len(status) >= min_tests and not extra)


# ---- M0: plugin skeleton, deployment rules, plain-DP topology ----
def stage_m0(rep, args):
    suite = run_tests(["test_cpu"], args.out, "unit_tests.log", timeout_s=args.test_timeout)
    status, secs = suite.statuses, suite.elapsed_s
    failed = [t for t, s in status.items() if s != "ok"]
    rep.add("U1", "CPU tests (config, rules, layout, checkpoint selection, hook activation in fresh processes)",
            f"{len(status)} run, {len(failed)} not ok {failed or ''}, {secs:.0f}s", ">= 28 run, all ok, 0 skipped",
            suite.ok and len(status) >= 28 and not failed)
    found = [ep.value for ep in importlib.metadata.entry_points(group="sglang.srt.plugins") if ep.name == "sgl_asap_plugin"]
    rep.add("U2", "entry point discoverable by sglang", found or "missing", "sgl_asap_plugin:register",
            found == ["sgl_asap_plugin:register"])
    code = ("import sys,time; t=time.perf_counter(); import sgl_asap_plugin; "
            "print(time.perf_counter()-t, 'torch' in sys.modules, 'sglang' in sys.modules)")
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=args.test_timeout)
    if child.returncode != 0:
        raise RuntimeError(f"package import probe exited {child.returncode}: {child.stderr[:1000]}")
    dt, torch_in, sglang_in = child.stdout.split()
    rep.add("U3", "importing the package stays cheap", f"{float(dt) * 1000:.0f} ms, torch={torch_in}, sglang={sglang_in}",
            "< 500 ms, torch=False, sglang=False", float(dt) < 0.5 and torch_in == sglang_in == "False")
    if args.skip_gpu:
        for cid in ("G0", "G1", "G2", "G3", "G4", "N1"):
            rep.add(cid, "GPU check excluded", "--skip-gpu", "required for full m0 acceptance", None)
        return
    hooks = len([s for s in __import__("sgl_asap_plugin.hooks", fromlist=["SPECS"]).SPECS if s.tier in ("base", "role")])
    tiny = ("--load-format", "dummy", "--json-model-override-args", sv.TINY_OVERRIDE, "--mem-fraction-static", "0.55",
            "--disable-cuda-graph")
    argv = sv.launch_server_argv(args.model, args.port, "--asap-role", "attn", "--dp-size", 2, "--tp-size", 1, *tiny)
    make = lambda: sv.Server("attn_dp2", argv, sv.server_env(args.gpus, SGLANG_DSV4_FP4_EXPERTS="0"), args.out, args.port)
    start = time.time()
    srv, attempts, ready = sv.start_server(make, args.startup_timeout)
    try:
        rep.add("G0", "A server (D=2, T=1, DSV4 4-layer dummy) comes up",
                f"{time.time() - start:.0f}s, {attempts} attempt(s)" if ready else f"not ready, see {srv.log_path}",
                f"/health 200 within {args.startup_timeout}s (retry only on the EADDRINUSE port race)", ready)
        if not ready:
            return
        ranks, args_ok = srv.attn_ranks(), sv.ARGS_OK_RE.findall(srv.log_text())
        rep.add("G1", "hooks are live in the launcher and in every scheduler",
                f"launcher {[a[2] + '/' + a[3] for a in args_ok]}, ranks {[r['ok'] + '/' + r['total'] for r in ranks]}",
                f"launcher and both ranks {hooks}/{hooks}", len(ranks) == 2 and bool(args_ok)
                and all(r["ok"] == r["total"] == str(hooks) for r in ranks) and all(a[2] == a[3] == str(hooks) for a in args_ok))
        rep.add("G2", "plain DP: every replica is its own distributed world",
                [(f"replica{r['replica']}", f"world_size={r['world']}", f"gpu={r['gpu']}") for r in ranks],
                "replicas {0,1}, world_size 1 each (not D*T = 2)",
                {r["replica"] for r in ranks} == {"0", "1"} and all(r["world"] == "1" for r in ranks))
        served = [sv.generate(args.port, rank=r, timeout=60)[:2] for r in (0, 1) for _ in range(3)]
        served += [sv.generate(args.port, timeout=60)[:2] for _ in range(4)]
        rep.add("G3", "replicas serve requests (3 per replica + 4 unrouted)", f"{sum(ok for ok, _ in served)}/{len(served)} ok",
                "all ok", all(ok for ok, _ in served))
        pids = {int(r["replica"]): int(r["pid"]) for r in ranks}
        os.kill(pids[0], signal.SIGSTOP)  # replica 1 must not notice
        try:
            live = [sv.generate(args.port, rank=1, timeout=30)[:2] for _ in range(5)]
            stuck_ok = sv.generate(args.port, rank=0, timeout=5)[0]
        finally:
            os.kill(pids[0], signal.SIGCONT)
        resumed = sv.generate(args.port, rank=0, timeout=60)[0]
        slowest = max(t for _, t in live)
        rep.add("G4", "replica independence: SIGSTOP replica 0, replica 1 keeps serving",
                f"replica 1 {sum(ok for ok, _ in live)}/5 ok, slowest {slowest:.2f}s; stopped replica answered: "
                f"{stuck_ok}; after SIGCONT ok: {resumed}", "5/5 ok within 10 s; no answer while stopped; serves again",
                all(ok for ok, _ in live) and slowest <= 10 and not stuck_ok and resumed)
    finally:
        srv.stop()
    if args.skip_control:
        rep.add("N1", "control excluded", "--skip-control", "required for full m0 acceptance", None)
        return
    port = args.port + 100  # control: stock attention DP is lock-stepped, so the same experiment must fail there
    argv = sv.launch_server_argv(args.model, port, "--tp-size", 2, "--dp-size", 2, "--enable-dp-attention", *tiny)
    make = lambda: sv.Server("control_attn_dp", argv, sv.server_env(args.control_gpus, SGLANG_DSV4_FP4_EXPERTS="0"),
                             args.out, port)
    srv, _, ready = sv.start_server(make, args.startup_timeout)
    try:
        if not ready:
            rep.add("N1", "control: stock attention-DP", f"did not start, see {srv.log_path}", "control must start", False)
            return
        procs = [p for p in srv.descendants() if "sglang::scheduler" in " ".join(p.cmdline())]
        before = [sv.generate(port, rank=r, timeout=60)[0] for r in (0, 1)]
        procs[0].send_signal(signal.SIGSTOP)
        try:
            after = [sv.generate(port, rank=r, timeout=8)[0] for r in (0, 1)]
        finally:
            procs[0].send_signal(signal.SIGCONT)
        rep.add("N1", "control: stock attention-DP, SIGSTOP one rank (G4 must be able to fail)",
                f"before {before}; after stopping one of {len(procs)} schedulers {after}",
                "served before; neither rank replies afterwards (lock-stepped)", all(before) and not any(after))
    finally:
        srv.stop()


# ---- PR-03: transport library ----
PR03_CHECKS = [
    ("C1", "wire format: doorbell fields, geometry rules, arena sizes",
     ["test_doorbell_round_trip_at_the_field_limits", "test_invalid_geometry_names_the_field",
      "test_sizes_at_the_design_point", "test_regions_are_aligned_and_flags_have_a_page_of_their_own"],
     "every field round-trips at its limits; bad fields and oversized n_max rejected; flags in their own page"),
    ("C2", "ping-pong, payload and layer id verified", ["test_one_a_one_f_1000_rounds"], "1000 rounds, 0 wrong, 0 lost"),
    ("C3", "three F ranks, partial sums add up", ["test_three_f_ranks_partial_sums_add_up"], "200 rounds, 0 wrong"),
    ("C4", "message sizes 8 KB to 64 MB", ["test_sizes_from_8kb_to_64mb"], "0 wrong"),
    ("C5", "sequence numbers wrap past 2^32", ["test_sequence_numbers_wrap_around_2_pow_32"], "0 wrong, 0 lost"),
    ("C6", "stress", ["test_stress_18k_rounds"], "18000 rounds, 0 wrong, 18000 jobs"),
    ("C7", "first come first served, three senders", ["test_first_come_first_served_with_three_senders"],
     "900 jobs, every sender served, > 100 replica switches, 0 wrong"),
    ("B1", "backpressure", ["test_second_send_into_a_busy_slot_blocks_and_nothing_is_lost"],
     "second send into a busy slot stays parked; both delivered in order; 0 lost"),
    ("B2", "control: no backpressure", ["test_control_without_backpressure_the_loss_is_detected"],
     "the overwrite is detected (lost_messages >= 1), so B1 can fail"),
    ("F1", "dead F rank", ["test_dead_f_times_out_instead_of_hanging"],
     "wait fails in 1.5-10 s (timeout 2 s) with ChannelPoisoned; later sends refused"),
    ("F2", "stale epoch and reconnect", ["test_stale_epoch_message_is_dropped_and_reconnect_recovers"],
     "stale message dropped and counted; reconnect works (20 rounds, 0 wrong)"),
    ("F3", "killed A rank", ["test_killed_a_does_not_disturb_f_or_the_other_a"],
     "other A 8000 rounds 0 wrong; a new incarnation reconnects; 0 lost"),
    ("F4", "handshake geometry mismatch", ["test_handshake_rejects_a_geometry_mismatch"], "fails fast, names the field"),
    ("F5", "cold work behind a parked wait", ["test_parked_wait_delays_cold_work_but_does_not_deadlock"],
     "cold work within release delay + 15 s; warm work < 0.5 s"),
    ("P1", "small-message round trip", ["test_small_message_round_trip"],
     "n=1 RTT p50 < 150 us with a spinning F; a napping F is > 40 us slower"),
    ("P2", "64 MB one way and full duplex", ["test_64mb_one_way_and_full_duplex"], "one way > 330 GB/s; overlapped > 650"),
    ("P3", "64 MB round trip", ["test_64mb_round_trip_bandwidth"], "aggregate > 150 GB/s"),
]


def pr03_metrics(stdout):
    out = {}
    m = re.search(r"RTT n=1 us, F spinning: \{'p50': ([\d.]+), 'p99': ([\d.]+)\}; F napping on every idle poll: "
                  r"\{'p50': ([\d.]+)", stdout)
    if m:
        out["P1"] = f"p50 {float(m.group(1)):.0f} us (p99 {float(m.group(2)):.0f}); napping F p50 {float(m.group(3)):.0f} us"
    m = re.search(r"64MB one way (\d+) GB/s; send and reply overlapped: (\d+) GB/s \((\d+) us per message\)", stdout)
    if m:
        out["P2"] = f"one way {m.group(1)} GB/s; overlapped {m.group(2)} GB/s ({m.group(3)} us per message)"
    m = re.search(r"64MB each way: RTT us \{'p50': ([\d.]+).*?aggregate (\d+) GB/s", stdout)
    if m:
        out["P3"] = f"RTT p50 {float(m.group(1)):.0f} us, aggregate {m.group(2)} GB/s"
    m = re.search(r"cold work behind a parked wait: ([\d.]+)s \(released after ([\d.]+)s\); warm work: ([\d.]+)s", stdout)
    if m:
        out["F5"] = f"cold {m.group(1)} s (release at {m.group(2)} s), warm {m.group(3)} s"
    return out


def stage_pr03(rep, args):
    names = ["test_cpu.TestLayout", "test_gpu.TestPingPong", "test_gpu.TestBackpressure", "test_gpu.TestFaults",
             "test_gpu.TestPerf"]
    suite = run_tests(names, args.out, "transport_tests.log", args.gpus, args.test_timeout)
    add_mapped(rep, PR03_CHECKS, suite, pr03_metrics(suite.stdout), min_tests=20)


# ---- PR-04 (unit part): the A-side MoE forward against synthetic F ranks ----
PR04_CHECKS = [
    ("A1", "moe_forward with one F rank: plain, hash layer, no shared expert, MoEOutput, empty batch, > n_max",
     ["test_one_f_rank"], "outputs bit-exact vs the synthetic expectation; messages = ceil(n / n_max); hash ids passed"),
    ("A2", "the same with two F ranks (partial sums added in fp32)", ["test_two_f_ranks_partial_sums"], "bit-exact"),
]


def stage_pr04(rep, args):
    suite = run_tests(["test_gpu.TestAttnForward"], args.out, "attn_tests.log", args.gpus, args.test_timeout)
    add_mapped(rep, PR04_CHECKS, suite, min_tests=2)


# ---- PR-05 (service part): the F service against a torch reference of the experts ----
PR05_CHECKS = [
    ("S1", "lifecycle: one rank, one layer", ["test_lifecycle_one_rank_one_layer"],
     "ready report: experts [0,256), MXFP4 FlashInfer method, 3.42 +- 0.2 GB; SIGTERM -> exit 0"),
    ("S2", "3 ranks (2 padding experts), 3 A clients, random layers in random order",
     ["test_three_ranks_match_the_reference_with_out_of_order_arrivals"],
     "relative error vs dequantized torch reference < 2e-2; >= 3 layers visited; every message served; experts-only memory"),
    ("S3", "handshake rejects an A model the service does not serve", ["test_model_mismatch_is_rejected_in_the_handshake"],
     "A fails with a message naming moe_layers"),
]


def stage_pr05(rep, args):
    suite = run_tests(["test_gpu.TestFfnService"], args.out, "ffn_tests.log", args.gpus, args.test_timeout)
    m = re.search(r"worst relative error vs the torch reference per A client: (\[.*?\]); layers in arrival order (\[.*?\])",
                  suite.stdout)
    measured = {"S2": f"worst rel err {m.group(1)}, arrival order {m.group(2)}"} if m else {}
    add_mapped(rep, PR05_CHECKS, suite, measured, min_tests=3)


# ---- e2e: the full model. PR-04/05 bring-up and PR-06 alignment with 1 replica, PR-07 with D replicas ----
def build_prompts(model, count, lo, hi, out):
    """``count`` token sequences, lengths log-spaced in [lo, hi], cut from one tokenized corpus (sglang's own
    sources: deterministic, offline, natural text and code). Cached in out/prompts.json."""
    path = os.path.join(out, "prompts.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from tokenizers import Tokenizer

    import sglang

    root = os.path.dirname(sglang.__file__)
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(root, "srt", "models")) for f in fs
                   if f.endswith(".py"))
    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    ids = []
    for p in files:
        with open(p, errors="replace") as f:
            ids += tok.encode(f.read()).ids
        if len(ids) > 4 * hi + count * 64:
            break
    lengths = [round(lo * (hi / lo) ** (i / max(1, count - 1))) for i in range(count)]
    prompts = [ids[(i * 7919) % (len(ids) - n):][:n] for i, n in enumerate(lengths)]
    with open(path, "w") as f:
        json.dump(prompts, f)
    return prompts


def input_id(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def collect(port: int, prompts: list[list[int]], tail: int = 256, rank: int | None = None) -> list[dict]:
    """Greedy first token with its top-5 logprobs, plus the logprobs of the last ``tail`` prompt tokens."""
    rows = []
    for ids in prompts:
        body_tail = max(0, len(ids) - tail)
        ok, secs, out = sv.generate(port, input_ids=ids, logprobs=True, rank=rank, logprob_start_len=body_tail)
        if not ok:
            rows.append({"error": str(out)[:300]})
            continue
        try:
            meta = out["meta_info"]
            raw = meta["input_token_logprobs"]
            # SGLang returns the requested input suffix, with its first unscored
            # token represented by None. Reject truncation before discarding it.
            if len(raw) != len(ids) - body_tail or [entry[1] for entry in raw] != ids[body_tail:]:
                raise ValueError("input logprobs do not cover the requested token suffix")
            if raw[0][0] is not None or any(entry[0] is None for entry in raw[1:]):
                raise ValueError("expected exactly one leading unscored input token")
            if len(meta["output_token_logprobs"]) != 1 or len(meta["output_top_logprobs"]) != 1:
                raise ValueError("expected exactly one generated token and one top-k result")
            output_logprob = meta["output_token_logprobs"][0][0]
            if (isinstance(output_logprob, bool) or not isinstance(output_logprob, (int, float))
                    or not math.isfinite(output_logprob)):
                raise ValueError("generated token logprob must be finite")
            row = {"token": meta["output_token_logprobs"][0][1],
                   "top": [t[:2] for t in meta["output_top_logprobs"][0]],
                   "prompt_lp": [t[0] for t in raw[1:]], "secs": secs,
                   "input_id": input_id(ids), "prompt_positions": list(range(body_tail + 1, len(ids)))}
            checked = compare([row], [row], expected_count=1)
            if not checked["valid"]:
                raise ValueError("; ".join(checked["problems"]))
            if not isinstance(secs, (int, float)) or not math.isfinite(secs) or secs <= 0:
                raise ValueError("request duration must be finite and positive")
            rows.append(row)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            rows.append({"error": f"invalid logprob response: {exc}"})
    return rows


def compare_results(ref: list[dict], got: list[dict], prompts: list[list[int]], tail: int = 256) -> dict:
    """Formal gates also bind both datasets to the input of this run."""
    metrics = compare(ref, got, expected_count=len(prompts))
    for label, rows in (("reference", ref), ("candidate", got)):
        if not isinstance(rows, list):
            continue
        for index, (row, ids) in enumerate(zip(rows, prompts)):
            if not isinstance(row, dict):
                continue
            positions = list(range(max(0, len(ids) - tail) + 1, len(ids)))
            if row.get("input_id") != input_id(ids) or row.get("prompt_positions") != positions:
                metrics["problems"].append(f"{label}[{index}] does not cover this run's input and positions; recollect")
    if metrics["problems"] or not metrics["identity_verified"]:
        metrics.update(valid=False, exact=False)
    return metrics


def reference(rep, args, ref_gpus, prompts):
    """Two stock (plugin-free) colocated layouts on the same prompts -> (reference, noise floor).

    The reference is --tp E --ep E (the F service's per-rank expert split). The second layout (attention DP,
    attention TP 1 like an A replica) shows how far two correct deployments of this model drift apart in bf16:
    the alignment criteria of the AFD run are relative to that distance, not to zero.
    """
    e, port, results = len(ref_gpus), args.port + 50, []
    layouts = [("reference", ("--tp-size", e, "--ep-size", e)),
               ("reference2", ("--tp-size", e, "--dp-size", e, "--enable-dp-attention", "--ep-size", e))]
    for name, flags in layouts:
        path = os.path.join(args.out, f"{name}.json")
        if args.reuse_reference and os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
            continue
        argv = sv.launch_server_argv(args.model, port, *flags, "--chunked-prefill-size", args.n_max,
                                     "--disable-cuda-graph")
        env = sv.server_env(",".join(map(str, ref_gpus)), SGLANG_PLUGINS="none")  # no plugin at all
        srv, _, ready = sv.start_server(lambda: sv.Server(name, argv, env, args.out, port), args.startup_timeout)
        try:
            rows = collect(port, prompts) if ready else None
        finally:
            srv.stop()
        if rows is None:
            rep.add("R0", f"stock layout {' '.join(map(str, flags))} comes up", srv.log_path, "/health 200", False)
            return None, None
        with open(path, "w") as f:
            json.dump(rows, f)
        results.append(rows)
    noise = compare_results(*results, prompts)
    rep.add("R0", f"noise floor: stock --tp {e} --ep {e} vs stock attention-DP --tp {e} --dp {e} --ep {e}",
            noise, "complete successful responses for this run's inputs; noise floor measured", noise["valid"])
    return (results[0], noise) if noise["valid"] else (None, None)


def start_afd(args, replicas, a_gpus, f_gpus, tag):
    """F (E ranks on f_gpus) then A (D replicas on a_gpus). A and F see the same GPU set: CUDA IPC needs it."""
    visible = ",".join(map(str, a_gpus + f_gpus))
    t0 = time.time()
    f = sv.FfnService(f"{tag}_ffn", sv.ffn_argv(args.model, len(f_gpus), replicas, args.rdv_port, args.n_max,
                                                "--base-gpu-id", len(a_gpus)), sv.server_env(visible), args.out).start()
    a_argv = sv.attn_argv(args.model, args.port, replicas, len(f_gpus), args.rdv_port, args.n_max, "--base-gpu-id", 0,
                          "--load-balance-method", "total_tokens")
    a, attempts, ready = sv.start_server(lambda: sv.Server(f"{tag}_attn", a_argv, sv.server_env(visible), args.out,
                                                           args.port), args.startup_timeout)
    return f, a, ready and f.is_ready(), time.time() - t0


def load_mem_gb(log):
    return [float(m) for m in re.findall(r"Load weight end\..*?mem usage=([\d.]+) GB", log)]


def soak(port, prompts_pool, count, concurrency, rank=None, seed=0):
    """``count`` requests from ``concurrency`` threads -> (errors, slowest seconds, wall seconds)."""
    import concurrent.futures
    import random

    rng = random.Random(seed)
    jobs = [rng.choice(prompts_pool) for _ in range(count)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda ids: sv.generate(port, input_ids=ids, rank=rank, timeout=600)[:2], jobs))
    return sum(not ok for ok, _ in results), max(s for _, s in results), time.time() - t0


def phase_one_replica(rep, args, a_gpus, f_gpus, prompts, ref, noise):
    e = len(f_gpus)
    f, a, ready, secs = start_afd(args, 1, a_gpus[:1], f_gpus, "afd_1a")
    try:
        frank = f.reports("ready")
        rep.add("E1", f"F service ({e} ranks, 43 layers) comes up", {r: (v["weights_gb"], v["total_s"]) for r, v in frank.items()}
                if frank else f"not ready, see {f.log_path}", f"every rank ready; weights ~ 147.2/{e} = {147.2 / e:.1f} GB "
                "per rank (experts only)", len(frank) == e and all(abs(v["weights_gb"] - 147.2 / e) < 3 for v in frank.values()))
        rep.add("E2", "A rank comes up on the full model and connects to F", f"{secs:.0f}s" if ready else a.log_path,
                f"/health 200 within {args.startup_timeout}s", ready)
        if not ready:
            return
        log, ranks, args_ok = a.log_text(), a.attn_ranks(), sv.ARGS_OK_RE.findall(a.log_text())
        connected = re.search(r"\[ASAP\] A rank 0 connected to (\d+) F rank", log)
        rep.add("E3", "hooks live in the launcher and the scheduler; AFD connected",
                f"launcher {[x[2] + '/' + x[3] for x in args_ok]}, rank {[r['ok'] + '/' + r['total'] for r in ranks]}, "
                f"connected to {connected.group(1) if connected else 0} F ranks", f"9/9 everywhere, afd=True, {e} F ranks",
                bool(args_ok) and all(x[2] == x[3] == "9" for x in args_ok) and len(ranks) == 1
                and ranks[0]["ok"] == ranks[0]["total"] == "9" and bool(connected) and int(connected.group(1)) == e)
        mem = load_mem_gb(log)
        rep.add("E4", "weights kept by an A rank (no routed expert allocated or read)", f"{mem} GB",
                "< 14 GB (stock colocated would need 166.9 GB / TP)", bool(mem) and mem[0] < 14)
        sizes = [10, 500, 3000, 6000, 16000]
        sized_prompts = [(prompts[-1] * (-(-n // len(prompts[-1]))))[:n] for n in sizes]
        served = [sv.generate(args.port, input_ids=ids, timeout=300)[:2] for ids in sized_prompts]
        rep.add("E5", "requests go through every MoE layer", dict(zip(sizes, [f"{s:.2f}s" if ok else "FAIL" for ok, s in served])),
                "all requested lengths succeed; chunk count follows --n-max", all(ok for ok, _ in served))
        got = collect(args.port, prompts)
        with open(os.path.join(args.out, "afd_1a.json"), "w") as fh:
            json.dump(got, fh)  # the PR-07 runs must reproduce these exactly
        if ref:
            m = compare_results(ref, got, prompts)
            rep.add("A1", f"first token vs the colocated reference ({m['n']} prompts, 128..{args.max_prompt} tokens)",
                    f"{m['first_token_match']}/{m['n']} identical, {m['errors']} errors",
                    f">= the stock-vs-stock floor ({noise['first_token_match']}/{noise['n']}) - 1, 0 errors",
                    m["valid"] and m["first_token_match"] >= noise["first_token_match"] - 1)
            rep.add("A2", f"logprob deltas vs the colocated reference ({m['prompt_positions']} prompt positions)",
                    m,
                    f"mean and p99 <= 1.25 x the floor ({noise['mean_d_prompt']:.4f}, {noise['p99_d_prompt']:.4f})",
                    m["valid"] and m["mean_d_prompt"] <= 1.25 * noise["mean_d_prompt"]
                    and m["p99_d_prompt"] <= 1.25 * noise["p99_d_prompt"])
        else:
            for cid in ("A1", "A2"):
                rep.add(cid, "reference alignment excluded", "--skip-reference", "required for full e2e acceptance", None)
        errors, slowest, wall = soak(args.port, [p[:4096] for p in prompts], args.soak_requests, 16)
        rep.add("A3", f"soak: {args.soak_requests} requests, 16 concurrent, 128..4096 tokens",
                f"{errors} errors, slowest {slowest:.1f}s, {wall:.0f}s wall", "0 errors, no request > 600 s", errors == 0)
    finally:
        a.stop()
        f.stop()
    stats = f.reports("stats")
    rep.add("E6", "F saw every MoE layer and lost nothing", {r: (s.get("layers_served"), s.get("jobs"), s.get("lost_messages", 0))
            for r, s in stats.items()}, "each rank: 43 layers, jobs > 0, lost 0, no stale epoch or dropped reply",
            set(stats) == set(range(e)) and all(s.get("layers_served") == 43 and s.get("jobs", 0) > 0
                                     and s.get("lost_messages", 0) == 0
                                     and not s.get("stale_epoch") and not s.get("dropped_replies") for s in stats.values()))
    return got


def windows(prompts, n, count, start=0):
    """``count`` distinct token windows of length n (distinct first tokens, so the radix cache cannot help)."""
    pool = [t for p in prompts for t in p]
    return [pool[(start + i * 997) % (len(pool) - n):][:n] for i in range(count)]


def run_load(short_inputs: list[list[int]], long_inputs: list[list[int]],
             request: Callable[[list[int], int], tuple[bool, float, object]], *, short_count: int,
             startup_timeout_s: float = 5, join_timeout_s: float = 905) -> dict:
    """Record actual request intervals; HTTP overlap does not prove GPU kernel overlap."""
    import threading
    from dataclasses import asdict

    def sample(ids, rank):
        started = time.monotonic()
        try:
            ok, _, response = request(ids, rank)
            error = None if ok else str(response)[:300]
        except Exception as exc:
            ok, error = False, repr(exc)
        finished = time.monotonic()
        return RequestSample(ok, started, finished, finished - started, error)

    solo = [sample(ids, 1) for ids in short_inputs[:short_count]]
    stopped, entered = threading.Event(), threading.Event()
    longs, loaded, worker_errors = [], [], []

    def long_load():
        try:
            for ids in long_inputs:
                if stopped.is_set():
                    break
                entered.set()
                result = sample(ids, 0)
                longs.append(result)
                if not result.ok:
                    break
        except BaseException as exc:
            worker_errors.append(repr(exc))

    worker = threading.Thread(target=long_load, daemon=True)
    worker.start()
    try:
        if not entered.wait(startup_timeout_s):
            worker_errors.append("background request did not start within deadline")
        else:
            loaded = [sample(ids, 1) for ids in short_inputs[short_count:]]
    finally:
        stopped.set()
        worker.join(timeout=join_timeout_s)
    # Snapshot before reporting: a timed-out worker must never mutate saved evidence.
    long_snapshot = list(longs)
    assessment = assess_load(solo, loaded, long_snapshot, expected_count=short_count,
                             worker_stopped=not worker.is_alive(), worker_error="; ".join(worker_errors) or None)

    def serializable(samples):
        rows = [asdict(sample) for sample in samples]
        for row in rows:
            for key in ("started_s", "finished_s", "elapsed_s"):
                if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                    row[key] = None  # Invalid raw value; assessment retains the failure reason.
        return rows

    return {"assessment": assessment, "clock": "time.monotonic (seconds)",
            "solo": serializable(solo), "loaded": serializable(loaded), "long": serializable(long_snapshot)}


def phase_replicas(rep, args, d, a_gpus, f_gpus, prompts, one_replica):
    """PR-07: D replicas share one F with no synchronization between them."""
    f, a, ready, secs = start_afd(args, d, a_gpus[:d], f_gpus, f"afd_{d}a")
    tag = f"M{d}"
    try:
        ranks = a.attn_ranks() if ready else []
        rep.add(f"{tag}.0", f"{d} replicas come up, each its own world, all connected to F",
                [(r["replica"], r["world"], r["ok"] + "/" + r["total"]) for r in ranks] if ready else a.log_path,
                f"{d} ranks, replicas 0..{d - 1}, world_size 1, hooks 9/9",
                ready and sorted(int(r["replica"]) for r in ranks) == list(range(d))
                and all(r["world"] == "1" and r["ok"] == r["total"] == "9" for r in ranks))
        if not ready:
            return
        count = min(16, len(prompts))
        if one_replica is None:
            rep.add(f"{tag}.1", "single-replica baseline missing", "current run did not produce a baseline",
                    "current-run single-replica results required", False)
        else:
            per = [compare_results(one_replica[:count], collect(args.port, prompts[:count], rank=r),
                                   prompts[:count]) for r in range(d)]
            rep.add(f"{tag}.1", "every replica reproduces the current 1-replica AFD results", per,
                    f"each replica: {count}/{count} first tokens, complete identity and bit-identical logprobs",
                    all(m["valid"] and m["identity_verified"] and m["exact"] for m in per))
        errors, slowest, wall = soak(args.port, windows(prompts, 2048, 64, 11), args.soak_requests * d, 8 * d)
        rep.add(f"{tag}.2", f"soak: {args.soak_requests * d} requests, {8 * d} concurrent, routed by sglang (total_tokens)",
                f"{errors} errors, slowest {slowest:.1f}s, {wall:.0f}s wall", "0 errors, no hang", errors == 0)
        short = windows(prompts, 1024, 2 * args.short_requests, 5)
        load = run_load(short, windows(prompts, args.long_len, 1000, 3),
                        lambda ids, rank: sv.generate(args.port, input_ids=ids, rank=rank, timeout=900),
                        short_count=args.short_requests)
        with open(os.path.join(args.out, f"{tag}_load.json"), "w") as output:
            json.dump(load, output, indent=2, allow_nan=False)
        rep.add(f"{tag}.3", "no barrier: short requests alone vs overlapping long requests", load["assessment"],
                "all requests succeed with in-flight HTTP overlap; short p50 ratio <= 1.5; "
                "loaded short p50 < 0.5 x overlapping long p50", load["assessment"]["passed"])
    finally:
        a.stop()
        f.stop()
    stats = f.reports("stats")
    switches = {r: s.get("replica_switches", 0) for r, s in stats.items()}
    skew = {r: _layer_skew(s.get("trace_tail", [])) for r, s in stats.items()}
    rep.add(f"{tag}.4", "F served the replicas interleaved and at different layers",
            f"replica switches {switches}; consecutive jobs of different replicas at different layers {skew}",
            "> 0 replica switches and > 0 cross-replica layer gaps on every rank; lost 0",
            set(stats) == set(range(len(f_gpus))) and all(switches[r] > 0 and skew[r] > 0 and stats[r].get("lost_messages", 0) == 0 for r in stats))


def _layer_skew(trace):
    """How often two consecutive jobs come from different replicas at different layers: > 0 means replicas
    advance independently (lock-stepped replicas would reach F with the same layer)."""
    return sum(1 for (r1, _, l1), (r2, _, l2) in zip(trace, trace[1:]) if r1 != r2 and l1 != l2)


def stage_e2e(rep, args):
    validate_args(args)
    a_gpus, f_gpus = [int(x) for x in args.a_gpus.split(",")], [int(x) for x in args.f_gpus.split(",")]
    prompts = build_prompts(args.model, args.prompts, 128, args.max_prompt, args.out)
    if len(prompts) != args.prompts or any(not isinstance(ids, list) or len(ids) < 2 for ids in prompts):
        raise ValueError("prompt cache does not contain the requested number of usable inputs; regenerate it")
    if any(d > 1 for d in args.replicas) and sum(map(len, prompts)) <= max(2048, args.long_len):
        raise ValueError("prompt corpus is too short for the requested load windows")
    ref_gpus = [int(x) for x in args.ref_gpus.split(",")] if args.ref_gpus else f_gpus
    if args.skip_reference:
        rep.add("R0", "reference excluded", "--skip-reference", "required for full e2e acceptance", None)
        ref, noise = None, None
    else:
        ref, noise = reference(rep, args, ref_gpus, prompts)
        if ref is None:
            return
    one_replica = None
    for d in sorted(args.replicas):
        if d == 1:
            one_replica = phase_one_replica(rep, args, a_gpus, f_gpus, prompts, ref, noise)
        else:
            phase_replicas(rep, args, d, a_gpus, f_gpus, prompts, one_replica)


def required_checks(args: argparse.Namespace) -> set[str]:
    if args.stage == "m0":
        return {"U1", "U2", "U3", "N1", *(f"G{i}" for i in range(5))}
    if args.stage in ("pr03", "pr04", "pr05"):
        checks = {"pr03": PR03_CHECKS, "pr04": PR04_CHECKS, "pr05": PR05_CHECKS}[args.stage]
        return {"U", *(check[0] for check in checks)}
    required = {"R0"}
    for replicas in args.replicas:
        if replicas == 1:
            required.update({*(f"E{i}" for i in range(1, 7)), "A1", "A2", "A3"})
        else:
            required.update(f"M{replicas}.{i}" for i in range(5))
    return required


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid scope/topology before loading models or starting processes."""
    for field in ("n_max", "prompts", "soak_requests", "short_requests", "long_len", "startup_timeout", "test_timeout"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.max_prompt < 128:
        raise ValueError("max_prompt must be at least 128")
    if not (1 <= args.port <= 65435 and 1 <= args.rdv_port <= 65535):
        raise ValueError("invalid port (including the +100 control offset)")
    if args.stage != "m0" and (args.skip_gpu or args.skip_control):
        raise ValueError("--skip-gpu and --skip-control apply only to m0")
    if args.stage != "e2e" and (args.skip_reference or args.reuse_reference):
        raise ValueError("reference flags apply only to e2e")
    if args.skip_reference and args.reuse_reference:
        raise ValueError("cannot both skip and reuse the reference")

    def gpu_list(value, field):
        ids = [int(part) for part in value.split(",")]
        if not ids or len(ids) != len(set(ids)) or any(gpu < 0 for gpu in ids):
            raise ValueError(f"{field} must contain distinct nonnegative GPU indices")
        return ids

    if args.stage == "e2e":
        a_gpus = gpu_list(args.a_gpus, "a_gpus")
        f_gpus = gpu_list(args.f_gpus, "f_gpus")
        if set(a_gpus) & set(f_gpus):
            raise ValueError("A and F GPU sets must be disjoint")
        if args.ref_gpus:
            gpu_list(args.ref_gpus, "ref_gpus")
        if len(args.replicas) != len(set(args.replicas)) or any(d < 1 or d > len(a_gpus) for d in args.replicas):
            raise ValueError("replicas must be distinct positive counts within the A GPU count")
        if args.replicas and 1 not in args.replicas:
            raise ValueError("multi-replica acceptance requires replica count 1 in the same run")
    else:
        gpu_list(args.gpus, "gpus")
        if args.stage == "m0" and not args.skip_gpu and len(args.gpus.split(",")) < 2:
            raise ValueError("m0 requires at least two A GPUs")
        if args.stage == "m0" and not (args.skip_gpu or args.skip_control):
            if len(gpu_list(args.control_gpus, "control_gpus")) < 2:
                raise ValueError("m0 control requires at least two GPUs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["m0", "pr03", "pr04", "pr05", "e2e"])
    ap.add_argument("--model", default=sv.MODEL)
    ap.add_argument("--gpus", default="0,1,2,3", help="m0: the A server's GPUs (2); pr03-pr05: GPUs the tests may use")
    ap.add_argument("--control-gpus", default="2,3", help="m0: GPUs of the stock attention-DP control")
    ap.add_argument("--a-gpus", default="0,1", help="e2e: GPUs of the A replicas (one each)")
    ap.add_argument("--f-gpus", default="4,5,6,7", help="e2e: GPUs of the F ranks (also the colocated reference)")
    ap.add_argument("--ref-gpus", default=None, help="e2e: GPUs of the colocated reference (default: --f-gpus)")
    ap.add_argument("--replicas", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 2],
                    help="e2e: replica counts to run, e.g. 1,2 ('' = only the reference)")
    ap.add_argument("--n-max", type=int, default=8192, help="e2e: --asap-n-max and --chunked-prefill-size")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=16384)
    ap.add_argument("--soak-requests", type=int, default=1000)
    ap.add_argument("--short-requests", type=int, default=20)
    ap.add_argument("--long-len", type=int, default=32768)
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--rdv-port", type=int, default=9600)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--test-timeout", type=int, default=3600, help="maximum seconds for each unittest subprocess")
    ap.add_argument("--out", default=None, help="logs and report (default /tmp/asap_reports/<stage>)")
    ap.add_argument("--reuse-reference", action="store_true", help="e2e: reuse out/reference*.json if present")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true", help="m0: CPU checks only")
    ap.add_argument("--skip-control", action="store_true", help="m0: no stock attention-DP control")
    args = ap.parse_args(argv)
    args.out = args.out or f"/tmp/asap_reports/{args.stage}"
    os.makedirs(args.out, exist_ok=True)
    rep = Report(args.out, required_checks(args), partial=args.stage == "e2e" and not args.replicas)
    try:
        validate_args(args)
        globals()[f"stage_{args.stage}"](rep, args)
    except (Exception, KeyboardInterrupt) as exc:
        with open(os.path.join(args.out, "exception.log"), "w") as output:
            traceback.print_exc(file=output)
        rep.add("EXCEPTION", "acceptance interrupted", f"{type(exc).__name__}: {exc}", "stage must complete", False)
    finally:
        rep.save()
    outcome = rep.outcome()
    print(f"\n{args.stage} acceptance: {outcome}  (report: {args.out}/report.md)")
    return {"PASS": 0, "FAIL": 1, "PARTIAL": 2}[outcome]


if __name__ == "__main__":
    sys.exit(main())
