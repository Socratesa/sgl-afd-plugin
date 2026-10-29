#!/usr/bin/env python
"""PR-03 acceptance: the transport test suite, one PASS/FAIL table, one JSON report.

    python scripts/pr03_acceptance.py --gpus 0,1,2,3

A and F ranks are separate processes on separate GPUs, so the suite needs 4 GPUs on one machine
(--gpus lists their ordinals). The tests assert the gates themselves; this script maps each test
to an acceptance ID, adds the measured numbers the tests print, and fails on any skipped test.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_TESTS = 22

# acceptance ID, title, test names (any of), criterion
CHECKS = [
    ("C1", "wire format: doorbell fields, geometry rules, arena sizes",
     ["test_doorbell_round_trip_at_the_field_limits", "test_doorbell_rejects_out_of_range_fields",
      "test_geometry_cannot_exceed_what_the_doorbell_can_announce", "test_invalid_geometry_names_the_field",
      "test_sizes_at_the_design_point", "test_done_flags_never_share_a_page_with_payload"],
     "every field round-trips at its limits; out-of-range and oversized geometry rejected"),
    ("C2", "ping-pong, payload and layer id verified", ["test_one_a_one_f_1000_rounds"],
     "1000 rounds, 0 wrong results, 0 lost messages"),
    ("C3", "three F ranks, partial sums add up", ["test_three_f_ranks_partial_sums_add_up"], "200 rounds, 0 wrong"),
    ("C4", "message sizes 8 KB to 64 MB", ["test_sizes_from_8kb_to_64mb"], "0 wrong"),
    ("C5", "sequence numbers wrap past 2^32", ["test_sequence_numbers_wrap_around_2_pow_32"],
     "start 0xFFFFFFF0, 0 wrong, 0 lost"),
    ("C6", "stress", ["test_stress_18k_rounds"], "18000 rounds, 0 wrong, 18000 jobs"),
    ("C7", "first come first served with three senders", ["test_first_come_first_served_with_three_senders"],
     "900 jobs, all senders served, interleaved (> 100 switches), 0 wrong"),
    ("B1", "backpressure", ["test_second_send_into_a_busy_slot_blocks_and_nothing_is_lost"],
     "second send into a busy slot stays pending; both messages delivered in order; 0 lost"),
    ("B2", "control: no backpressure", ["test_control_without_backpressure_the_loss_is_detected"],
     "the overwrite is detected (lost_messages >= 1), so B1 can fail"),
    ("F1", "dead F rank", ["test_dead_f_times_out_instead_of_hanging"],
     "wait fails in 1.5-10 s (timeout 2 s) with ChannelPoisoned; later sends refused"),
    ("F2", "stale epoch and reconnect", ["test_stale_epoch_message_is_dropped_and_reconnect_recovers"],
     "stale message dropped and counted; reconnect gives a working channel (20 rounds, 0 wrong)"),
    ("F3", "killed A rank", ["test_killed_a_does_not_disturb_f_or_the_other_a"],
     "other A: 8000 rounds, 0 wrong; a new incarnation reconnects; 0 lost"),
    ("F4", "handshake geometry mismatch", ["test_handshake_rejects_a_geometry_mismatch"],
     "A fails fast with a message naming the field"),
    ("F5", "cold work behind a parked wait", ["test_parked_wait_delays_cold_work_but_does_not_deadlock"],
     "cold work finishes within release delay + 15 s; warm work < 0.5 s"),
    ("P1", "small-message round trip", ["test_small_message_round_trip"],
     "n=1 RTT p50 < 150 us with a spinning F; a napping F is > 40 us slower"),
    ("P2", "64 MB one way and full duplex", ["test_64mb_one_way_and_full_duplex"],
     "one way > 330 GB/s; send and reply overlapped > 650 GB/s"),
    ("P3", "64 MB round trip", ["test_64mb_round_trip_bandwidth"], "aggregate > 150 GB/s"),
]


def run_suite(gpus, out):
    paths = [os.path.join(REPO, "python"), os.path.join(REPO, "tests")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), ASAP_TEST_GPUS=gpus)
    start = time.time()
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "test_transport", "test_transport_faults",
                        "-v"], cwd=REPO, env=env, capture_output=True, text=True)
    with open(os.path.join(out, "transport_tests.log"), "w") as f:
        f.write(p.stderr + "\n--- stdout ---\n" + p.stdout)
    status = {m.group(1): m.group(2) for m in re.finditer(r"^(test_\w+) \(\S+\) \.\.\. (.*)$", p.stderr, re.M)}
    return status, p.stdout, time.time() - start


def metrics(stdout):
    """The numbers the perf tests print, keyed by acceptance ID."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3", help="ordinals the tests may use (>= 4)")
    ap.add_argument("--out", default="/tmp/asap_reports/pr03", help="logs and report (not on the shared volume)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    status, stdout, secs = run_suite(args.gpus, args.out)
    seen = metrics(stdout)
    mapped = {t for _, _, tests, _ in CHECKS for t in tests}
    rows = []
    for cid, title, tests, criterion in CHECKS:
        results = [status.get(t, "not run") for t in tests]
        ok = all(r == "ok" for r in results)
        measured = seen.get(cid) or (f"{len(tests)} test(s) ok" if ok else ", ".join(
            f"{t}: {r}" for t, r in zip(tests, results) if r != "ok"))
        rows.append({"id": cid, "title": title, "measured": measured, "criterion": criterion,
                     "status": "PASS" if ok else "FAIL"})
    extra = sorted(set(status) - mapped)
    total_ok = len(status) >= MIN_TESTS and not extra
    rows.append({"id": "U1", "title": "suite size and mapping",
                 "measured": f"{len(status)} tests, {secs:.0f} s, unmapped: {extra or 'none'}",
                 "criterion": f">= {MIN_TESTS} tests, every test mapped to an ID, 0 skipped",
                 "status": "PASS" if total_ok and not any(v.startswith("skipped") for v in status.values()) else "FAIL"})
    for r in rows:
        print(f"[{r['status']}] {r['id']} {r['title']}: {r['measured']}   (criterion: {r['criterion']})")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    lines = ["| ID | Check | Measured | Criterion | Result |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['title']} | {r['measured']} | {r['criterion']} | {r['status']} |" for r in rows]
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    ok = all(r["status"] == "PASS" for r in rows)
    print(f"\nPR-03 acceptance: {'PASS' if ok else 'FAIL'}  (report: {args.out}/report.md)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
