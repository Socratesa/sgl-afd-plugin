"""Process orchestration for the transport tests: spawn F and A roles, collect their reports."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time

import transport_roles as roles

DEFAULT_TIMEOUT = 300.0


def gpu_ordinals():
    return [int(x) for x in os.environ.get("ASAP_TEST_GPUS", "0,1,2,3").split(",")]


class Cluster:
    """E F ranks (one process each) plus any number of A processes on one machine."""

    def __init__(self, geometry, f_devices, *, hold_f=False, seq_base=0, f_mode="compute", f_spin_s=None):
        self.ctx = mp.get_context("spawn")
        self.q, self.stop, self.go = self.ctx.Queue(), self.ctx.Event(), self.ctx.Event()
        if not hold_f:
            self.go.set()
        self.g, self.f_devices, self.seq_base, self.f_mode = geometry, f_devices, seq_base, f_mode
        self.f_spin_s = f_spin_s
        self.run_dir = tempfile.mkdtemp(prefix="asap_t_", dir="/dev/shm")
        self.f_procs, self.a_procs, self.inbox, self.endpoints = {}, {}, [], []

    def start_f(self):
        for f, dev in enumerate(self.f_devices):
            cfg = dict(f=f, device=dev, geometry=self.g.as_dict(), run_dir=self.run_dir, seq_base=self.seq_base,
                       f_mode=self.f_mode, spin_s=self.f_spin_s)
            proc = self.ctx.Process(target=roles.f_main, args=(cfg, self.q, self.go, self.stop), daemon=True)
            proc.start()
            self.f_procs[f] = proc
        ports = {m[1]: m[2] for m in self.wait("f_ready", len(self.f_devices))}
        self.endpoints = [("127.0.0.1", ports[f]) for f in range(len(self.f_devices))]
        return self

    def start_a(self, a, device, scenario, **extra):
        cfg = dict(a=a, device=device, geometry=self.g.as_dict(), endpoints=self.endpoints, scenario=scenario,
                   seq_base=self.seq_base, **extra)
        solo = scenario in roles.SOLO
        proc = self.ctx.Process(target=roles.solo_main if solo else roles.a_main,
                                args=(cfg, self.q) if solo else (cfg, self.q, self.go), daemon=True)
        proc.start()
        self.a_procs[a] = proc
        return proc

    def wait(self, kind, count=1, timeout=DEFAULT_TIMEOUT, errors_ok=False, match=None):
        """Block until ``count`` messages of ``kind`` (that satisfy ``match``) arrived; a role's traceback fails the test.

        Messages that do not match stay in the inbox for a later wait."""
        out, deadline = [], time.time() + timeout
        while True:
            for m in [m for m in self.inbox if m[0] == kind and (match is None or match(m))][:count - len(out)]:
                self.inbox.remove(m)
                out.append(m)
            if len(out) >= count:
                return out
            if time.time() > deadline:
                raise TimeoutError(f"waiting for {count} x {kind}, got {len(out)}; inbox={self.inbox}")
            try:
                m = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if m[0] in ("f_error", "a_error") and not (errors_ok and m[0] == kind):
                raise RuntimeError(f"{m[0]}[{m[1]}]:\n{m[2]}")
            self.inbox.append(m)

    def result(self, a, timeout=DEFAULT_TIMEOUT):
        return self.wait("a_result", 1, timeout, match=lambda m: m[1] == a)[0][2]

    def finish(self):
        """Stop the live F ranks and return {f: (stats, service_order, jobs)}."""
        live = [f for f, p in self.f_procs.items() if p.is_alive()]
        self.stop.set()
        self.go.set()
        stats = {m[1]: m[2:] for m in self.wait("f_stats", len(live), 60)}
        for proc in self.f_procs.values():
            proc.join(timeout=10)
        return stats

    def close(self):
        for proc in [*self.a_procs.values(), *self.f_procs.values()]:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
        shutil.rmtree(self.run_dir, ignore_errors=True)
