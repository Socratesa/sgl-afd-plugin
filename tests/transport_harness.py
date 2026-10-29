"""Multi-process harness of the GPU tests: F ranks and A ranks as spawned processes (each owns its CUDA context),
the A-side scenarios they run, and the synthetic F computation they check.

Synthetic F (f_mode="compute"): rank f answers each row with y = (f + 1) * base(x, ids, w, layer), so the sum over
the E ranks is S * base with S = E(E+1)/2; x, ids, w and the layer id all take part in the check, and every value
is a small integer, exact in bf16 and fp32. f_mode="echo" answers zeros without computing (pure transport).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

DEFAULT_TIMEOUT = 300.0


def gpu_ordinals():
    return [int(x) for x in os.environ.get("ASAP_TEST_GPUS", "0,1,2,3").split(",")]


class Cluster:
    """E F ranks plus any number of A processes on one machine; messages from the roles arrive on one queue."""

    def __init__(self, geometry, f_devices, *, hold_f=False, seq_base=0, f_mode="compute", f_spin_s=None):
        self.ctx = mp.get_context("spawn")
        self.q, self.stop, self.go = self.ctx.Queue(), self.ctx.Event(), self.ctx.Event()
        if not hold_f:
            self.go.set()
        self.g, self.f_devices, self.seq_base, self.f_mode, self.f_spin_s = geometry, f_devices, seq_base, f_mode, f_spin_s
        self.run_dir = tempfile.mkdtemp(prefix="asap_t_", dir="/dev/shm")
        self.f_procs, self.a_procs, self.inbox, self.endpoints = {}, {}, [], []

    def start_f(self):
        for f, dev in enumerate(self.f_devices):
            cfg = dict(f=f, device=dev, geometry=self.g.as_dict(), run_dir=self.run_dir, seq_base=self.seq_base,
                       f_mode=self.f_mode, spin_s=self.f_spin_s)
            proc = self.ctx.Process(target=f_main, args=(cfg, self.q, self.go, self.stop), daemon=True)
            proc.start()
            self.f_procs[f] = proc
        ports = {m[1]: m[2] for m in self.wait("f_ready", len(self.f_devices))}
        self.endpoints = [("127.0.0.1", ports[f]) for f in range(len(self.f_devices))]
        return self

    def start_a(self, a, device, scenario, **extra):
        cfg = dict(a=a, device=device, geometry=self.g.as_dict(), endpoints=self.endpoints, scenario=scenario,
                   seq_base=self.seq_base, **extra)
        proc = self.ctx.Process(target=a_main, args=(cfg, self.q, self.go), daemon=True)
        proc.start()
        self.a_procs[a] = proc
        return proc

    def wait(self, kind, count=1, timeout=DEFAULT_TIMEOUT, errors_ok=False, match=None):
        """Block until ``count`` messages of ``kind`` (satisfying ``match``) arrived; a role's traceback fails the
        test. Non-matching messages stay in the inbox for a later wait."""
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
        """Stop the live F ranks; returns {f: (stats, trace, rows)}."""
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


def tview(torch, ptr, shape, dtype, device):
    """A tensor aliasing device memory owned by the transport."""
    from sgl_asap_plugin.transport.cuda import DeviceBuffer

    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    return torch.as_tensor(DeviceBuffer(ptr, nbytes), device=f"cuda:{device}").view(dtype).view(shape)


def pattern(torch, g, a, r, n, device):
    i = torch.arange(n, device=f"cuda:{device}")[:, None]
    c = torch.arange(g.hidden, device=f"cuda:{device}")[None, :]
    k = torch.arange(g.topk, device=f"cuda:{device}")[None, :]
    x = ((r * 7 + i * 3 + c + a * 3) % 20).to(torch.bfloat16)
    ids = ((r + i + k + a) % 10).to(torch.int32)
    w = ((r + i + k) % 5 + 1).to(torch.float32) / 4
    return x, ids, w


def expected(torch, g, x, ids, w, layer_id):
    base = x.float() * 2
    base[:, 0] = ids.sum(1).float()
    base[:, 1] = w.sum(1) * 4
    base[:, 2] = float(layer_id % 50)
    return base * (g.num_f * (g.num_f + 1) // 2)


def f_main(cfg, out_q, go_evt, stop_evt):
    """A synthetic F rank: the real FfnServer, with the deterministic computation of the module docstring."""
    try:
        import torch

        from sgl_asap_plugin.transport import FfnServer, Geometry

        dev, g = cfg["device"], Geometry(**cfg["geometry"])
        torch.cuda.set_device(dev)
        torch.zeros(1, device=f"cuda:{dev}")
        srv = FfnServer(cfg["f"], g, device=dev, run_dir=cfg["run_dir"], seq_base=cfg.get("seq_base", 0),
                        trace_len=100000).start()
        out_q.put(("f_ready", cfg["f"], srv.port))
        go_evt.wait()
        stream, scale, ybuf, rows = torch.cuda.Stream(dev), cfg["f"] + 1, {}, []
        zeros = torch.zeros(g.n_max, g.hidden, dtype=torch.bfloat16, device=f"cuda:{dev}")

        def handler(job):
            for row in job.rows:
                rows.append((row.a, job.layer_id, row.seq, row.n_tokens))
                n, nbytes = row.n_tokens, row.n_tokens * g.hidden * 2
                if cfg.get("f_mode") == "echo":  # pure transport: nothing between receive and reply
                    srv.reply(row, zeros.data_ptr(), nbytes, stream.cuda_stream)
                    continue
                x = tview(torch, row.x_ptr, (n, g.hidden), torch.bfloat16, dev)
                ids = tview(torch, row.ids_ptr, (n, g.topk), torch.int32, dev)
                w = tview(torch, row.w_ptr, (n, g.topk), torch.float32, dev)
                y = ybuf.setdefault((row.a, row.slot), torch.empty(g.n_max, g.hidden, dtype=torch.bfloat16,
                                                                   device=f"cuda:{dev}"))
                with torch.cuda.stream(stream):
                    torch.mul(x, 2 * scale, out=y[:n])
                    y[:n, 0] = (ids.sum(1) * scale).to(torch.bfloat16)
                    y[:n, 1] = (w.sum(1) * 4 * scale).to(torch.bfloat16)
                    y[:n, 2] = (job.layer_id % 50) * scale
                srv.reply(row, y.data_ptr(), nbytes, stream.cuda_stream)

        spin = {} if cfg.get("spin_s") is None else {"spin_s": cfg["spin_s"]}
        srv.serve_forever(handler, stop_evt, **spin)
        torch.cuda.synchronize(dev)
        out_q.put(("f_stats", cfg["f"], dict(srv.stats), list(srv.trace), rows))
        srv.close()
    except BaseException:
        out_q.put(("f_error", cfg["f"], traceback.format_exc()))


def a_main(cfg, out_q, go_evt):
    """An A rank running one scenario. Scenarios with ``channel=False`` get no channel and need no F."""
    try:
        import torch

        from sgl_asap_plugin.transport import AfdChannel, Geometry

        dev, scenario = cfg["device"], SCENARIOS[cfg["scenario"]]
        torch.cuda.set_device(dev)
        torch.zeros(1, device=f"cuda:{dev}")
        ch, g = None, Geometry(**{**cfg["geometry"], **cfg.get("geometry_override", {})})
        if getattr(scenario, "channel", True):
            ch = AfdChannel(cfg["a"], g, [tuple(e) for e in cfg["endpoints"]], device=dev,
                            seq_base=cfg.get("seq_base", 0), timeout_s=cfg.get("timeout_s", 30.0),
                            **cfg.get("channel_kwargs", {})).connect()
        result = scenario(ch, torch, g, cfg, out_q, go_evt)
        out_q.put(("a_result", cfg["a"], result))
        if ch is not None:
            ch.close()
    except BaseException:
        out_q.put(("a_error", cfg["a"], traceback.format_exc()))


# ---- A-side scenarios: each returns a dict for the test to assert on ----
def _buffers(torch, g, dev):
    d = f"cuda:{dev}"
    return [(torch.empty(g.n_max, g.hidden, dtype=torch.bfloat16, device=d),
             torch.empty(g.n_max, g.topk, dtype=torch.int32, device=d),
             torch.empty(g.n_max, g.topk, dtype=torch.float32, device=d)) for _ in range(g.slots)]


def _percentiles(values):
    values = sorted(values)
    return {"p50": values[len(values) // 2], "p99": values[min(len(values) - 1, int(len(values) * 0.99))]}


def _raises(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return None


def pingpong(ch, torch, g, cfg, out_q, go_evt):
    """Strict request/response rounds; every payload array and the layer id are verified."""
    dev, a = cfg["device"], cfg["a"]
    stream = torch.cuda.Stream(dev)
    s, bufs, sizes = stream.cuda_stream, _buffers(torch, g, dev), cfg.get("sizes") or [cfg.get("n", 4)]
    rng, bad, lats = random.Random(a), 0, []
    for r in range(1, cfg["rounds"] + 1):
        n, slot, layer = sizes[r % len(sizes)], r % g.slots, r % 61
        xb, ib, wb = bufs[slot]
        with torch.cuda.stream(stream):
            x, ids, w = pattern(torch, g, a, r, n, dev)
            xb[:n].copy_(x)
            ib[:n].copy_(ids)
            wb[:n].copy_(w)
        if cfg.get("think_us"):
            time.sleep(rng.uniform(0, cfg["think_us"]) / 1e6)
        stream.synchronize()  # data generation must not count as transport time
        t0 = time.perf_counter()
        ticket = ch.send(layer, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s)
        ptrs = ch.wait(ticket, s)
        if cfg.get("verify", True):
            with torch.cuda.stream(stream):
                got = sum(tview(torch, p, (n, g.hidden), torch.bfloat16, dev).float() for p in ptrs)
                bad += 0 if torch.equal(got, expected(torch, g, x, ids, w, layer)) else 1
        else:
            stream.synchronize()
        lats.append((time.perf_counter() - t0) * 1e6)
        ch.check()
    return {"rounds": cfg["rounds"], "bad": bad, "lat_us": _percentiles(lats)}


def oneway(ch, torch, g, cfg, out_q, go_evt):
    """Copy rate of A's send of a full n_max message, timed on A's own stream with CUDA events."""
    dev = cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, bufs, n, ms = stream.cuda_stream, _buffers(torch, g, dev), g.n_max, []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(dev)
    for r in range(cfg["rounds"]):
        slot = r % g.slots
        xb, ib, wb = bufs[slot]
        e0.record(stream)
        ticket = ch.send(r % 61, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s)
        e1.record(stream)
        ch.wait(ticket, s)  # the reply frees the slot for the next round
        e1.synchronize()
        ms.append(e0.elapsed_time(e1))
    stream.synchronize()
    ch.check()
    sent = n * (g.hidden * 2 + 2 * g.topk * 4)
    return {"gbps": sent / (sorted(ms)[len(ms) // 2] * 1e-3) / 1e9}


def duplex(ch, torch, g, cfg, out_q, go_evt):
    """One message per slot in flight, so A's next send overlaps F's reply to the previous one (reverse link)."""
    dev = cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, bufs, n, pending = stream.cuda_stream, _buffers(torch, g, dev), g.n_max, []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(dev)
    for r in range(cfg["warmup"] + cfg["rounds"]):
        if r == cfg["warmup"]:
            e0.record(stream)
        slot = r % g.slots
        xb, ib, wb = bufs[slot]
        pending.append(ch.send(r % 61, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s))
        if len(pending) == g.slots:
            ch.wait(pending.pop(0), s)
    for ticket in pending:
        ch.wait(ticket, s)
    e1.record(stream)
    e1.synchronize()
    ch.check()
    ms = e0.elapsed_time(e1)
    moved = cfg["rounds"] * n * (g.hidden * 2 + 2 * g.topk * 4 + g.hidden * 2)  # A to F plus the reply back
    return {"gbps": moved / (ms * 1e-3) / 1e9, "us_per_message": ms * 1e3 / cfg["rounds"]}


def backpressure(ch, torch, g, cfg, out_q, go_evt):
    """F is held back by the test; A sends twice into the same slot: the second send must stay parked."""
    dev = cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, n, msgs = stream.cuda_stream, 4, []
    for r in (1, 2):
        with torch.cuda.stream(stream):
            msgs.append([t.contiguous() for t in pattern(torch, g, cfg["a"], r, n, dev)])
    torch.cuda.synchronize(dev)
    tickets = [ch.send(r, 0, m[0].data_ptr(), m[1].data_ptr(), m[2].data_ptr(), n, s) for r, m in zip((1, 2), msgs)]
    time.sleep(cfg["hold_s"])  # > Watchdog.SCAN_S, so the states below are current
    state = [t.send.state for t in tickets]
    go_evt.set()
    for t in tickets:
        ch.wait(t, s)
    stream.synchronize()
    ch.check()
    return {"send_states_while_f_held": state}


def expect_timeout(ch, torch, g, cfg, out_q, go_evt):
    """One message that is never answered (F dead or dropping it): the wait must fail, not hang."""
    dev = cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, n = stream.cuda_stream, 4
    with torch.cuda.stream(stream):
        x, ids, w = [t.contiguous() for t in pattern(torch, g, cfg["a"], 1, n, dev)]
    torch.cuda.synchronize(dev)
    ch.wait(ch.send(1, 0, x.data_ptr(), ids.data_ptr(), w.data_ptr(), n, s), s)
    out_q.put(("a_waiting", cfg["a"]))
    t0 = time.perf_counter()
    stream.synchronize()  # returns only because the watchdog released the stream
    out = {"waited_s": round(time.perf_counter() - t0, 2), "check": _raises(ch.check)}
    out["send_after"] = _raises(lambda: ch.send(2, 1, x.data_ptr(), ids.data_ptr(), w.data_ptr(), n, s))
    if cfg.get("recover"):  # reconnect (new epoch, fresh ResultArena), then the channel must work again
        ch._epoch_override = None
        ch.reconnect()
        out["after_reconnect"] = pingpong(ch, torch, g, dict(cfg, rounds=20), out_q, go_evt)
    return out


_RELEASER = """
import sys, time
from sgl_asap_plugin.transport import cuda as cu
cu.Device(int(sys.argv[3])).activate()
flag = cu.ipc_open(bytes.fromhex(sys.argv[1]))
stream = cu.stream_create()
time.sleep(float(sys.argv[2]))
cu.write32(stream, flag, 1)
cu.stream_sync(stream)
"""


def parked_wait_delays_cold_work_only(ch, torch, g, cfg, out_q, go_evt):
    """A stream parked in a memop wait delays cold work (first-use kernels, cudaMalloc) until the peer's write
    arrives and leaves warm work alone. The release comes from another process, as it does from F."""
    from sgl_asap_plugin.transport import cuda as cu

    dev, delay = cfg["device"], 2.0
    cu.Device(dev).activate()
    flag = cu.mem_alloc(256)
    handle, work = cu.ipc_export(flag).hex(), torch.cuda.Stream(dev)

    def op():
        x = torch.arange(4096.0, device=f"cuda:{dev}")
        return torch.linalg.vector_norm(x.cumsum(0).sin())

    def timed():
        parked = cu.stream_create()
        cu.memset32(flag, 0, 64)
        cu.wait32(parked, flag, 1)
        helper = subprocess.Popen([sys.executable, "-c", _RELEASER, handle, str(delay), str(cfg["release_device"])])
        t0 = time.perf_counter()
        with torch.cuda.stream(work):
            op()
        work.synchronize()
        elapsed = time.perf_counter() - t0
        helper.wait()
        cu.stream_sync(parked)
        return round(elapsed, 3)

    cold, warm = timed(), timed()
    return {"delay_s": delay, "cold_s": cold, "warm_s": warm}


parked_wait_delays_cold_work_only.channel = False


def attn_forward(ch, torch, g, cfg, out_q, go_evt):
    """attn.moe_forward against synthetic F ranks, with fake gate / top-k / shared-expert modules. Cases are
    (name, n_tokens, is_hash, skip_shared, return_moe_output); returns per case whether the output is exact."""
    from types import SimpleNamespace

    from sglang.test.test_utils import publish_build_topology

    from sgl_asap_plugin import attn

    publish_build_topology(tp_size=1, ep_size=1, pp_size=1)  # what post_experts_all_reduce reads (1 rank: no-op)
    dev, a, d = cfg["device"], cfg["a"], f"cuda:{cfg['device']}"
    client = attn.AttnClient(ch, dev)
    out, scale = {}, 1.5
    for idx, (name, n, is_hash, skip_shared, want_moe_output) in enumerate(cfg["cases"]):
        layer = 3 + idx
        x, ids, w = pattern(torch, g, a, idx + 1, n, dev)
        seen = {}

        def topk(h, logits, num_token_non_padded=None, expert_location_dispatch_info=None, input_ids=None):
            seen["input_ids"] = input_ids
            return SimpleNamespace(topk_ids=ids, topk_weights=w)

        experts = attn.RemoteExperts(256, 0, g.topk, g.hidden, 2048, layer)
        experts.client = client
        moe = SimpleNamespace(layer_id=layer, is_hash=is_hash, routed_scaling_factor=scale, reduce_results=True,
                              _shared_expert_tp1=False, experts=experts, gate=lambda h, alloc: h, topk=topk,
                              _adds_replicated_output=lambda: True,
                              _forward_shared_experts=lambda h, alloc: h * 0.25 + 1)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: None)
        ids_global = torch.arange(n, device=d)
        before = ch._seq[0]
        y = attn.moe_forward(None, moe, x, batch, None, None, ids_global, skip_shared, want_moe_output)
        if want_moe_output:
            y = y.get_merged()
        torch.cuda.synchronize(dev)
        ch.check()
        routed = expected(torch, g, x, ids, w, layer).to(torch.bfloat16)
        want = routed if skip_shared else (routed.float() + (x * 0.25 + 1).float()).to(torch.bfloat16)
        out[name] = {"shape_ok": tuple(y.shape) == (n, g.hidden), "equal": bool(torch.equal(y, want)),
                     "messages": ch._seq[0] - before,
                     "hash_ids_passed": (seen.get("input_ids") is ids_global) if is_hash and n else None}
    stock = SimpleNamespace(experts=torch.nn.Identity())  # a layer with local experts runs the original forward
    out["local_layer_untouched"] = attn.moe_forward(lambda *a: "original", stock, None) == "original"
    return out


def ffn_numerics(ch, torch, g, cfg, out_q, go_evt):
    """A synthetic A rank against the real F service: random layers in random order, checked against a torch
    reference of the routed experts (restricted to a pool of experts that spans every F rank)."""
    from _util import load_reference_experts, reference_moe

    dev, rng, device = cfg["device"], random.Random(cfg["seed"]), f"cuda:{cfg['device']}"
    pool, layers = cfg["pool"], cfg["layers"]
    refs = load_reference_experts(cfg["model_dir"], pool, layers, device)
    stream = torch.cuda.Stream(dev)
    s, bufs = stream.cuda_stream, _buffers(torch, g, dev)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    worst, order, bad = 0.0, [], []
    for r in range(cfg["rounds"]):
        layer, n, slot = rng.randrange(layers), rng.choice(cfg["sizes"]), r % g.slots
        order.append(layer)
        xb, ib, wb = bufs[slot]
        with torch.cuda.stream(stream):  # one stream: generation, send, reply, reference
            pick = torch.rand(n, len(pool), device=device, generator=gen).topk(g.topk, dim=1).indices
            ids = torch.tensor(pool, device=device)[pick].to(torch.int32)
            w = torch.rand(n, g.topk, device=device, generator=gen, dtype=torch.float32) + 0.1
            x = (torch.randn(n, g.hidden, device=device, generator=gen) * 0.5).to(torch.bfloat16)
            xb[:n].copy_(x)
            ib[:n].copy_(ids)
            wb[:n].copy_(w)
            ptrs = ch.wait(ch.send(layer, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s), s)
            got = sum(tview(torch, p, (n, g.hidden), torch.bfloat16, dev).float() for p in ptrs)
            want = reference_moe(refs, layer, x, ids, w)
            err = float((got - want).norm() / want.norm())
            if err > 2e-2:
                bad.append({"round": r, "layer": layer, "n": n, "err": err})
            worst = max(worst, err)
        stream.synchronize()
        ch.check()
        if cfg.get("think_us"):
            time.sleep(rng.uniform(0, cfg["think_us"]) / 1e6)
    return {"rounds": cfg["rounds"], "worst_rel_err": worst, "layers_visited": sorted(set(order)),
            "layer_order_head": order[:12], "bad": bad[:6]}


SCENARIOS = {fn.__name__: fn for fn in (pingpong, oneway, duplex, backpressure, expect_timeout,
                                         parked_wait_delays_cold_work_only, attn_forward, ffn_numerics)}
