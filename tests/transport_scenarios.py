"""A-side scenarios of the transport tests. Each returns a result dict for the test to assert on."""

from __future__ import annotations

import random
import subprocess
import sys
import time


def _roles():
    import transport_roles  # imported late: transport_roles imports this module

    return transport_roles


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
    """Strict request/response rounds with every payload array and the layer id verified."""
    R, dev, a = _roles(), cfg["device"], cfg["a"]
    stream = torch.cuda.Stream(dev)
    s = stream.cuda_stream
    bufs, sizes = _buffers(torch, g, dev), cfg.get("sizes") or [cfg.get("n", 4)]
    rng, bad, lats = random.Random(a), 0, []
    for r in range(1, cfg["rounds"] + 1):
        n, slot, layer = sizes[r % len(sizes)], r % g.slots, r % 61
        xb, ib, wb = bufs[slot]
        with torch.cuda.stream(stream):
            x, ids, w = R.pattern(torch, g, a, r, n, dev)
            xb[:n].copy_(x)
            ib[:n].copy_(ids)
            wb[:n].copy_(w)
        if cfg.get("think_us"):
            time.sleep(rng.uniform(0, cfg["think_us"]) / 1e6)
        stream.synchronize()  # data generation must not count as transport time
        t0 = time.perf_counter()
        ticket = ch.send(layer, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s)
        ptrs = ch.wait_combine(ticket, s)
        if cfg.get("verify", True):
            with torch.cuda.stream(stream):
                got = sum(R.tview(torch, p, n * g.hidden, torch.bfloat16, (n, g.hidden), dev).float() for p in ptrs)
                bad += 0 if torch.equal(got, R.expected(torch, g, x, ids, w, layer)) else 1
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
        ch.wait_combine(ticket, s)  # the reply frees the slot for the next round
        e1.synchronize()
        ms.append(e0.elapsed_time(e1))
    stream.synchronize()
    ch.check()
    sent = n * (g.hidden * 2 + 2 * g.topk * 4)
    return {"gbps": sent / (sorted(ms)[len(ms) // 2] * 1e-3) / 1e9}


def duplex(ch, torch, g, cfg, out_q, go_evt):
    """One message per slot in flight, so A's next send overlaps F's reply to the previous one on the reverse link."""
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
            ch.wait_combine(pending.pop(0), s)
    for ticket in pending:
        ch.wait_combine(ticket, s)
    e1.record(stream)
    e1.synchronize()
    ch.check()
    ms = e0.elapsed_time(e1)
    moved = cfg["rounds"] * n * (g.hidden * 2 + 2 * g.topk * 4 + g.hidden * 2)  # A to F plus the reply back
    return {"gbps": moved / (ms * 1e-3) / 1e9, "us_per_message": ms * 1e3 / cfg["rounds"]}


def backpressure(ch, torch, g, cfg, out_q, go_evt):
    """F is held back by the test. A sends twice into the same slot: the second send must stay blocked."""
    R, dev = _roles(), cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, n, msgs = stream.cuda_stream, 4, []
    for r in (1, 2):
        with torch.cuda.stream(stream):
            msgs.append([t.contiguous() for t in R.pattern(torch, g, cfg["a"], r, n, dev)])
    torch.cuda.synchronize(dev)
    tickets = [ch.send(r, 0, m[0].data_ptr(), m[1].data_ptr(), m[2].data_ptr(), n, s)
               for r, m in zip((1, 2), msgs)]
    time.sleep(cfg["hold_s"])
    state = [t.send_phase.state for t in tickets]
    go_evt.set()
    for t in tickets:
        ch.wait_combine(t, s)
    stream.synchronize()
    ch.check()
    return {"send_states_while_f_held": state}


def expect_timeout(ch, torch, g, cfg, out_q, go_evt):
    """One message that is never answered (F dead or dropping it): the wait must fail, not hang."""
    R, dev = _roles(), cfg["device"]
    stream = torch.cuda.Stream(dev)
    s, n = stream.cuda_stream, 4
    with torch.cuda.stream(stream):
        x, ids, w = [t.contiguous() for t in R.pattern(torch, g, cfg["a"], 1, n, dev)]
    torch.cuda.synchronize(dev)
    ticket = ch.send(1, 0, x.data_ptr(), ids.data_ptr(), w.data_ptr(), n, s)
    ch.wait_combine(ticket, s)
    out_q.put(("a_waiting", cfg["a"]))
    t0 = time.perf_counter()
    stream.synchronize()  # returns only because the watchdog released the stream
    out = {"waited_s": round(time.perf_counter() - t0, 2), "check": _raises(ch.check)}
    out["send_after"] = _raises(lambda: ch.send(2, 1, x.data_ptr(), ids.data_ptr(), w.data_ptr(), n, s))
    if cfg.get("recover"):  # reconnect (new epoch, fresh result arena), then the channel must work again
        ch._epoch_override = None
        ch.reconnect()
        out["after_reconnect"] = pingpong(ch, torch, g, dict(cfg, rounds=20), out_q, go_evt)
    return out


_RELEASER = """
import sys, time
from sgl_asap_plugin.transport import cuda as cu, streams as st
cu.Device(int(sys.argv[3])).activate()
flag = cu.ipc_open(bytes.fromhex(sys.argv[1]))
stream = st.stream_create()
time.sleep(float(sys.argv[2]))
st.write32(stream, flag, 1)
st.stream_sync(stream)
"""


def parked_wait_delays_cold_work_only(ch, torch, g, cfg, out_q, go_evt):
    """A stream parked in a memop wait delays cold work (first-use kernels, cudaMalloc) until the peer's write
    arrives, and leaves warm work alone. The release comes from another process, as it does from F."""
    from sgl_asap_plugin.transport import cuda as cu
    from sgl_asap_plugin.transport import streams as st

    dev, delay = cfg["device"], 2.0
    cu.Device(dev).activate()
    flag = cu.mem_alloc(256)
    handle, work = cu.ipc_export(flag).hex(), torch.cuda.Stream(dev)

    def op():
        x = torch.arange(4096.0, device=f"cuda:{dev}")
        return torch.linalg.vector_norm(x.cumsum(0).sin())

    def timed():
        parked = st.stream_create()
        cu.memset32(flag, 0, 64)
        st.wait32(parked, flag, 1)
        helper = subprocess.Popen([sys.executable, "-c", _RELEASER, handle, str(delay), str(cfg["release_device"])])
        t0 = time.perf_counter()
        with torch.cuda.stream(work):
            op()
        work.synchronize()
        elapsed = time.perf_counter() - t0
        helper.wait()
        st.stream_sync(parked)
        return round(elapsed, 3)

    cold, warm = timed(), timed()
    return {"delay_s": delay, "cold_s": cold, "warm_s": warm}
