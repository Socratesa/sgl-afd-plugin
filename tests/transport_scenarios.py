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


def afd_forward(ch, torch, g, cfg, out_q, go_evt):
    """The A-side MoE forward (model/afd_forward.py) against loopback F ranks, with fake gate/top-k/shared modules."""
    from types import SimpleNamespace

    from sgl_asap_plugin.model import afd_forward as af
    from sgl_asap_plugin.model import client as cl
    from sgl_asap_plugin.model import remote_experts as rx

    R, dev, a = _roles(), cfg["device"], cfg["a"]
    client = cl.AttnClient.__new__(cl.AttnClient)  # wrap the channel the harness already connected
    client.cfg, client.a, client.device, client.geometry, client.channel, client._views = None, a, dev, g, ch, {}
    cl.set_client(client)
    rx.set_description({"quant_method": "Mxfp4FlashinferCutlassMoEMethod", "fuse_routed_scaling_in_topk": cfg["fused"]})
    out, scale = {}, 1.5
    for name, n, valid, is_hash, skip_shared in cfg["cases"]:
        x, ids, w = R.pattern(torch, g, a, len(out) + 1, n, dev)
        layer = 3 + len(out)
        moe = SimpleNamespace(
            layer_id=layer, is_hash=is_hash, routed_scaling_factor=scale, reduce_results=True, _shared_expert_tp1=False,
            experts=rx.AfdRemoteExperts(256, 0, g.topk, g.hidden, 2048, layer),
            gate=lambda h, alloc: h, _adds_replicated_output=lambda: True,
            topk=lambda h, logits, num_token_non_padded=None, expert_location_dispatch_info=None, input_ids=None:
            SimpleNamespace(topk_ids=ids, topk_weights=w),
            _forward_shared_experts=lambda h, alloc: h * 0.25 + 1)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: None if valid == n else torch.tensor(valid, device=f"cuda:{dev}"))
        before = ch._seq[0]
        y = af.remote_moe_forward(moe, x, batch, None, torch.arange(n, device=f"cuda:{dev}") if is_hash else None,
                                  skip_shared)
        torch.cuda.synchronize(dev)
        ch.check()
        routed = R.expected(torch, g, x[:valid], ids[:valid], w[:valid], layer).to(torch.bfloat16)
        routed = torch.cat([routed, routed.new_zeros((n - valid, g.hidden))])
        alpha = 1.0 if cfg["fused"] else scale
        if skip_shared:
            want = routed if cfg["fused"] else routed * scale
        else:
            want = (x * 0.25 + 1).add(routed, alpha=alpha)
        out[name] = {"shape_ok": tuple(y.shape) == (n, g.hidden), "equal": bool(torch.equal(y, want)),
                     "messages": ch._seq[0] - before}
    return out


def ffn_numerics(ch, torch, g, cfg, out_q, go_evt):
    """A synthetic A rank against the real F service: random layers in random order, checked against a torch
    reference of the routed experts restricted to a pool of experts that spans every F rank."""
    import ffn_reference as ref

    dev, a, rng = cfg["device"], cfg["a"], random.Random(cfg["seed"])
    pool, layers, device = cfg["pool"], cfg["layers"], f"cuda:{cfg['device']}"
    refs = ref.load_reference(cfg["model_dir"], pool, layers, device)
    R = _roles()
    stream = torch.cuda.Stream(dev)
    s, bufs = stream.cuda_stream, _buffers(torch, g, dev)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    worst, order, sizes, bad = 0.0, [], [], []
    for r in range(cfg["rounds"]):
        layer, n, slot = rng.randrange(layers), rng.choice(cfg["sizes"]), r % g.slots
        order.append(layer)
        sizes.append(n)
        xb, ib, wb = bufs[slot]
        with torch.cuda.stream(stream):  # everything on one stream: generation, send, reply, reference
            pool_t = torch.tensor(pool, device=device)
            pick = torch.rand(n, len(pool), device=device, generator=gen).topk(g.topk, dim=1).indices
            ids = pool_t[pick].to(torch.int32)
            w = torch.rand(n, g.topk, device=device, generator=gen, dtype=torch.float32) + 0.1
            x = (torch.randn(n, g.hidden, device=device, generator=gen) * 0.5).to(torch.bfloat16)
            xb[:n].copy_(x)
            ib[:n].copy_(ids)
            wb[:n].copy_(w)
            ticket = ch.send(layer, slot, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), n, s)
            ptrs = ch.wait_combine(ticket, s)
            got = sum(R.tview(torch, p, n * g.hidden, torch.bfloat16, (n, g.hidden), dev).float() for p in ptrs)
            want = ref.reference_moe(refs, layer, x, ids, w)
            err = float((got - want).norm() / want.norm())
            if err > 2e-2 and cfg.get("ranges"):  # which F rank is off?
                per_rank = [float((R.tview(torch, p, n * g.hidden, torch.bfloat16, (n, g.hidden), dev).float()
                                   - ref.reference_moe(refs, layer, x, ids, w, lo=lo, hi=hi)).norm()
                                  / ref.reference_moe(refs, layer, x, ids, w, lo=lo, hi=hi).norm().clamp(min=1e-6))
                            for p, (lo, hi) in zip(ptrs, cfg["ranges"])]
                bad.append({"round": r, "layer": layer, "n": n, "err": err, "per_rank": per_rank})
            worst = max(worst, err)
        stream.synchronize()
        ch.check()
        if cfg.get("think_us"):
            time.sleep(rng.uniform(0, cfg["think_us"]) / 1e6)
    return {"rounds": cfg["rounds"], "worst_rel_err": worst, "layers_visited": sorted(set(order)),
            "layer_order_head": order[:12], "sizes": sorted(set(sizes)), "bad": bad[:6]}
