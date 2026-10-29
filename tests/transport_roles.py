"""Child-process roles for the transport tests. Spawned (not forked), so each owns its CUDA context.

F rank f answers every row with y = (f + 1) * base(x, ids, w, layer); A sums the E replies, so a
correct sum is S * base with S = E(E+1)/2 and every payload array (x, ids, w) and the header's
layer id take part in the check. All values are small integers, exact in bf16 and fp32.
"""

from __future__ import annotations

import time
import traceback

import transport_scenarios as scenarios

SOLO = {"parked_wait_delays_cold_work_only"}  # scenarios that need no channel


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (int(ptr), False),
                                         "version": 2}


def tview(torch, ptr, count, dtype, shape, device):
    """Alias raw device memory as a tensor (owned elsewhere)."""
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    return torch.as_tensor(_CAI(ptr, nbytes), device=f"cuda:{device}").view(dtype).view(*shape)


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
    try:
        import torch

        from sgl_asap_plugin.transport import FfnServer, Geometry

        dev, g = cfg["device"], Geometry(**cfg["geometry"])
        torch.cuda.set_device(dev)
        torch.zeros(1, device=f"cuda:{dev}")
        srv = FfnServer(cfg["f"], g, device=dev, run_dir=cfg["run_dir"], seq_base=cfg.get("seq_base", 0),
                        port=cfg.get("port", 0), info=cfg.get("info")).start()
        out_q.put(("f_ready", cfg["f"], srv.port))
        go_evt.wait()
        stream, scale, ybuf, jobs = torch.cuda.Stream(dev), cfg["f"] + 1, {}, []
        echo_src = torch.zeros(g.n_max, g.hidden, dtype=torch.bfloat16, device=f"cuda:{dev}")

        def handler(job):
            for row in job.rows:
                jobs.append((row.a, job.layer_id, row.seq, row.n_tokens))
                n, nbytes = row.n_tokens, row.n_tokens * g.hidden * 2
                if cfg.get("f_mode") == "echo":  # pure transport: no compute between receive and reply
                    srv.reply(row, echo_src.data_ptr(), nbytes, stream.cuda_stream)
                    continue
                x = tview(torch, row.x_ptr, n * g.hidden, torch.bfloat16, (n, g.hidden), dev)
                ids = tview(torch, row.ids_ptr, n * g.topk, torch.int32, (n, g.topk), dev)
                w = tview(torch, row.w_ptr, n * g.topk, torch.float32, (n, g.topk), dev)
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
        out_q.put(("f_stats", cfg["f"], dict(srv.stats), list(srv.service_order), jobs))
        srv.close()
    except BaseException:
        out_q.put(("f_error", cfg["f"], traceback.format_exc()))


def a_main(cfg, out_q, go_evt=None):
    try:
        import torch

        from sgl_asap_plugin.transport import AfdChannel, Geometry

        dev = cfg["device"]
        g = Geometry(**{**cfg["geometry"], **cfg.get("geometry_override", {})})
        torch.cuda.set_device(dev)
        torch.zeros(1, device=f"cuda:{dev}")
        t0 = time.perf_counter()
        ch = AfdChannel(cfg["a"], g, [tuple(e) for e in cfg["endpoints"]], device=dev,
                        seq_base=cfg.get("seq_base", 0), timeout_s=cfg.get("timeout_s", 30.0),
                        **cfg.get("channel_kwargs", {})).connect()
        result = getattr(scenarios, cfg["scenario"])(ch, torch, g, cfg, out_q, go_evt)
        result["connect_s"] = round(time.perf_counter() - t0, 3)
        out_q.put(("a_result", cfg["a"], result))
        ch.close()
    except BaseException:
        out_q.put(("a_error", cfg["a"], traceback.format_exc()))


def solo_main(cfg, out_q):
    """Scenarios that need no channel and no F."""
    try:
        import torch

        dev = cfg["device"]
        torch.cuda.set_device(dev)
        torch.zeros(1, device=f"cuda:{dev}")
        out_q.put(("a_result", cfg["a"], getattr(scenarios, cfg["scenario"])(None, torch, None, cfg, out_q, None)))
    except BaseException:
        out_q.put(("a_error", cfg["a"], traceback.format_exc()))
