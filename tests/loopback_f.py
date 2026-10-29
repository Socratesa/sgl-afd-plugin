"""Stand-in for the F service that computes nothing: it answers every message with zeros.

It speaks the real A<->F protocol (describe, handshake, doorbells, done flags), so an A server
can be brought up and driven through a real model forward pass without the experts. Used by the
PR-04 checks (the A role alone) and handy when debugging the transport.

    python tests/loopback_f.py --ranks 1 --gpu-base 3 --port 9600 --a-ranks 1 --layers 4
"""

import argparse
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import threading
import time


def rank_main(f, args, ready_q, stop_evt):
    import torch

    from sgl_asap_plugin.transport import FfnServer, Geometry

    dev = args.gpu_base + f
    torch.cuda.set_device(dev)
    g = Geometry(num_a=args.a_ranks, group=args.group, slots=args.slots, n_max=args.n_max, hidden=args.hidden,
                 topk=args.topk, num_f=args.ranks)
    info = {"quant_method": args.quant_method, "fuse_routed_scaling_in_topk": True,
            "model": {"hidden": args.hidden, "topk": args.topk, "layers": list(range(args.layers)),
                      "n_experts": 256, "routed_scaling_factor": 1.5}}
    srv = FfnServer(f, g, device=dev, run_dir=args.run_dir, port=args.port + f, info=info).start()
    zeros = torch.zeros(g.n_max, g.hidden, dtype=torch.bfloat16, device=f"cuda:{dev}")
    stream = torch.cuda.Stream(dev)
    layers_seen = set()

    def handler(job):
        layers_seen.add(job.layer_id)
        for row in job.rows:
            srv.reply(row, zeros.data_ptr(), row.n_tokens * g.hidden * 2, stream.cuda_stream)

    ready_q.put(("ready", f, srv.port))
    srv.serve_forever(handler, stop_evt)
    torch.cuda.synchronize(dev)
    ready_q.put(("stats", f, dict(srv.stats), sorted(layers_seen)))
    srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--gpu-base", type=int, default=0)
    ap.add_argument("--port", type=int, default=9600)
    ap.add_argument("--a-ranks", type=int, default=1)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--quant-method", default="Mxfp4FlashinferCutlassMoEMethod")
    args = ap.parse_args()
    args.run_dir = tempfile.mkdtemp(prefix="asap_loopback_", dir="/dev/shm")
    ctx = mp.get_context("spawn")
    q, stop = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=rank_main, args=(f, args, q, stop), daemon=True) for f in range(args.ranks)]
    for p in procs:
        p.start()
    for _ in procs:
        print("[loopback-f] ready", q.get(timeout=300), flush=True)
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    stop.set()
    for _ in procs:
        try:
            print("[loopback-f] stats", q.get(timeout=30), flush=True)
        except Exception:
            break
    for p in procs:
        p.join(timeout=10)
    import shutil

    shutil.rmtree(args.run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
