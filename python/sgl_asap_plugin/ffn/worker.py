"""One F rank: build the experts, load them, serve A's messages until told to stop."""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Dict, Tuple

import torch

from sgl_asap_plugin import layout
from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.model.tensors import tensor_at
from sgl_asap_plugin.transport.arena import Geometry
from sgl_asap_plugin.transport.server import FfnServer, Job

logger = logging.getLogger("sgl_asap_plugin")


class Servicer:
    """Turns a complete job (the T rows of one A replica for one layer) into one expert call and the replies."""

    def __init__(self, server: FfnServer, stack, geometry: Geometry, device: int):
        self.server, self.stack, self.g, self.device = server, stack, geometry, device
        self._full: Dict[Tuple[int, str], torch.Tensor] = {}
        self.layer_calls = [0] * stack.num_layers

    def _rows(self, ptr: int, kind: str, n: int) -> torch.Tensor:
        """First n rows of the region at ptr; the full-size view is built once per region."""
        view = self._full.get((ptr, kind))
        if view is None:
            g = self.g
            dtype, cols = {"x": (torch.bfloat16, g.hidden), "ids": (torch.int32, g.topk),
                           "w": (torch.float32, g.topk)}[kind]
            view = self._full[(ptr, kind)] = tensor_at(ptr, dtype, (g.n_max, cols), self.device)
        return view[:n]

    def __call__(self, job: Job) -> None:
        rows, hidden = job.rows, self.g.hidden
        parts = [(self._rows(r.x_ptr, "x", r.n_tokens), self._rows(r.ids_ptr, "ids", r.n_tokens),
                  self._rows(r.w_ptr, "w", r.n_tokens)) for r in rows]
        if len(parts) == 1:
            x, ids, w = parts[0]
        else:
            x, ids, w = (torch.cat([p[i] for p in parts]) for i in range(3))
        y = self.stack.forward(job.layer_id, x, ids, w)
        stream = torch.cuda.current_stream().cuda_stream
        offset = 0
        for r in rows:
            self.server.reply(r, y[offset].data_ptr(), r.n_tokens * hidden * 2, stream)
            offset += r.n_tokens
        self.layer_calls[job.layer_id] += 1


def worker_main(ns, f: int, gpu_id: int, run_dir: str, conn, stop_evt) -> None:
    """Process target. Reports ("ready", ...) or ("error", ...) on ``conn``."""
    try:
        logging.basicConfig(level=logging.INFO, format=f"[%(asctime)s F{f}] %(message)s")
        from sglang.srt.server_args import ServerArgs

        from sgl_asap_plugin.ffn.experts import ExpertStack, init_f_process

        t_start = time.time()
        asap = AsapConfig.from_namespace(ns)
        server_args = ServerArgs.from_cli_args(ns)
        init_f_process(server_args, f, gpu_id)
        from sglang.srt.arg_groups.overrides import resolving_view

        view = resolving_view(server_args)
        stack_probe = layout.read_moe_shape(view.model_path, view.json_model_override_args)
        partition = layout.partition_experts(stack_probe.n_experts, view.ep_size)
        stack = ExpertStack(server_args, partition, f, gpu_id, layers=stack_probe.layers)
        load = stack.load(workers=int(os.environ.get("ASAP_FFN_LOAD_THREADS", "4")))
        geometry = Geometry(num_a=asap.num_replicas * asap.replica_tp, group=asap.replica_tp, slots=asap.slots,
                            n_max=asap.n_max, hidden=stack_probe.hidden, topk=stack_probe.topk, num_f=view.ep_size)
        t_warm = time.time()
        buckets = stack.warmup(asap.n_max)
        host, _, port = asap.rdv_addr.rpartition(":")
        server = FfnServer(f, geometry, device=gpu_id, run_dir=run_dir, host=host, port=int(port) + f,
                           info=stack.describe()).start()
        servicer = Servicer(server, stack, geometry, gpu_id)
        conn.send(("ready", f, server.port, {
            "weights_gb": round(load.weight_bytes / 1e9, 2), "load_s": round(load.seconds, 1),
            "repack_s": round(load.post_seconds, 1), "warmup_s": round(time.time() - t_warm, 1),
            "total_s": round(time.time() - t_start, 1), "buckets": len(buckets),
            "experts": [stack.lo, stack.hi], "quant_method": stack.quant_method,
            "arena_gb": round(geometry.recv_bytes / 1e9, 2)}))
        server.serve_forever(servicer, stop_evt)
        torch.cuda.synchronize(gpu_id)
        conn.send(("stats", f, dict(server.stats), servicer.layer_calls))
        server.close()
    except BaseException:
        conn.send(("error", f, traceback.format_exc()))
        raise
