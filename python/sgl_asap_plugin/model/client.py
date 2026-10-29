"""A-side client of the A<->F transport: one per A rank, i.e. per scheduler process.

It owns the ``AfdChannel`` and turns the raw device pointers of the F replies into tensors.
Environment knobs (seconds): ASAP_CONNECT_TIMEOUT_S, how long to wait for the F service to
come up (default 1800, F loads tens of GB first); ASAP_WAIT_TIMEOUT_S, how long one layer's
wait for F may take before the channel is declared dead (default 60).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.transport.arena import Geometry
from sgl_asap_plugin.transport.channel import AfdChannel
from sgl_asap_plugin.transport.handshake import request

logger = logging.getLogger("sgl_asap_plugin")

ENV_CONNECT_TIMEOUT = "ASAP_CONNECT_TIMEOUT_S"
ENV_WAIT_TIMEOUT = "ASAP_WAIT_TIMEOUT_S"
SLOT = 0  # the synchronous forward keeps one micro-batch in flight


def endpoints_of(cfg: AsapConfig) -> List[Tuple[str, int]]:
    """F rank f listens on the --asap-rdv-addr port + f."""
    host, _, port = cfg.rdv_addr.rpartition(":")
    return [(host, int(port) + f) for f in range(cfg.ffn_ranks)]


def describe_ffn(cfg: Optional[AsapConfig] = None) -> Dict:
    """Ask every F rank what it serves (waits for it to come up) and check that they agree."""
    cfg = cfg or AsapConfig.from_env()
    if cfg is None or cfg.ffn_ranks is None:
        raise AsapConfigError("[ASAP] --asap-ffn-ranks E is required to talk to the F service")
    timeout = float(os.environ.get(ENV_CONNECT_TIMEOUT, "1800"))
    descs = [request(host, port, {"op": "describe"}, connect_timeout=timeout) for host, port in endpoints_of(cfg)]
    first = descs[0]
    for f, d in enumerate(descs):
        same = all(d[k] == first[k] for k in ("quant_method", "fuse_routed_scaling_in_topk", "model", "geometry"))
        if d["num_f"] != cfg.ffn_ranks or d["f"] != f or not same:
            raise AsapConfigError(f"[ASAP] F rank {f} at {endpoints_of(cfg)[f]} disagrees with rank 0 or with "
                                  f"--asap-ffn-ranks {cfg.ffn_ranks}: {d}")
    return first


def check_model(desc: Dict, dims: Dict) -> None:
    """The F service must hold the experts of exactly the model this A rank runs."""
    m, problems = desc["model"], []
    for key in ("hidden", "topk"):
        if m[key] != dims[key]:
            problems.append(f"{key}: A={dims[key]} F={m[key]}")
    if list(m["layers"]) != list(dims["layers"]):
        problems.append(f"MoE layers: A={dims['layers']} F={m['layers']}")
    if problems:
        raise AsapConfigError("[ASAP] the A model and the F service do not match: " + "; ".join(problems))


class AttnClient:
    def __init__(self, cfg: AsapConfig, a_rank: int, device: int, hidden: int, topk: int):
        if cfg.ffn_ranks is None or cfg.num_replicas is None or cfg.replica_tp is None:
            raise AsapConfigError("[ASAP] --asap-ffn-ranks, --asap-num-replicas and --asap-replica-tp are required "
                                  "on the attention role in AFD mode")
        self.cfg, self.a, self.device = cfg, a_rank, device
        self.geometry = Geometry(num_a=cfg.num_replicas * cfg.replica_tp, group=cfg.replica_tp, slots=cfg.slots,
                                 n_max=cfg.n_max, hidden=hidden, topk=topk, num_f=cfg.ffn_ranks)
        self.channel: Optional[AfdChannel] = None
        self._views: Dict[int, torch.Tensor] = {}

    def connect(self) -> "AttnClient":
        self.channel = AfdChannel(
            self.a, self.geometry, endpoints_of(self.cfg), device=self.device,
            timeout_s=float(os.environ.get(ENV_WAIT_TIMEOUT, "60")),
            connect_timeout_s=float(os.environ.get(ENV_CONNECT_TIMEOUT, "1800"))).connect()
        logger.info("[ASAP] A rank %d connected to %d F rank(s) (n_max=%d, slots=%d)", self.a,
                    self.geometry.num_f, self.geometry.n_max, self.geometry.slots)
        return self

    def close(self) -> None:
        if self.channel is not None:
            torch.cuda.synchronize(self.device)
            self.channel.close()
            self.channel = None

    # ---- one layer ----
    def send(self, layer_id: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, n: int):
        """Enqueue the message of one layer on the current stream. x, ids, w must stay alive (stream order)."""
        g = self.geometry
        if n > g.n_max:
            raise RuntimeError(f"[ASAP] {n} tokens in one MoE layer exceed --asap-n-max {g.n_max}; raise it or lower "
                               "--chunked-prefill-size / --max-prefill-tokens")
        if x.dtype != torch.bfloat16 or ids.dtype != torch.int32 or w.dtype != torch.float32:
            raise RuntimeError(f"[ASAP] expected bf16 activations, int32 ids and fp32 weights, got "
                               f"{x.dtype}/{ids.dtype}/{w.dtype}")
        if not (x.is_contiguous() and ids.is_contiguous() and w.is_contiguous()):
            raise RuntimeError("[ASAP] send() needs contiguous tensors")
        return self.channel.send(layer_id, SLOT, x.data_ptr(), ids.data_ptr(), w.data_ptr(), n,
                                 torch.cuda.current_stream().cuda_stream)

    def combine(self, ticket, n: int) -> torch.Tensor:
        """Enqueue the wait for every F rank and return sum_f partial_f for the first n rows (a fresh tensor)."""
        ptrs = self.channel.wait_combine(ticket, torch.cuda.current_stream().cuda_stream)
        parts = [self._view(p)[:n] for p in ptrs]
        if len(parts) == 1:
            return parts[0].clone()  # the arena slot is overwritten by the next reply
        acc = parts[0].float()
        for p in parts[1:]:
            acc.add_(p)
        return acc.to(torch.bfloat16)

    def check(self) -> None:
        self.channel.check()

    def _view(self, ptr: int) -> torch.Tensor:
        t = self._views.get(ptr)
        if t is None:
            g = self.geometry

            class _Mem:  # CUDA array interface over memory owned by the channel
                __cuda_array_interface__ = {"shape": (g.n_max * g.hidden * 2,), "typestr": "|u1",
                                            "data": (int(ptr), False), "version": 2}

            t = torch.as_tensor(_Mem(), device=f"cuda:{self.device}").view(torch.bfloat16).view(g.n_max, g.hidden)
            self._views[ptr] = t
        return t


_CLIENT: Optional[AttnClient] = None


def set_client(client: Optional[AttnClient]) -> None:
    global _CLIENT
    _CLIENT = client


def get_client() -> AttnClient:
    """The process's A client; connects on first use (normally done right after the scheduler is built)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = connect_for_this_process()
    return _CLIENT


def connect_for_this_process() -> AttnClient:
    from sgl_asap_plugin.model.remote_experts import ffn_description, model_dims

    cfg = AsapConfig.from_env()
    if cfg is None or cfg.role != "attn":
        raise AsapConfigError("[ASAP] this process is not an attention-role process")
    dims, desc = model_dims(), ffn_description()
    check_model(desc, dims)
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    a = (parallel.dp_rank or 0) * cfg.replica_tp + parallel.tp_rank
    return AttnClient(cfg, a, torch.cuda.current_device(), dims["hidden"], dims["topk"]).connect()
