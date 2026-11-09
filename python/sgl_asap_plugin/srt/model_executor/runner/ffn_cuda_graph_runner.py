"""CUDA graphs of the expert calls of an FFN rank (cf. ``decode_cuda_graph_runner.py``).

An FFN rank runs one ``FusedMoE`` call per job, eagerly: 0.3-0.9 ms of Python and
kernel launches per job (docs/BENCH_MOE.md), as much as the GPU time of a job of a
few thousand tokens. ``FfnCudaGraphRunner`` captures the call per (layer, token
bucket), the buckets being the powers of two up to a limit, largest first, all into
one memory pool. A job of n tokens copies its rows into the static inputs, pads them
to the bucket and replays; the output is a view of a static output buffer, valid until
the next replay of a graph that writes the same buffer. There are two buffers and each
(layer, bucket) has a graph for each, so the caller can send the output of a job while
the next job runs, by replaying the graphs of the two buffers in turn. Padding rows
route to an expert of another rank, which the experts skip; on a rank that holds every
expert they are computed with weight 0 and dropped.
"""

from __future__ import annotations

import bisect
import logging
import time
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class FfnCudaGraphRunner:
    def __init__(
        self, experts: Dict[int, torch.nn.Module], max_tokens: int, local: range
    ):
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        first = next(iter(experts.values()))
        hidden, topk, num_experts = first.hidden_size, first.top_k, first.num_experts
        self.buckets = [
            1 << i for i in range(max_tokens.bit_length()) if 1 << i < max_tokens
        ]
        self.buckets.append(max_tokens)
        self.pad_id = (
            local.stop % num_experts if len(local) < num_experts else local.start
        )
        device = torch.cuda.current_device()
        n = self.buckets[-1]
        self.x = torch.zeros(n, hidden, dtype=torch.bfloat16, device=device)
        self.ids = torch.full((n, topk), self.pad_id, dtype=torch.int32, device=device)
        self.w = torch.zeros(n, topk, dtype=torch.float32, device=device)
        self.outs = [
            torch.zeros(n, hidden, dtype=torch.bfloat16, device=device) for _ in range(2)
        ]
        self.graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}
        tic, pool = time.perf_counter(), torch.cuda.graph_pool_handle()
        reserved = torch.cuda.memory_reserved()
        stream = torch.cuda.Stream()
        for b in reversed(self.buckets):
            topk_output = StandardTopKOutput(self.w[:b], self.ids[:b], None)
            for layer_id, layer in experts.items():
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):  # first call outside the capture
                    layer(self.x[:b], topk_output)
                torch.cuda.current_stream().wait_stream(stream)
                for k, out in enumerate(self.outs):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
                        out[:b].copy_(layer(self.x[:b], topk_output))
                    self.graphs[(layer_id, b, k)] = graph
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - tic
        self.pool_gb = (torch.cuda.memory_reserved() - reserved) / 1e9
        logger.info(
            f"[ASAP] captured {len(self.graphs)} expert graphs (buckets {self.buckets}) "
            f"in {self.capture_s:.1f} s, {self.pool_gb:.2f} GB"
        )

    def run(
        self, layer_id: int, x, topk_ids, topk_weights, buffer: int = 0
    ) -> Optional[torch.Tensor]:
        """The expert output of the job's rows, in output buffer ``buffer`` (0 or 1), or
        None if the job is larger than the largest bucket (the caller runs it eagerly)."""
        n = x.shape[0]
        i = bisect.bisect_left(self.buckets, n)
        if i == len(self.buckets):
            return None
        b = self.buckets[i]
        self.x[:n].copy_(x)
        self.ids[:n].copy_(topk_ids)
        self.w[:n].copy_(topk_weights)
        if n < b:
            self.ids[n:b].fill_(self.pad_id)
            self.w[n:b].zero_()
        self.graphs[(layer_id, b, buffer)].replay()
        return self.outs[buffer][:n]
