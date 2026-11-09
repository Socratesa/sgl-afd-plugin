"""The scheduler of an FFN rank, which serves the model's routed experts.

``FfnScheduler`` replaces SGLang's Scheduler class on the FFN role. It runs
``Scheduler.__init__`` up to the model worker, which loads the FFN-rank model of
``srt/models/`` (this rank's experts, EP rank f of E), and builds no KV cache,
attention backend or request path. With --asap-ffn-cuda-graph-max-tokens it captures
the expert calls in CUDA graphs (``srt/model_executor/runner/``). Its event loop
takes the jobs of all attention ranks in arrival order and answers each with the
partial output of its experts. A job is one MoE layer of one micro-batch of one
attention replica.

On an attention rank whose experts run on the FFN ranks, ``get_new_batch_prefill``
(AROUND) holds back a small prefill batch: length-aware batching.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import os
import signal
import statistics
import sys
import threading
import time

import torch
from sglang.srt.distributed import bootstrap
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.runtime_context import get_device, get_parallel

from sgl_asap_plugin.srt.entrypoints.http_server import ENV_RUN_DIR
from sgl_asap_plugin.srt.model_loader import weight_utils
from sgl_asap_plugin.srt.models import check_asap_model, register_ffn_models
from sgl_asap_plugin.srt.server_args import get_asap_args, transport_geometry

logger = logging.getLogger(__name__)

# Set to 1 to time every job (scripts/bench_moe.py).
ENV_TIMING = "ASAP_FFN_TIMING"
# Set to 0 to send the replies on the compute stream, behind the experts of the next job.
ENV_REPLY_STREAM = "ASAP_FFN_REPLY_STREAM"


def get_new_batch_prefill(original, scheduler, running_batch):
    """AROUND ``Scheduler.get_new_batch_prefill`` on an attention rank with
    --asap-min-batch-tokens: no new prefill batch while the waiting requests have
    fewer tokens to compute than that (their prompts less the prefixes the radix
    cache holds) and none has waited --asap-max-wait-ms. Every MoE layer of
    a batch is one job on every FFN rank, and a small job costs an FFN rank several
    times more per token than a big one (docs/BENCH_MOE.md), so a short wait lets
    the requests that arrive meanwhile share the jobs. The scheduler polls without
    sleeping (unless --sleep-on-idle), so it comes back here until one of the two
    holds. A batch released with more tokens than --chunked-prefill-size is cut by
    SGLang's chunked prefill, which carries the rest of the last request over."""
    from sglang.srt.managers.schedule_batch import NextBatchPlan

    args = get_asap_args()
    if hold_prefill(
        scheduler.waiting_queue,
        scheduler.tree_cache,
        scheduler.chunked_req is not None,
        args.asap_min_batch_tokens,
        args.asap_max_wait_ms / 1e3,
        time.perf_counter(),
    ):
        return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
    return original(scheduler, running_batch)


def new_tokens(req, tree_cache) -> int:
    """Tokens the prefill of ``req`` computes: its prompt less the prefix the radix
    cache holds. The scheduler matches a waiting request only when it admits it, so
    this matches once, on the first call, and keeps the result on the request; it is
    an estimate, the scheduler matches again at admission."""
    n = getattr(req, "asap_new_tokens", None)
    if n is None:
        match_kv_cache(req, tree_cache)
        cached = req.num_matched_prefix_tokens
        n = len(req.origin_input_ids) + len(req.output_ids) - cached
        req.asap_new_tokens = n
    return n


def hold_prefill(
    waiting, tree_cache, chunking: bool, min_tokens: int, max_wait_s: float, now: float
):
    """Whether to wait for more requests before the next prefill batch: never while a
    request is mid chunked prefill (``chunking``: the scheduler cut it in the last
    step, and its next chunk must run). SGLang fills that step up to the chunk size
    from the queue anyway. The running batch is not looked at: this role only prefills,
    and what it holds are requests that finished with their prefill."""
    if not min_tokens or chunking or not waiting:
        return False
    total = 0
    for req in waiting:
        total += new_tokens(req, tree_cache)
        if total >= min_tokens:
            return False
    oldest = min(req.time_stats.wait_queue_entry_time for req in waiting)
    return now - oldest < max_wait_s


class JobTimer:
    """Times of every job by token count, rounded up to a power of two: the host time
    of its launch, and the GPU time of its experts and of its replies (CUDA events,
    read once they have completed, so timing adds no synchronization). The experts
    time is kept apart for jobs that found the GPU idle ("alone": it includes the
    gaps while the host launches the kernels) and jobs queued behind another one
    ("queued": the GPU time the job costs a loaded rank)."""

    def __init__(self, keep: int = 4096):
        self.keep = keep
        self.pending = (
            collections.deque()
        )  # (bucket, alone, start, experts, replied, host_s)
        self.samples = collections.defaultdict(lambda: collections.defaultdict(list))
        self.free = []

    def event(self) -> torch.cuda.Event:
        event = self.free.pop() if self.free else torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def start(self) -> tuple:
        alone = not self.pending or self.pending[-1][4].query()
        return alone, self.event()

    def add(self, n_tokens: int, started: tuple, experts, host_s: float) -> None:
        bucket = 1 << (n_tokens - 1).bit_length()
        self.pending.append((bucket, *started, experts, self.event(), host_s))
        self.drain()

    def drain(self) -> None:
        while self.pending and self.pending[0][4].query():
            bucket, alone, start, experts, replied, host_s = self.pending.popleft()
            rows = self.samples[bucket]
            if len(rows["host_us"]) < self.keep:
                rows["host_us"].append(host_s * 1e6)
                rows["reply_ms"].append(experts.elapsed_time(replied))
                kind = "alone" if alone else "queued"
                rows[f"experts_ms_{kind}"].append(start.elapsed_time(experts))
            self.free += [start, experts, replied]

    def summary(self) -> dict:
        torch.cuda.synchronize()
        self.drain()
        out = {}
        for bucket, rows in sorted(self.samples.items()):
            out[bucket] = {"jobs": len(rows["host_us"])}
            for name, values in sorted(rows.items()):
                out[bucket][f"{name}_p50"] = round(statistics.median(values), 4)
                out[bucket][f"{name}_n"] = len(values)
        return out


class FfnScheduler(Scheduler):
    """Replaces the Scheduler class on the FFN role."""

    def __init__(self, server_args, port_args):
        # Scheduler.__init__ up to init_model_worker, which keeps only the model.
        self.init_startup_timing_begin()
        self.is_initializing = True
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.enable_priority_scheduling = False
        self.enable_lora = False
        self.enable_hierarchical_cache = False
        self.draft_worker = None
        self.init_model_config()
        bootstrap.init_parallel_runtime(
            server_args=server_args,
            device=get_device().device,
            dist_port=self.nccl_port,
        )
        bootstrap.init_layer_runtime(model_config=self.model_config)
        self.init_metrics_collector()
        self.init_moe_gemm_config()
        self.init_ffn_model()
        self.init_tp_model_worker()
        # No KV cache: this rank never schedules a request.
        self.kv_cache_allocation_time = 0.0
        self.max_total_num_tokens = 0
        self.max_req_input_len = 0
        self.init_ffn_server()
        self.init_metrics_reporter()
        self.gracefully_exit = False
        self.is_initializing = False
        self.init_startup_timing_summary()

    def init_ffn_model(self) -> None:
        """The model worker builds the FFN-rank model, which reads only this rank's
        experts."""
        check_asap_model(self.model_config.hf_config.architectures[0])
        register_ffn_models()
        weight_utils.read_routed_experts_only(
            self.local_experts(), self.model_config.hf_text_config.num_hidden_layers
        )

    def local_experts(self) -> range:
        """The routed experts of this EP rank, as FusedMoE assigns them."""
        config = self.model_config.hf_text_config
        num_experts = next(
            getattr(config, k)
            for k in ("n_routed_experts", "num_experts", "num_local_experts")
            if hasattr(config, k)
        )
        parallel = get_parallel()
        per_rank = num_experts // parallel.moe_ep_size
        return range(
            parallel.moe_ep_rank * per_rank, (parallel.moe_ep_rank + 1) * per_rank
        )

    def init_ffn_server(self) -> None:
        from sgl_asap_plugin.transport.cuda_wrapper import DeviceBuffer
        from sgl_asap_plugin.transport.server import FfnServer

        model = self.tp_worker.model_runner.model
        self.experts = {
            m.layer_id: m for m in model.modules() if isinstance(m, FusedMoE)
        }
        first, local = next(iter(self.experts.values())), self.local_experts()
        for layer in self.experts.values():
            if (
                range(
                    layer.moe_ep_rank * layer.num_local_experts,
                    (layer.moe_ep_rank + 1) * layer.num_local_experts,
                )
                != local
            ):
                raise RuntimeError(
                    f"Layer {layer.layer_id} holds other experts than {local} were read"
                )
            if layer.num_fused_shared_experts:
                raise RuntimeError("The shared experts stay on the attention ranks")
        self.f = get_parallel().moe_ep_rank
        args = get_asap_args()
        torch.cuda.empty_cache()  # the quant methods' repack replaced the loaded weights
        weights_gb = torch.cuda.memory_allocated() / 1e9
        experts_gb = sum(
            t.numel() * t.element_size()
            for layer in self.experts.values()
            for t in itertools.chain(layer.parameters(), layer.buffers())
        )
        tic = time.perf_counter()
        buckets = self.warmup(args.asap_max_dispatch_tokens)
        warmup_s = time.perf_counter() - tic
        self.graph_runner = None
        if args.asap_ffn_cuda_graph_max_tokens:
            from sgl_asap_plugin.srt.model_executor.runner.ffn_cuda_graph_runner import (
                FfnCudaGraphRunner,
            )

            self.graph_runner = FfnCudaGraphRunner(
                self.experts, args.asap_ffn_cuda_graph_max_tokens, local
            )

        self.g = transport_geometry(args, first.hidden_size, first.top_k)
        host, _, port = args.asap_bootstrap_addr.rpartition(":")
        served = {
            "experts": [local.start, local.stop],
            "quant_method": type(first.quant_method).__name__,
        }
        model_info = {
            "hidden": first.hidden_size,
            "topk": first.top_k,
            "n_experts": first.num_experts,
            "moe_layers": sorted(self.experts),
        }
        self.ffn_server = FfnServer(
            self.f,
            self.g,
            device=get_device().gpu_id,
            run_dir=os.environ[ENV_RUN_DIR],
            host=host,
            port=int(port) + self.f,
            memory=args.asap_transport_mem,
            model=model_info,
            info=served,
        ).start()
        arena = DeviceBuffer(self.ffn_server.recv_base, self.g.recv_bytes)
        self.recv_arena = torch.as_tensor(arena, device=f"cuda:{get_device().gpu_id}")
        self.init_reply_stream()
        self.layer_calls = collections.Counter()
        self.job_timer = JobTimer() if os.environ.get(ENV_TIMING) == "1" else None
        self.ready_info = {
            "f": self.f,
            **served,
            "weights_gb": round(weights_gb, 2),
            "experts_gb": round(experts_gb / 1e9, 2),
            "load_s": round(self.tp_worker.model_runner.weight_load_time, 1),
            "warmup_s": round(warmup_s, 1),
            "buckets": len(buckets),
            "graphs": len(self.graph_runner.graphs) if self.graph_runner else 0,
            **(
                {
                    "graph_capture_s": round(self.graph_runner.capture_s, 1),
                    "graph_pool_gb": round(self.graph_runner.pool_gb, 2),
                }
                if self.graph_runner
                else {}
            ),
            "port": self.ffn_server.port,
            "memory": f"{self.ffn_server.mem.kind} ({self.ffn_server.mem.reason})",
            "arena_gb": round(self.g.recv_bytes / 1e9, 2),
        }

    def init_reply_stream(self) -> None:
        """The replies (a copy over NVLink and a done flag per row) take a stream of their
        own, so that they run while the experts of the next job do. The graph runner has
        two output buffers and graph jobs use them in turn: a job's reply reads its
        buffer while the next job writes the other, and a replay of the first buffer
        again waits (``reply_free``) until the reply that read it has been sent."""
        self.reply_stream = None
        if os.environ.get(ENV_REPLY_STREAM, "1") == "0":
            return
        self.reply_stream = torch.cuda.Stream(torch.cuda.current_device())
        self.reply_ready = torch.cuda.Event()
        self.reply_free = [torch.cuda.Event(), torch.cuda.Event()]
        self.reply_turn = 0

    @torch.no_grad()
    def warmup(self, n_max: int) -> list:
        """Run every power-of-two token count up to n_max once (layer 0; every layer up
        to 256 tokens): FlashInfer tunes each new token count on its first call."""
        layers = sorted(self.experts)
        first = self.experts[layers[0]]
        device = torch.cuda.current_device()
        buckets = [1 << i for i in range(n_max.bit_length()) if 1 << i < n_max] + [
            n_max
        ]
        gen = torch.Generator(device=device).manual_seed(0)
        for n in buckets:
            x = torch.randn(
                n, first.hidden_size, device=device, dtype=torch.bfloat16, generator=gen
            )
            scores = torch.rand(n, first.num_experts, device=device, generator=gen)
            topk_ids = scores.topk(first.top_k, dim=1).indices.to(torch.int32)
            topk_weights = torch.rand(n, first.top_k, device=device, generator=gen)
            topk_output = StandardTopKOutput(topk_weights, topk_ids, None)
            for layer_id in layers[:1] if n > 256 else layers:
                self.experts[layer_id](x, topk_output)
        torch.cuda.synchronize()
        return buckets

    def get_init_info(self) -> dict:
        total_s = time.perf_counter() - self.scheduler_startup_begin
        return {
            "status": "ready",
            "max_total_num_tokens": 0,
            "max_req_input_len": 0,
            "startup_time": self.startup_time,
            "asap": {**self.ready_info, "total_s": round(total_s, 1)},
        }

    def run_event_loop(self) -> None:
        self.event_loop_ffn()

    @torch.no_grad()
    def event_loop_ffn(self) -> None:
        """Serve jobs until SIGTERM or SIGINT, then print this rank's statistics."""
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        self.ffn_server.serve_forever(self.run_job, stop)
        torch.cuda.synchronize()
        stats = {
            **self.ffn_server.stats,
            "layers_served": len(self.layer_calls),
            # (replica, slot, layer) of the last jobs
            "trace_tail": list(self.ffn_server.trace),
        }
        if self.job_timer is not None:
            stats["job_times"] = self.job_timer.summary()
        # One write: the ranks share the launcher's stdout.
        sys.stdout.write(f"[ASAP] ffn rank {self.f} stats: {json.dumps(stats)}\n")
        sys.stdout.flush()
        self.gracefully_exit = True

    def run_job(self, job) -> None:
        """One expert call for the T rows of one replica's micro-batch, then T replies.
        Runs between two polls, so it only enqueues GPU work. The experts run on the
        compute stream and the replies on ``reply_stream`` (``init_reply_stream``)."""
        tic, timer = time.perf_counter(), self.job_timer
        started = timer.start() if timer else None
        rows = [self.rows_of(row) for row in job.rows]
        if len(rows) == 1:
            x, topk_ids, topk_weights = rows[0]
        else:
            x, topk_ids, topk_weights = (torch.cat(part) for part in zip(*rows))
        output = None
        reply, buffer = self.reply_stream, 0
        if self.graph_runner is not None:
            if reply is not None:
                buffer = self.reply_turn
                if x.shape[0] <= self.graph_runner.buckets[-1]:
                    # the reply that last read this buffer, before the replay overwrites it
                    torch.cuda.current_stream().wait_event(self.reply_free[buffer])
            output = self.graph_runner.run(
                job.layer_id, x, topk_ids, topk_weights, buffer
            )
        from_graph = output is not None
        if output is None:
            topk_output = StandardTopKOutput(topk_weights, topk_ids, None)
            output = self.experts[job.layer_id](x, topk_output)
        experts = timer.event() if timer else None
        if reply is None:
            stream = torch.cuda.current_stream().cuda_stream
        else:
            if not from_graph:
                output.record_stream(reply)
            self.reply_ready.record(torch.cuda.current_stream())
            reply.wait_event(self.reply_ready)
            stream = reply.cuda_stream
        start = 0
        for row in job.rows:
            nbytes = row.n_tokens * self.g.hidden * 2
            self.ffn_server.reply(row, output[start].data_ptr(), nbytes, stream)
            start += row.n_tokens
        self.layer_calls[job.layer_id] += 1
        if reply is not None:
            if from_graph:
                self.reply_free[buffer].record(reply)
                self.reply_turn = 1 - buffer
            if timer:
                with torch.cuda.stream(reply):  # replied: when the answers are sent
                    timer.add(x.shape[0], started, experts, time.perf_counter() - tic)
        elif timer:
            timer.add(x.shape[0], started, experts, time.perf_counter() - tic)

    def rows_of(self, row):
        """x, topk_ids and topk_weights of one row: views into the receive buffer."""
        g, arena = self.g, self.recv_arena
        n, off = row.n_tokens, row.x_ptr - self.ffn_server.recv_base
        x = arena[off : off + n * g.hidden * 2].view(torch.bfloat16).view(n, g.hidden)
        ids_off, w_off = off + g.ids_off, off + g.w_off
        topk_ids = (
            arena[ids_off : ids_off + n * g.topk * 4].view(torch.int32).view(n, g.topk)
        )
        topk_weights = (
            arena[w_off : w_off + n * g.topk * 4].view(torch.float32).view(n, g.topk)
        )
        return x, topk_ids, topk_weights

    def release_host_resources(self) -> None:
        self.ffn_server.close()
