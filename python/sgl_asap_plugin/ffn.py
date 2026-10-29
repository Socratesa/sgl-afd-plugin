"""The expert (F) role: ``python -m sgl_asap_plugin.ffn <sglang serve flags> --asap-role ffn --asap-*``.

The launcher resolves the flags through sglang's own ServerArgs pipeline, checks the deployment rules
(config.check_ffn) and starts one process per F rank, rank f on GPU ``--base-gpu-id + f * --gpu-id-step``.
Rank f
  1. publishes its place in sglang's runtime context: EP rank f of E, no process groups (ranks never talk);
  2. builds one stock FusedMoE per MoE layer with the arguments DeepseekV2MoE passes, so the quantization method
     (flashinfer MXFP4 W4A16 on Hopper), the weight layout and the kernels are those of the colocated model;
  3. reads only its own experts' tensors (checkpoint.expert_selector), then runs the method's post-load repack;
  4. runs every token-count bucket once, because FlashInfer tunes a bucket on its first call;
  5. opens the transport (FfnServer) and serves jobs in arrival order until SIGTERM.
The launcher prints "[ASAP] ffn ready" once every rank listens.

Per job (one layer of one micro-batch of one A replica) rank f computes
    y = sum over k with ids[k] owned by f of  w[k] * expert_ids[k](x)
with the weights A sent. They are final: A's TopK already applied routed_scaling_factor, so the FusedMoE here is
built without one. The forward calls dispatcher.dispatch -> run_moe_core -> dispatcher.combine directly;
FusedMoE.forward would need a process group and a forward context, and would all-reduce.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from typing import Dict, List

from sgl_asap_plugin import checkpoint, config
from sgl_asap_plugin.config import AsapConfigError

logger = logging.getLogger("sgl_asap_plugin")

# remapped (sglang) name of a routed-expert tensor -> layer, the mapping key "experts.<E>.<proj>.", suffix
_EXPERT_NAME = re.compile(r"^model\.layers\.(\d+)\.mlp\.(experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.)"
                          r"(weight|weight_scale_inv)$")


class _NoGroup:
    """Stands in for the TP process group: F ranks never communicate, and the only reader of the group on the
    path used here is the symmetric-memory switch, which stays off."""


def init_rank(server_args, f: int, gpu_id: int) -> None:
    """Publish this process as EP rank f (of --ep-size) and seed the global kernel configs FusedMoE reads."""
    import torch
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish

    torch.cuda.set_device(gpu_id)
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=f, gpu_id=gpu_id))
    get_parallel().override_permanently(tp_group=_NoGroup())
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_bf16_gemm_config()
    initialize_fp4_gemm_config()


class ExpertStack:
    """The routed experts of F rank f: one FusedMoE per layer holding logical experts [lo, hi)."""

    def __init__(self, server_args, f: int, ffn_ranks: int, gpu_id: int):
        import torch
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.utils import RoutingMethodType
        from sglang.srt.model_loader.loader import _get_quantization_config

        self.model_config = ModelConfig.from_server_args(server_args)
        hf = self.hf = self.model_config.hf_config
        self.f, self.gpu_id, self.num_layers = f, gpu_id, hf.num_hidden_layers
        per_rank = config.experts_per_rank(hf.n_routed_experts, ffn_ranks)
        self.lo, self.hi = min(f * per_rank, hf.n_routed_experts), min((f + 1) * per_rank, hf.n_routed_experts)
        quant_config = _get_quantization_config(self.model_config, LoadConfig(load_format="auto", tp_rank=f))
        t0 = time.time()
        with torch.device(f"cuda:{gpu_id}"):  # the arguments of DeepseekV2MoE.__init__ (deepseek_v2.py)
            self.layers = [FusedMoE(
                num_experts=per_rank * ffn_ranks, num_fused_shared_experts=0, top_k=hf.num_experts_per_tok,
                hidden_size=hf.hidden_size, intermediate_size=hf.moe_intermediate_size, layer_id=i,
                quant_config=quant_config, routed_scaling_factor=None,  # A's weights are final (module docstring)
                routing_method_type=getattr(hf, "routing_method_type", RoutingMethodType.DeepSeekV3),
                swiglu_limit=getattr(hf, "swiglu_limit", None), prefix=f"model.layers.{i}.mlp.experts")
                for i in range(self.num_layers)]
        if self.layers[0].num_local_experts != per_rank:
            raise RuntimeError(f"[ASAP] FusedMoE kept {self.layers[0].num_local_experts} local experts, expected "
                               f"{per_rank}: the published EP size is not --ep-size {ffn_ranks}?")
        logger.info("[ASAP] F rank %d built %d layers of experts [%d, %d) in %.1fs (%s)", f, self.num_layers,
                    self.lo, self.hi, time.time() - t0, self.quant_method)

    @property
    def quant_method(self) -> str:
        return type(self.layers[0].quant_method).__name__

    def fingerprint(self) -> dict:
        """What an A rank must run to be served here; compared field by field in the handshake."""
        hf = self.hf
        return {"hidden": hf.hidden_size, "topk": hf.num_experts_per_tok, "n_experts": hf.n_routed_experts,
                "moe_layers": list(range(self.num_layers))}

    def load(self, workers: int = 4) -> Dict[str, float]:
        """Read this rank's expert tensors, hand them to FusedMoE's weight loaders, then repack."""
        import torch
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        mapping = {weight_name: (param_name, expert_id, shard_id) for param_name, weight_name, expert_id, shard_id
                   in FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name="gate_proj", ckpt_down_proj_name="down_proj",
                                                          ckpt_up_proj_name="up_proj", num_experts=self.hi)}
        params = {f"model.layers.{i}.mlp.{name}": p for i, layer in enumerate(self.layers)
                  for name, p in layer.named_parameters(prefix="experts")}
        keep = checkpoint.expert_selector(self.num_layers, self.lo, self.hi)
        files, expected = checkpoint.select_files(self.model_config.model_path, keep)
        want = self.num_layers * (self.hi - self.lo) * 6  # w1, w2, w3, each with its scale
        if expected != want:
            raise RuntimeError(f"[ASAP] the checkpoint index lists {expected} tensors for experts [{self.lo}, "
                               f"{self.hi}) of {self.num_layers} layers, expected {want}")
        t0, loaded = time.time(), 0
        for raw, tensor in checkpoint.filtered_safetensors_iterator(files, keep, max_workers=workers):
            name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                raw, is_nextn=False, num_hidden_layers=self.num_layers)
            m = _EXPERT_NAME.match(name)
            if m is None:
                raise RuntimeError(f"[ASAP] unexpected expert tensor {raw!r} (remapped to {name!r})")
            param_name, expert_id, shard_id = mapping[m.group(2)]
            key = f"model.layers.{m.group(1)}.mlp.{param_name}{m.group(3)}"
            params[key].weight_loader(params[key], tensor, key, shard_id=shard_id, expert_id=expert_id)
            loaded += 1
        if loaded != want:
            raise RuntimeError(f"[ASAP] loaded {loaded} expert tensors, expected {want}")
        torch.cuda.synchronize(self.gpu_id)
        t1 = time.time()
        del params  # the repack replaces the parameters; drop our references so the originals can be freed
        for layer in self.layers:
            layer.quant_method.process_weights_after_loading(layer)
        torch.cuda.synchronize(self.gpu_id)
        torch.cuda.empty_cache()
        stats = {"tensors": loaded, "load_s": round(t1 - t0, 1), "repack_s": round(time.time() - t1, 1),
                 "weights_gb": round(torch.cuda.memory_allocated(self.gpu_id) / 1e9, 2)}
        logger.info("[ASAP] F rank %d loaded %s", self.f, stats)
        return stats

    def forward(self, layer_id: int, x, ids, w):
        """Partial output of this rank's experts for rows x; ids are global logical ids, non-local ones add 0."""
        import torch
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        layer = self.layers[layer_id]
        with torch.no_grad():
            dispatched = layer.dispatcher.dispatch(
                hidden_states=x, topk_output=StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=None))
            y = layer.dispatcher.combine(combine_input=layer.run_moe_core(dispatch_output=dispatched))
        return y[..., : x.shape[-1]]  # some methods pad hidden to a multiple of 256

    def warmup(self, n_max: int) -> List[int]:
        """Call every power-of-two token bucket up to n_max once (layer 0; all layers for buckets <= 256), so no
        request pays a kernel's first-call tuning or compilation."""
        import torch

        hf, device = self.hf, f"cuda:{self.gpu_id}"
        buckets = [1 << i for i in range(max(1, n_max).bit_length()) if 1 << i < n_max] + [n_max]
        gen = torch.Generator(device=device).manual_seed(0)
        for n in buckets:
            x = torch.randn(n, hf.hidden_size, device=device, dtype=torch.bfloat16, generator=gen)
            ids = torch.rand(n, hf.n_routed_experts, device=device, generator=gen).topk(
                hf.num_experts_per_tok, dim=1).indices.to(torch.int32)
            w = torch.rand(n, hf.num_experts_per_tok, device=device, generator=gen)
            for layer_id in ([0] if n > 256 else range(self.num_layers)):
                self.forward(layer_id, x, ids, w)
        torch.cuda.synchronize(self.gpu_id)
        return buckets


def _rank_main(ns, f: int, gpu_id: int, run_dir: str, conn, stop) -> None:
    """Process body of F rank f. Reports ("ready", f, info) / ("error", f, traceback), then ("stats", f, ...)."""
    try:
        logging.basicConfig(level=logging.INFO, format=f"[%(asctime)s F{f}] %(message)s")
        import torch
        from sglang.srt.server_args import ServerArgs

        from sgl_asap_plugin.transport.cuda import DeviceBuffer
        from sgl_asap_plugin.transport.server import FfnServer

        t_start = time.time()
        server_args = ServerArgs.from_cli_args(ns)
        cfg = config.AsapConfig.from_namespace(ns).with_defaults(ffn_ranks=server_args.ep_size)
        init_rank(server_args, f, gpu_id)
        stack = ExpertStack(server_args, f, cfg.ffn_ranks, gpu_id)
        load = stack.load(workers=int(os.environ.get("ASAP_FFN_LOAD_THREADS", "4")))
        t_warm = time.time()
        buckets = stack.warmup(cfg.n_max)
        hf = stack.hf
        g = cfg.geometry(hf.hidden_size, hf.num_experts_per_tok)
        host, _, port = cfg.rdv_addr.rpartition(":")
        served = {"experts": [stack.lo, stack.hi], "quant_method": stack.quant_method}
        server = FfnServer(f, g, device=gpu_id, run_dir=run_dir, host=host, port=int(port) + f,
                           model=stack.fingerprint(), info=served).start()
        arena = torch.as_tensor(DeviceBuffer(server.recv_base, g.recv_bytes), device=f"cuda:{gpu_id}")
        stream = torch.cuda.current_stream().cuda_stream
        calls = [0] * stack.num_layers

        def rows_of(row):
            """x, ids, w of one row: views into the RecvArena (layout.py), no copy."""
            off, n = row.x_ptr - server.recv_base, row.n_tokens
            x = arena[off: off + n * g.hidden * 2].view(torch.bfloat16).view(n, g.hidden)
            ids = arena[off + g.ids_off: off + g.ids_off + n * g.topk * 4].view(torch.int32).view(n, g.topk)
            w = arena[off + g.w_off: off + g.w_off + n * g.topk * 4].view(torch.float32).view(n, g.topk)
            return x, ids, w

        def handle(job):
            """One job = the T rows of one replica's micro-batch for one layer: one expert call, T replies."""
            parts = [rows_of(r) for r in job.rows]
            x, ids, w = parts[0] if len(parts) == 1 else [torch.cat(p) for p in zip(*parts)]
            y = stack.forward(job.layer_id, x, ids, w).contiguous()
            start = 0
            for r in job.rows:
                server.reply(r, y[start].data_ptr(), r.n_tokens * g.hidden * 2, stream)
                start += r.n_tokens
            calls[job.layer_id] += 1

        conn.send(("ready", f, {**served, **load, "port": server.port, "buckets": len(buckets),
                                "warmup_s": round(time.time() - t_warm, 1), "total_s": round(time.time() - t_start, 1),
                                "arena_gb": round(g.recv_bytes / 1e9, 2)}))
        server.serve_forever(handle, stop)
        torch.cuda.synchronize(gpu_id)
        conn.send(("stats", f, {**server.stats, "layers_served": sum(1 for c in calls if c),
                                "trace_tail": list(server.trace)}))  # (replica, slot, layer) of the last jobs
        server.close()
    except BaseException:
        conn.send(("error", f, traceback.format_exc()))
        raise


def build_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(prog="python -m sgl_asap_plugin.ffn")
    ServerArgs.add_cli_args(parser)
    config.add_cli_args(parser)
    parser.add_argument("--asap-dry-run", action="store_true", help="Validate, print the layout and exit.")
    return parser


def plan(ns) -> dict:
    """Resolve and validate the launch; return the static layout (expert ranges, arena sizes)."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.server_args import ServerArgs

    ns.asap_role = ns.asap_role or "ffn"
    cfg = config.AsapConfig.from_namespace(ns)
    if cfg.role != "ffn":
        raise AsapConfigError("[ASAP] this entry point only runs the ffn role")
    view = resolving_view(ServerArgs.from_cli_args(ns))
    config.check_ffn(view, cfg)
    cfg = cfg.with_defaults(ffn_ranks=view.ep_size)
    shape = config.read_moe_shape(view.model_path, view.json_model_override_args)
    per_rank = config.experts_per_rank(shape.n_experts, cfg.ffn_ranks)
    g = cfg.geometry(shape.hidden, shape.topk)
    return {"ffn_ranks": cfg.ffn_ranks, "n_experts": shape.n_experts, "experts_per_rank": per_rank,
            "expert_ranges": [[min(f * per_rank, shape.n_experts), min((f + 1) * per_rank, shape.n_experts)]
                              for f in range(cfg.ffn_ranks)],
            "layers": shape.layers, "rdv_addr": cfg.rdv_addr, "geometry": g.as_dict(),
            "f_recv_arena_bytes": g.recv_bytes, "a_result_arena_bytes": g.result_bytes}


def serve(ns, layout: dict) -> int:
    """Start the F ranks, wait until all are ready, supervise them until SIGTERM / SIGINT; the exit status is 0
    only if every rank came up and none died."""
    import multiprocessing as mp
    import shutil
    import tempfile

    ctx = mp.get_context("spawn")
    ranks, base, step = layout["ffn_ranks"], ns.base_gpu_id or 0, ns.gpu_id_step or 1
    run_dir = tempfile.mkdtemp(prefix="asap_ffn_", dir="/dev/shm")  # the slot files; A ranks mmap them
    stop, quit_requested = ctx.Event(), threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: quit_requested.set())
    workers = []
    for f in range(ranks):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_rank_main, args=(ns, f, base + f * step, run_dir, child, stop), name=f"asap-ffn-{f}")
        proc.start()
        workers.append((proc, parent))

    def report(f, conn):
        """Print every pending message of rank f; False if it failed."""
        ok = True
        try:
            while conn.poll():
                kind, rank, payload = conn.recv()
                if kind == "ready":
                    ready.add(rank)
                ok &= kind != "error"
                text = payload if kind == "error" else json.dumps(payload, sort_keys=True)
                print(f"[ASAP] ffn rank {rank} {kind}: {text}", file=sys.stderr if kind == "error" else sys.stdout,
                      flush=True)
        except (EOFError, OSError):
            ok = False
        return ok

    ready, status = set(), 0
    deadline = time.time() + float(os.environ.get("ASAP_FFN_START_TIMEOUT_S", "3600"))
    while not quit_requested.is_set() and status == 0:
        for f, (proc, conn) in enumerate(workers):
            if not report(f, conn) or (not proc.is_alive() and f not in ready):
                print(f"[ASAP] ffn rank {f} failed to start (exit code {proc.exitcode})", file=sys.stderr, flush=True)
                status = 1
        if len(ready) == ranks:
            break
        if time.time() > deadline:
            print("[ASAP] ffn ranks did not come up in time", file=sys.stderr, flush=True)
            status = 1
        time.sleep(0.2)
    if status == 0 and len(ready) == ranks:
        print(f"[ASAP] ffn ready: ranks={ranks} rdv={layout['rdv_addr']}+rank "
              f"experts_per_rank={layout['experts_per_rank']} layers={layout['layers']}", flush=True)
        while not quit_requested.wait(0.5):
            if any(not p.is_alive() for p, _ in workers):
                print("[ASAP] an ffn rank exited; shutting the service down", file=sys.stderr, flush=True)
                status = 1
                break
    stop.set()
    for f, (proc, conn) in enumerate(workers):
        proc.join(timeout=30)
        report(f, conn)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    shutil.rmtree(run_dir, ignore_errors=True)
    print("[ASAP] ffn stopped", flush=True)
    return status


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        layout = plan(ns)
    except AsapConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    print("[ASAP] ffn layout " + json.dumps(layout, sort_keys=True), flush=True)
    return 0 if ns.asap_dry_run else serve(ns, layout)


if __name__ == "__main__":
    sys.exit(main())
