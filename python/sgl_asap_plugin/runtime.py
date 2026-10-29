"""Per-process ASAP runtime state.

The process records its identity (which replica, which rank, how big its distributed world is).
In AFD mode (``--asap-ffn-ranks``) the A<->F channel is connected here, right after the
scheduler has been built and before it serves anything.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from sgl_asap_plugin.config import AsapConfig

logger = logging.getLogger("sgl_asap_plugin")


@dataclass(frozen=True)
class AsapRuntime:
    config: AsapConfig
    dp_rank: Optional[int]
    tp_rank: Optional[int]
    tp_size: Optional[int]
    world_size: Optional[int]
    gpu_id: Optional[int]
    pid: int


_RUNTIME: Optional[AsapRuntime] = None


def _safe(read):
    try:
        return read()
    except Exception:  # ParallelContext raises RuntimeError for names not written yet
        return None


def get_runtime() -> AsapRuntime:
    if _RUNTIME is None:
        raise RuntimeError("[ASAP] runtime is not initialized in this process")
    return _RUNTIME


def init_scheduler_runtime(scheduler) -> AsapRuntime:
    global _RUNTIME
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_device, get_parallel

    from sgl_asap_plugin import compat, hooks

    config = AsapConfig.from_env()
    if config is None:
        raise RuntimeError("[ASAP] role hooks are active but ASAP_ROLE is not set in this process")
    parallel = get_parallel()
    _RUNTIME = AsapRuntime(
        config=config,
        dp_rank=_safe(lambda: parallel.dp_rank),
        tp_rank=_safe(lambda: parallel.tp_rank),
        tp_size=_safe(lambda: parallel.tp_size),
        world_size=dist.get_world_size() if dist.is_available() and dist.is_initialized() else None,
        gpu_id=_safe(lambda: get_device().gpu_id),
        pid=os.getpid(),
    )
    specs = hooks.active_specs()
    patched = len(specs) - len(compat.missing_patches(specs))
    r = _RUNTIME
    logger.info("[ASAP] scheduler ready: role=%s dp_rank=%s tp_rank=%s tp_size=%s world_size=%s gpu_id=%s "
                "pid=%d hooks=%d/%d", config.role, r.dp_rank, r.tp_rank, r.tp_size, r.world_size, r.gpu_id,
                r.pid, patched, len(specs))
    if config.ffn_ranks is not None:
        _connect_afd()
    else:
        logger.warning("[ASAP] --asap-ffn-ranks is not set: the experts stay local and this server is plain "
                       "data parallelism, not AFD")
    return _RUNTIME


def _connect_afd() -> None:
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled

    from sgl_asap_plugin.model.client import get_client

    if not cuda_graph_fully_disabled():
        raise RuntimeError("[ASAP] the AFD attention role does not support CUDA graphs (the stream waits on the F "
                           "service cannot be captured): add --disable-cuda-graph")
    get_client()
