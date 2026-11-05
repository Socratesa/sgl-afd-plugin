"""Hooks on ``ModelRunner`` of an attention rank."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def load_model(result, model_runner) -> None:
    """AFTER ``ModelRunner.load_model``: the weights are loaded and the KV cache is
    not sized yet. With --asap-ffn-size, connect to the FFN ranks now."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    from sgl_asap_plugin import hooks
    from sgl_asap_plugin.srt.models import check_asap_model
    from sgl_asap_plugin.srt.server_args import get_asap_args, is_afd_attn

    if getattr(model_runner, "draft_model_idx", None) is not None:
        return
    args, parallel = get_asap_args(), get_parallel()
    # Plain DP: the world of a replica is its TP group, not dp_size * tp_size.
    world_size = dist.get_world_size() if dist.is_initialized() else None
    logger.info(
        f"[ASAP] attn rank ready: replica={parallel.dp_rank or 0} "
        f"tp_rank={parallel.tp_rank} world_size={world_size} "
        f"gpu={model_runner.gpu_id} pid={os.getpid()} "
        f"hooks={hooks.hooks_applied()} afd={is_afd_attn(args)}"
    )
    if is_afd_attn(args):
        from sgl_asap_plugin.srt.batch_overlap.single_batch_overlap import (
            install_shared_experts_overlap,
        )
        from sgl_asap_plugin.srt.layers.moe.token_dispatcher.asap import AsapBuffer

        check_asap_model(type(model_runner.model).__name__)
        count = install_shared_experts_overlap(model_runner.model)
        logger.info(
            f"[ASAP] shared experts overlap: {count} permanent pre-combine hooks"
        )
        AsapBuffer.connect(model_runner)


def _prepare_eager_forward_batch(result, model_runner, forward_batch) -> None:
    """AFTER ``ModelRunner._prepare_eager_forward_batch`` on an attention rank whose
    experts run on the FFN ranks: with --enable-two-batch-overlap, split a prefill
    batch of at least --asap-tbo-min-tokens tokens into its two micro-batches. SGLang splits in the attention-DP sync
    (``prepare_mlp_sync_batch``), where the DP ranks agree on it; an ASAP replica
    decides alone, here, before the attention metadata is built for the children."""
    from sglang.srt.batch_overlap.two_batch_overlap import (
        TboForwardBatchPreparer,
        compute_split_seq_index,
    )
    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.srt.layers.moe import is_tbo_enabled
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from sgl_asap_plugin.srt.server_args import get_asap_args

    if (
        not is_tbo_enabled()
        or forward_batch.forward_mode != ForwardMode.EXTEND
        or forward_batch.tbo_split_seq_index is not None
        or forward_batch.input_ids.shape[0] < get_asap_args().asap_tbo_min_tokens
    ):
        return
    split = compute_split_seq_index(
        forward_mode=forward_batch.forward_mode,
        num_tokens=forward_batch.input_ids.shape[0],
        extend_lens=forward_batch.extend_seq_lens_cpu,
        token_num_per_seq=None,
    )
    if split is None:
        return
    forward_batch.tbo_split_seq_index = split
    forward_batch.global_forward_mode = forward_batch.forward_mode
    if forward_batch.dp_padding_mode is None:
        # The TBO engine reads it; one rank gathers nothing, so either mode will do.
        forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    TboForwardBatchPreparer.prepare(
        forward_batch, is_draft_worker=model_runner.is_draft_worker
    )
