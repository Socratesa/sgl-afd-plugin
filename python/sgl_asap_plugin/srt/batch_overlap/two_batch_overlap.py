"""When an attention rank whose routed experts run on the FFN ranks overlaps two batches.

SGLang's models gate two-batch overlap on the collective it hides: an EP all-to-all, or
the gather and scatter of attention DP. An ASAP replica has neither (plain data
parallelism, ``--moe-a2a-backend none``); what it hides is the transfer to the FFN
ranks (``operations_strategy.py``). ``can_run_tbo`` is SGLang's gate without that path
check.
"""

from __future__ import annotations


def can_run_tbo(model, forward_batch) -> bool:
    """The gate of ``DeepseekV4Model._can_run_tbo`` minus its path check, for a model
    with one pipeline stage."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.moe import is_tbo_enabled

    return (
        is_tbo_enabled()
        and forward_batch.can_run_tbo
        and forward_batch.tbo_children is not None
        and forward_batch.global_forward_mode is not None
        # Target-verify also reports is_extend(); only a real prefill is overlapped.
        and forward_batch.global_forward_mode.is_extend_without_speculative()
        and not is_cp_active(forward_batch)
        and model.pp_group.world_size == 1
    )
