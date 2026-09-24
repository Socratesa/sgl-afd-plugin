"""AFD override for SGLang's initial expert-location metadata.

The mapping is consumed at `topk.py:_post_process_topk_ids` ->
`topk_ids_logical_to_physical` (eplb/expert_location_dispatch.py:83), i.e.
before dispatch, because moe_a2a_backend derives the destination rank
from the physical id itself.
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_parallel,
)

from sgl_afd_plugin.afd.config import get_afd
from sgl_afd_plugin.afd.expert_map import build_afd_expert_layout

logger = logging.getLogger(__name__)


def _around_compute_logical_to_all_physical_map(
    original_fn,
    physical_to_logical_map,
    num_logical_experts,
    ep_size,
    moe_ep_rank,
):
    """Park the -1 slots in a bucket of their own, then slice that bucket off."""
    if not (physical_to_logical_map < 0).any():
        return original_fn(
            physical_to_logical_map, num_logical_experts, ep_size, moe_ep_rank
        )
    # The inversion keys its buckets by the map's values, so a -1 lands on the
    # last bucket (Python list[-1]). One bucket more than there are logical
    # experts makes that bucket the throwaway one; the slice drops it again.
    return original_fn(
        physical_to_logical_map, num_logical_experts + 1, ep_size, moe_ep_rank
    )[:, :num_logical_experts, :]


def _gather_ranks() -> tuple[tuple[int, int, str], ...] | None:
    """(node_rank, moe_ep_rank, role) of every rank, or None without a world."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if group.world_size <= 1:
        return None
    local = (get_parallel().node_rank, get_parallel().moe_ep_rank, get_afd().role)
    return tuple(group.all_gather_object(local))


def _install_layout(layout) -> None:
    """Publish the layout knobs that follow from the FFN rank set."""
    redundant = get_exec().moe.ep_num_redundant_experts
    if redundant not in (0, layout.ep_num_redundant_experts):
        raise ValueError(
            f"[AFD] --ep-num-redundant-experts={redundant} conflicts with the AFD "
            f"layout, which needs {layout.ep_num_redundant_experts} "
            f"(ep_size={layout.ep_size}, ffn_ep_ranks={list(layout.ffn_ep_ranks)}, "
            f"logical={layout.num_logical_experts}). Drop the flag; AFD derives it."
        )

    # Padding for the attention ranks' slots, which hold no expert.
    # Also necessary for the topk remap: _eplb_remap_enabled() (topk.py:1544) returns
    # True, otherwise topk_ids_logical_to_physical is skipped.
    get_context().override(
        "afd.expert_location",
        ep_num_redundant_experts=layout.ep_num_redundant_experts,
    )
    # Verify the override landed instead of letting a mismatched layout through.
    if get_exec().moe.ep_num_redundant_experts != layout.ep_num_redundant_experts:
        raise RuntimeError(
            "[AFD] ep_num_redundant_experts override did not take effect; the "
            "physical expert width would not match the AFD layout."
        )
    logger.info(
        "[AFD] ffn ep ranks=%s, attn ep ranks=%s, physical experts=%d, local=%d",
        list(layout.ffn_ep_ranks),
        list(layout.attn_ep_ranks),
        layout.num_physical_experts,
        layout.num_physical_experts // layout.ep_size,
    )


def _maybe_apply_afd_layout(model_config):
    from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

    model_config_for_expert_location = (
        ModelConfigForExpertLocation.from_model_config(model_config)
    )

    if model_config_for_expert_location is None:
        return None

    try:
        ranks = _gather_ranks()
    except Exception:
        logger.warning("[AFD] role gather failed; keeping the host layout")
        return None
    if ranks is None:
        return None

    parallel = get_parallel()
    layout = build_afd_expert_layout(
        ep_size=parallel.ep_size,
        num_layers=model_config_for_expert_location.num_layers,
        num_logical_experts=model_config_for_expert_location.num_logical_experts,
        ffn_ep_ranks=[
            rank for _, rank, role in ranks if role == "ffn"
        ],
        device=get_device().device,
    )
    _install_layout(layout)
    return layout


def _around_init_trivial(original_fn, model_config, moe_ep_rank):
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    layout = _maybe_apply_afd_layout(model_config)
    if layout is None:
        return original_fn(model_config, moe_ep_rank)

    metadata = ExpertLocationMetadata.init_by_mapping(
        model_config,
        physical_to_logical_map=layout.physical_to_logical_map,
        moe_ep_rank=moe_ep_rank,
    )
    if metadata is None:
        return original_fn(model_config, moe_ep_rank)

    return metadata


def install():
    HookRegistry.register(
        "sglang.srt.eplb.expert_location.ExpertLocationMetadata.init_trivial",
        _around_init_trivial,
        HookType.AROUND,
    )
    # The layout marks the attention ranks' slots -1; keep them out of the
    # tables the host derives from the map.
    HookRegistry.register(
        "sglang.srt.eplb.expert_location._compute_logical_to_all_physical_map",
        _around_compute_logical_to_all_physical_map,
        HookType.AROUND,
    )
