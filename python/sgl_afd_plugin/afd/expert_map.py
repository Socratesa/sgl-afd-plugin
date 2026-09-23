"""AFD logical -> physical expert layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AFDExpertLayout:
    """Rank-independent AFD expert layout."""

    ep_size: int
    num_layers: int
    num_logical_experts: int
    num_physical_experts: int
    ep_num_redundant_experts: int
    attn_ep_ranks: tuple[int, ...]
    ffn_ep_ranks: tuple[int, ...]
    # (num_layers, num_physical_experts); -1 on an attention rank's slots
    physical_to_logical_map: torch.Tensor


def build_afd_expert_layout(
    *,
    ep_size: int,
    num_layers: int,
    num_logical_experts: int,
    ffn_ep_ranks,
    device,
) -> AFDExpertLayout:
    """Place the logical experts on the FFN ranks' contiguous blocks."""
    ffn = tuple(sorted(set(ffn_ep_ranks)))
    if not 0 < len(ffn) < ep_size:
        raise ValueError(f"AFD needs 1..{ep_size - 1} FFN ep ranks, got {len(ffn)}")

    slots = -(-num_logical_experts // len(ffn))  # ceil: > L/f means replication
    num_physical = ep_size * slots

    physical = torch.arange(num_physical, device=device)
    owner = physical // slots
    # Position of the owning rank inside the FFN rank list, -1 for attention.
    position = torch.full((ep_size,), -1, dtype=torch.long, device=device)
    position[torch.tensor(ffn, dtype=torch.long, device=device)] = torch.arange(
        len(ffn), device=device
    )
    owner_position = position[owner]
    # FFN slots hold the logical experts in order. An attention slot holds no
    # expert at all, so it is marked -1 and is never a dispatch destination.
    physical_to_logical = torch.where(
        owner_position >= 0,
        (owner_position * slots + physical % slots) % num_logical_experts,
        -1,
    )

    attn = tuple(r for r in range(ep_size) if r not in set(ffn))
    return AFDExpertLayout(
        ep_size=ep_size,
        num_layers=num_layers,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical,
        ep_num_redundant_experts=num_physical - num_logical_experts,
        attn_ep_ranks=attn,
        ffn_ep_ranks=ffn,
        physical_to_logical_map=physical_to_logical.unsqueeze(0).repeat(num_layers, 1),
    )
