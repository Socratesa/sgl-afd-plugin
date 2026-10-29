"""Static layout math shared by both roles: expert partition across F ranks and arena sizing.

Pure Python, so it can be unit-tested and later reused by the handshake (PR-03), which makes
A and F agree on exactly these numbers before any data moves. Formulas follow the roadmap
(02b section 6).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from sgl_asap_plugin.errors import AsapConfigError

HEADER_BYTES = 64  # per-slot message header
ACT_BYTES = 2  # bf16 activations and partial sums
TOPK_ID_BYTES = 4  # int32 logical expert id
TOPK_W_BYTES = 4  # fp32 routing weight


@dataclass(frozen=True)
class ExpertPartition:
    """Contiguous split of the logical experts over F ranks, padded with dummy experts."""

    n_experts: int
    ep_size: int
    per_rank: int

    @property
    def padded(self) -> int:
        return self.per_rank * self.ep_size

    @property
    def padding(self) -> int:
        return self.padded - self.n_experts

    def real_range(self, rank: int) -> Tuple[int, int]:
        """Logical experts [lo, hi) owned by ``rank``; dummy experts are the tail ids >= n_experts."""
        lo = rank * self.per_rank
        return (min(lo, self.n_experts), min(lo + self.per_rank, self.n_experts))

    def ranges(self) -> List[Tuple[int, int]]:
        return [self.real_range(r) for r in range(self.ep_size)]


def partition_experts(n_experts: int, ep_size: int) -> ExpertPartition:
    if n_experts < 1 or ep_size < 1:
        raise AsapConfigError(f"[ASAP] n_experts and ep_size must be >= 1, got {n_experts}, {ep_size}")
    if ep_size > n_experts:
        raise AsapConfigError(f"[ASAP] ep_size {ep_size} exceeds the number of experts {n_experts}")
    return ExpertPartition(n_experts, ep_size, -(-n_experts // ep_size))


@dataclass(frozen=True)
class ArenaPlan:
    bytes_per_token: int
    region_bytes: int  # one (A rank, slot) region in an F recv arena
    f_recv_arena_bytes: int  # per F rank
    a_result_arena_bytes: int  # per A rank


def arena_plan(num_replicas: int, replica_tp: int, ep_size: int, slots: int, n_max: int,
               hidden: int, topk: int) -> ArenaPlan:
    per_token = hidden * ACT_BYTES + topk * (TOPK_ID_BYTES + TOPK_W_BYTES)
    region = HEADER_BYTES + n_max * per_token
    return ArenaPlan(
        bytes_per_token=per_token,
        region_bytes=region,
        f_recv_arena_bytes=num_replicas * replica_tp * slots * region,
        a_result_arena_bytes=ep_size * slots * n_max * hidden * ACT_BYTES,
    )


@dataclass(frozen=True)
class MoeShape:
    n_experts: int
    hidden: int
    topk: int
    layers: int


def read_moe_shape(model_path: str, override_json: str = "{}") -> MoeShape:
    """Read the MoE dimensions from config.json (+ --json-model-override-args) without loading torch."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        cfg.update(json.loads(override_json or "{}"))
        cfg = cfg.get("text_config", cfg)
        n_experts = next(cfg[k] for k in ("n_routed_experts", "num_experts", "num_local_experts") if k in cfg)
        return MoeShape(n_experts=n_experts, hidden=cfg["hidden_size"], topk=cfg["num_experts_per_tok"],
                        layers=cfg["num_hidden_layers"])
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise AsapConfigError(f"[ASAP] cannot read MoE shape from {model_path}/config.json: {exc!r}") from exc
