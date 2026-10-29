"""Static memory layouts of one A<->F deployment.

* F rank f owns a RecvArena in device memory: payload regions [a][slot] written by A ranks.
* F rank f also owns a slot file in /dev/shm: one doorbell word (own cache line) per [a][slot].
  A's GPU writes it with a 64-bit memop after the payload; F's CPU polls it.
* A rank a owns a ResultArena in device memory: done flags [f][slot] in a head page of their
  own, followed by the partial-sum segments [f][slot], written by F rank f.

Numbers follow roadmap 02b section 6; regions are 256-byte aligned, flags get a cache line each.
The flags page is FLAGS_PAGE bytes so that no payload shares a page with a flag A polls: measured
on H100 NVLink, with payload 4 KiB behind the flags the F->A copy ran at 287 GB/s and the
pipelined send+reply at 549 GB/s; with 64 KiB or more it ran at 384 GB/s and 687 GB/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from sgl_asap_plugin.errors import AsapConfigError
from sgl_asap_plugin.transport.protocol import MAX_TOKENS

ALIGN = 256
FLAG_STRIDE = 64
FLAGS_PAGE = 2 << 20
HOST_SLOT_BYTES = 64  # one doorbell word per cache line
ACT_BYTES = 2  # bf16
TOPK_ID_BYTES = 4  # int32 logical expert id
TOPK_W_BYTES = 4  # fp32 routing weight


def align_up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


@dataclass(frozen=True)
class Geometry:
    num_a: int  # D * T A ranks
    group: int  # T: A ranks (TP rows) whose messages form one F job
    slots: int  # in-flight micro-batch slots per A rank
    n_max: int  # max tokens per message
    hidden: int
    topk: int
    num_f: int  # E F ranks

    def __post_init__(self):
        bad = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
               if not isinstance(getattr(self, f.name), int) or getattr(self, f.name) < 1]
        if not bad and self.num_a % self.group:
            bad.append(f"num_a={self.num_a} is not a multiple of group={self.group}")
        if not bad and self.n_max > MAX_TOKENS:
            bad.append(f"n_max={self.n_max} exceeds the doorbell's {MAX_TOKENS} tokens")
        if bad:
            raise AsapConfigError("[ASAP] invalid transport geometry: " + ", ".join(bad))

    # ---- F rank: RecvArena (device) ----
    @property
    def ids_off(self) -> int:
        return align_up(self.n_max * self.hidden * ACT_BYTES)

    @property
    def w_off(self) -> int:
        return self.ids_off + align_up(self.n_max * self.topk * TOPK_ID_BYTES)

    @property
    def region_bytes(self) -> int:
        return self.w_off + align_up(self.n_max * self.topk * TOPK_W_BYTES)

    @property
    def recv_bytes(self) -> int:
        return self.num_a * self.slots * self.region_bytes

    def region_off(self, a: int, s: int) -> int:
        return (a * self.slots + s) * self.region_bytes

    # ---- F rank: slot file (host) ----
    @property
    def host_bytes(self) -> int:
        return align_up(self.num_a * self.slots * HOST_SLOT_BYTES, 4096)

    def host_doorbell_off(self, a: int, s: int) -> int:
        return (a * self.slots + s) * HOST_SLOT_BYTES

    # ---- A rank: ResultArena (device) ----
    @property
    def flags_bytes(self) -> int:
        return align_up(self.num_f * self.slots * FLAG_STRIDE, FLAGS_PAGE)

    @property
    def seg_bytes(self) -> int:
        return align_up(self.n_max * self.hidden * ACT_BYTES)

    @property
    def result_bytes(self) -> int:
        return self.flags_bytes + self.num_f * self.slots * self.seg_bytes

    def done_off(self, f: int, s: int) -> int:
        return (f * self.slots + s) * FLAG_STRIDE

    def seg_off(self, f: int, s: int) -> int:
        return self.flags_bytes + (f * self.slots + s) * self.seg_bytes

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(**{f.name: d[f.name] for f in fields(cls)})
