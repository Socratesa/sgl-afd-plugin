"""What both sides of the A<->F transport must agree on: memory layout, doorbell word, error types.

One message = one micro-batch (slot ``s``) of one MoE layer from A rank ``a`` to every F rank ``f``.
A's CPU only enqueues; the GPU streams carry the dependencies:

    A rank a, one CUDA stream                                F rank f, CPU poller + one CUDA stream
    1. wait32  done[f][s] >= seq-1    (backpressure: the
               previous reply of slot s has landed)
    2. memcpy  x, topk_ids, topk_w -> F.recv[a][s]  (NVLink)
    3. write64 doorbell(seq, n, layer, epoch)
               -> F.bell[a][s]  (host memory)           ->   sees seq advance (plain loads), launches
                                                             experts(x, ids, w) = sum of w_k * expert_k(x)
                                                             over the experts it owns
                                   A.seg[f][s]   <-  memcpy  partial sum
                                   A.done[f][s]  <-  write32 seq
    4. wait32  done[f][s] >= seq for every f; the caller adds the E partial sums

Memory owned by F rank f (``Geometry`` gives every offset; regions are ALIGN-aligned):

    RecvArena, device memory, written by A ranks through CUDA IPC (recv_bytes):
    | region(a=0,s=0)                                  | region(0,1) | ... | region(num_a-1, slots-1) |
    | x [n_max, hidden] bf16 | ids [n_max, topk] i32 | w [n_max, topk] f32 |
    0                     ids_off                  w_off             region_bytes
    region(a, s) starts at (a * slots + s) * region_bytes

    slot file, host memory (/dev/shm, host_bytes), written by A's GPU with one 64-bit memop:
    | bell(0,0) .. 64 B | bell(0,1) .. 64 B | ... |     one cache line per (a, s), doorbell in word 0

Memory owned by A rank a:

    ResultArena, device memory, written by F ranks through CUDA IPC (result_bytes):
    | done(0,0) 64 B | done(0,1) | ... (FLAGS_PAGE = 2 MiB) | seg(f=0,s=0) [n_max, hidden] bf16 | seg(0,1) | ... |
    0                                                      flags_bytes = seg_off(0, 0)

Why it looks like this:
* Every flag and doorbell has a cache line of its own, so two writers never share a line.
* The done flags sit in a 2 MiB page with no payload. Measured on H100 NVLink: with the first segment 4 KiB after
  the flags, the F->A copy ran at 287 GB/s and the pipelined send+reply at 549 GB/s; with 64 KiB or more, at 384
  and 687 GB/s. The cause is unknown (ruled out: allocator, alignment, stream flags, flag location and width).
* The doorbell is ONE 64-bit word, so F's CPU can never see a ready flag without its metadata, and A never puts a
  host-to-host copy into its stream: such a cuMemcpyAsync blocks the calling CPU thread until all earlier work of
  the stream has finished, which defeats a stream parked in a wait.
* Doorbells live in host memory because F's CPU decides what to launch next anyway; polling device memory would
  cost a device-to-host copy per poll. F reads them with plain loads: atomics or acquire loads on host-mapped
  memory hang (spike finding).
* Everything that never changes per message (protocol version, hidden, topk, dtypes, capacities) is checked once
  in the handshake (``Geometry`` must match field by field), so the doorbell carries only what varies.

Doorbell bits:  0-31 seq | 32-48 n_tokens (1..MAX_TOKENS) | 49-58 layer_id (0..MAX_LAYER) | 59-63 epoch (1..MAX_EPOCH)
* seq: per (a, s) message number starting at the handshake's seq_base; it is also the ready flag. It wraps at 2^32
  and is always compared as a cyclic int32 difference (the CUDA GEQ wait is wrap-safe).
* epoch: F hands out a new one (1..31, cycling) at every handshake of rank a and drops doorbells of older ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

ALIGN = 256
FLAG_STRIDE = 64  # one done flag per cache line
FLAGS_PAGE = 2 << 20  # done flags get a page with no payload (see the module docstring)
BELL_STRIDE = 64  # one doorbell per cache line of the slot file
ACT_BYTES = 2  # x is bf16
ID_BYTES = 4  # topk ids are int32 logical expert ids
W_BYTES = 4  # topk weights are fp32

PROTO_VERSION = 3
SEQ_MASK = 0xFFFFFFFF
MAX_TOKENS = (1 << 17) - 1
MAX_LAYER = (1 << 10) - 1
MAX_EPOCH = (1 << 5) - 1


class TransportError(RuntimeError):
    """Base class of every transport failure."""


class HandshakeError(TransportError):
    """The peer could not be reached or rejected the connection."""


class ChannelPoisoned(TransportError):
    """A wait timed out. Its stream was released with garbage results; the slots stay unusable until reconnect()."""


def align_up(n: int, a: int = ALIGN) -> int:
    """Round n up to a multiple of a."""
    return -(-n // a) * a


def pack_doorbell(seq: int, n_tokens: int, layer_id: int, epoch: int) -> int:
    """The 64-bit doorbell (bit layout in the module docstring); out-of-range fields raise."""
    if not (1 <= n_tokens <= MAX_TOKENS and 0 <= layer_id <= MAX_LAYER and 1 <= epoch <= MAX_EPOCH):
        raise ValueError(f"doorbell field out of range: n_tokens={n_tokens} layer_id={layer_id} epoch={epoch}")
    return (seq & SEQ_MASK) | (n_tokens << 32) | (layer_id << 49) | (epoch << 59)


def unpack_doorbell(word: int):
    """-> (seq, n_tokens, layer_id, epoch)"""
    return word & SEQ_MASK, (word >> 32) & MAX_TOKENS, (word >> 49) & MAX_LAYER, word >> 59


@dataclass(frozen=True)
class Geometry:
    """Sizes of one deployment; A and F refuse to connect unless every field matches."""

    num_a: int  # A ranks = D replicas * T TP ranks each; A rank a = replica * group + tp_rank
    group: int  # T: the A ranks of one replica; F joins their T rows into one job
    slots: int  # micro-batches in flight per A rank (the synchronous forward uses 1, dual-batch 2)
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
            bad.append(f"n_max={self.n_max} exceeds the {MAX_TOKENS} tokens a doorbell can announce")
        if bad:
            raise ValueError("invalid transport geometry: " + ", ".join(bad))

    # ---- F rank: RecvArena (device) ----
    @property
    def ids_off(self) -> int:
        return align_up(self.n_max * self.hidden * ACT_BYTES)

    @property
    def w_off(self) -> int:
        return self.ids_off + align_up(self.n_max * self.topk * ID_BYTES)

    @property
    def region_bytes(self) -> int:
        return self.w_off + align_up(self.n_max * self.topk * W_BYTES)

    @property
    def recv_bytes(self) -> int:
        return self.num_a * self.slots * self.region_bytes

    def region_off(self, a: int, s: int) -> int:
        """Offset of A rank a's slot-s region in an F rank's RecvArena."""
        return (a * self.slots + s) * self.region_bytes

    # ---- F rank: slot file (host) ----
    @property
    def host_bytes(self) -> int:
        return align_up(self.num_a * self.slots * BELL_STRIDE, 4096)

    def bell_off(self, a: int, s: int) -> int:
        """Offset of A rank a's slot-s doorbell in an F rank's slot file."""
        return (a * self.slots + s) * BELL_STRIDE

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
        """Offset of F rank f's slot-s done flag in an A rank's ResultArena (flags page)."""
        return (f * self.slots + s) * FLAG_STRIDE

    def seg_off(self, f: int, s: int) -> int:
        """Offset of F rank f's slot-s partial sum in an A rank's ResultArena."""
        return self.flags_bytes + (f * self.slots + s) * self.seg_bytes

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(**{f.name: d[f.name] for f in fields(cls)})
