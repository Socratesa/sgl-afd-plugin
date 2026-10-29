"""A side of the transport: one AfdChannel per A rank, connected to every F rank (memory layout: layout.py).

send() and wait() only enqueue work on the caller's stream (copies, stream memops, one event per group of
waits). The CPU never waits for F, so it keeps launching the next layers while the GPU waits. Caller rules:

* x / ids / w must not change until the send's copies have run; keeping everything on one stream does it.
* the result segment of slot s must be consumed (in stream order) before the next send() on slot s, because F
  overwrites it when it answers that next message.
* a parked wait also delays work that needs a device-wide sync (the first launch of a kernel, cudaMalloc,
  cudaFree) until F answers. That is a delay, not a deadlock, as long as the answer comes from another process;
  warm such paths up before serving.
* after a timeout the released stream has read garbage: check() raises ChannelPoisoned, reconnect() recovers.
"""

from __future__ import annotations

import collections
import ctypes
import json
import mmap
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport.layout import (ACT_BYTES, ID_BYTES, PROTO_VERSION, SEQ_MASK, ChannelPoisoned,
                                              Geometry, HandshakeError, pack_doorbell)

MAX_HANDSHAKE_BYTES = 1 << 20


def request(host: str, port: int, msg: dict, *, connect_timeout: float = 60.0, io_timeout: float = 30.0) -> dict:
    """Send one JSON line to an F rank and return its JSON line answer.

    Retries the connect until F listens (F loads tens of GB before it does) or connect_timeout passes.
    """
    deadline = time.monotonic() + connect_timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            break
        except OSError as exc:
            if time.monotonic() > deadline:
                raise HandshakeError(f"cannot reach F at {host}:{port}: {exc}") from exc
            time.sleep(0.2)
    with sock, sock.makefile("rwb") as f:
        sock.settimeout(io_timeout)
        try:
            f.write(json.dumps(msg).encode() + b"\n")
            f.flush()
            line = f.readline(MAX_HANDSHAKE_BYTES)
        except OSError as exc:
            raise HandshakeError(f"handshake with F at {host}:{port} failed: {exc}") from exc
    if not line:
        raise HandshakeError(f"F at {host}:{port} closed the connection without answering")
    reply = json.loads(line)
    if not reply.get("ok"):
        raise HandshakeError(f"F at {host}:{port} rejected the connection: {reply.get('reason')}")
    return reply


class _Phase:
    """A group of stream waits plus the event recorded right after them. ``release`` lists (address, value) pairs
    that, when written, satisfy every wait of the group."""

    __slots__ = ("event", "release", "what", "state", "t_head")

    def __init__(self, event: int, release: List[Tuple[int, int]], what: str):
        self.event, self.release, self.what = event, release, what
        self.state = "pending"  # pending -> done | failed
        self.t_head: Optional[float] = None  # when this became the oldest unfinished phase


class Watchdog(threading.Thread):
    """Releases A's stream waits when F does not answer within ``timeout_s``.

    A stream parked in cuStreamWaitValue32 never times out by itself, so a dead F would hang A's GPU and every
    later torch.cuda.synchronize(). Phases of one stream finish in order, so only the oldest unfinished phase is
    watched; its clock starts when it becomes the oldest. That keeps a scan O(1), which matters because this
    thread shares the GIL with the scheduler. On timeout the watchdog writes every awaited value from a stream of
    its own (releasing all parked waits, not just the late one) and calls ``on_timeout``.
    """

    SCAN_S = 0.02

    def __init__(self, device: cu.Device, timeout_s: float, on_timeout: Callable[[_Phase], None]):
        super().__init__(name="asap-watchdog", daemon=True)
        self.device, self.timeout_s, self.on_timeout = device, timeout_s, on_timeout
        self._queue: collections.deque = collections.deque()
        self._free_events: List[int] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

    def watch(self, stream: int, release: List[Tuple[int, int]], what: str) -> _Phase:
        """Record an event on ``stream`` after the waits just enqueued and start tracking them."""
        with self._lock:
            event = self._free_events.pop() if self._free_events else cu.event_create()
        cu.event_record(event, stream)
        phase = _Phase(event, release, what)
        with self._lock:
            self._queue.append(phase)
        return phase

    def run(self) -> None:
        self.device.activate()
        recovery = cu.stream_create()
        while not self._stop_evt.wait(self.SCAN_S):
            with self._lock:
                while self._queue and cu.event_done(self._queue[0].event):
                    done = self._queue.popleft()
                    if done.state == "pending":
                        done.state = "done"
                    self._free_events.append(done.event)
                head = self._queue[0] if self._queue else None
                if head is None or head.state != "pending":
                    continue
                now = time.monotonic()
                if head.t_head is None:
                    head.t_head = now
                    continue
                if now - head.t_head <= self.timeout_s:
                    continue
                late = [p for p in self._queue if p.state == "pending"]
                for p in late:
                    p.state = "failed"
                    for addr, value in p.release:
                        cu.write32(recovery, addr, value)
            self.on_timeout(head)
        cu.stream_destroy(recovery)

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=5.0)
        for event in self._free_events + [p.event for p in self._queue]:
            cu.event_destroy(event)
        self._free_events, self._queue = [], collections.deque()


@dataclass
class Ticket:
    """One message in flight. ``send`` / ``reply`` are the watched phases of its two groups of waits."""

    slot: int
    seq: int
    layer_id: int
    n_tokens: int
    send: Optional[_Phase] = None
    reply: Optional[_Phase] = None
    result_ptrs: List[int] = field(default_factory=list)


class AfdChannel:
    """A rank ``a``'s connection to every F rank (``endpoints[f]`` is F rank f's handshake address)."""

    def __init__(self, a: int, geometry: Geometry, endpoints: Sequence[Tuple[str, int]], *, device: int,
                 seq_base: int = 0, timeout_s: float = 60.0, connect_timeout_s: float = 60.0,
                 hello: Optional[dict] = None, _backpressure: bool = True, _epoch_override: Optional[int] = None):
        """``hello`` is merged into the handshake message (the plugin sends its model fingerprint there).
        ``_backpressure`` / ``_epoch_override`` exist only for tests that must provoke protocol violations."""
        if len(endpoints) != geometry.num_f:
            raise ValueError(f"need one endpoint per F rank ({geometry.num_f}), got {len(endpoints)}")
        if not 0 <= a < geometry.num_a:
            raise ValueError(f"A rank {a} is outside [0, {geometry.num_a})")
        self.a, self.g, self.endpoints, self.seq_base = a, geometry, list(endpoints), seq_base
        self.device, self.timeout_s, self.connect_timeout_s = cu.Device(device), timeout_s, connect_timeout_s
        self.hello = dict(hello or {})
        self._backpressure, self._epoch_override = _backpressure, _epoch_override
        self._poison: Optional[ChannelPoisoned] = None
        self._watchdog: Optional[Watchdog] = None
        self._result: Optional[int] = None
        self._recv: List[int] = []  # F rank f's RecvArena, mapped here
        self._bells: List[int] = []  # device address of F rank f's slot file
        self._maps: List[tuple] = []  # (mmap, host address) per slot file
        self._epoch: List[int] = []
        self.replies: List[dict] = []  # F's handshake answers (what each rank serves, for logs)

    def connect(self) -> "AfdChannel":
        """Allocate the ResultArena, hand it to every F rank, map their RecvArenas and slot files."""
        g = self.g
        self.device.activate()
        self._result = cu.mem_alloc(g.result_bytes)
        cu.memset32(self._result, self.seq_base, g.flags_bytes // 4)  # every done flag starts at seq_base
        hello = {**self.hello, "proto": PROTO_VERSION, "a": self.a, "geometry": g.as_dict(), "pid": os.getpid(),
                 "result_handle": cu.ipc_export(self._result).hex()}
        try:
            for host, port in self.endpoints:
                reply = request(host, port, hello, connect_timeout=self.connect_timeout_s)
                if reply["seq_base"] != self.seq_base:
                    raise HandshakeError(f"seq_base A={self.seq_base} F={reply['seq_base']}")
                self._recv.append(cu.ipc_open(bytes.fromhex(reply["recv_handle"])))
                self._bells.append(self._map_slot_file(reply["shm_path"]))
                self._epoch.append(reply["epoch"])
                self.replies.append(reply)
        except BaseException:
            self.close()
            raise
        self._seq = [self.seq_base] * g.slots
        self._poison = None
        self._watchdog = Watchdog(self.device, self.timeout_s, self._on_timeout)
        self._watchdog.start()
        return self

    def _map_slot_file(self, path: str) -> int:
        """mmap F's /dev/shm slot file and register it so this GPU can write the doorbells -> device address."""
        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, self.g.host_bytes)
        finally:
            os.close(fd)
        buf = ctypes.c_char.from_buffer(mm)
        addr = ctypes.addressof(buf)
        del buf  # the address stays valid while the mapping is open; a live export would block mm.close()
        self._maps.append((mm, addr))
        return cu.host_register(addr, self.g.host_bytes)

    def reconnect(self) -> "AfdChannel":
        """The only way out of a poisoned channel: a new epoch and a fresh ResultArena on every F rank."""
        self.close()
        return self.connect()

    def close(self) -> None:
        """Release everything. Synchronize the streams that used the channel first."""
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        self.device.activate()
        for base in self._recv:
            cu.ipc_close(base)
        for mm, addr in self._maps:
            cu.host_unregister(addr)
            mm.close()
        if self._result is not None:
            cu.mem_free(self._result)
        self._recv, self._bells, self._maps, self._epoch, self.replies, self._result = [], [], [], [], [], None

    def _on_timeout(self, phase: _Phase) -> None:
        self._poison = ChannelPoisoned(
            f"A rank {self.a}: {phase.what} did not complete within {self.timeout_s}s, F is dead or stuck; "
            "the results of every pending message are garbage until reconnect()")

    def check(self) -> None:
        """Raise ChannelPoisoned if a wait has timed out."""
        if self._poison is not None:
            raise self._poison

    def _done(self, f: int, slot: int) -> int:
        return self._result + self.g.done_off(f, slot)

    def send(self, layer_id: int, slot: int, x_ptr: int, ids_ptr: int, w_ptr: int, n_tokens: int,
             stream: int) -> Ticket:
        """Enqueue one message on ``stream``: ``n_tokens`` contiguous rows of x (bf16 [n, hidden]), topk ids
        (int32 [n, topk]) and topk weights (fp32 [n, topk]) at the given device addresses, to every F rank."""
        self.check()
        g = self.g
        if not (0 <= slot < g.slots and 1 <= n_tokens <= g.n_max):
            raise ValueError(f"slot {slot} / n_tokens {n_tokens} outside the geometry {g}")
        seq = self._seq[slot] = (self._seq[slot] + 1) & SEQ_MASK
        x_n, k_n = n_tokens * g.hidden * ACT_BYTES, n_tokens * g.topk * ID_BYTES
        if self._backpressure:  # the previous message of this slot must be answered before its buffers are reused
            for f in range(g.num_f):
                cu.wait32(stream, self._done(f, slot), seq - 1)
        for f in range(g.num_f):
            epoch = self._epoch[f] if self._epoch_override is None else self._epoch_override
            dst = self._recv[f] + g.region_off(self.a, slot)
            cu.memcpy_async(dst, x_ptr, x_n, stream)
            cu.memcpy_async(dst + g.ids_off, ids_ptr, k_n, stream)
            cu.memcpy_async(dst + g.w_off, w_ptr, k_n, stream)  # fp32 weights: same size as the int32 ids
            # the doorbell goes last in the same stream: F sees it only after the payload is visible
            cu.write64(stream, self._bells[f] + g.bell_off(self.a, slot),
                       pack_doorbell(seq, n_tokens, layer_id, epoch))
        ticket = Ticket(slot, seq, layer_id, n_tokens)
        ticket.send = self._watchdog.watch(stream, [(self._done(f, slot), seq - 1) for f in range(g.num_f)],
                                           f"send of layer {layer_id} slot {slot} seq {seq}")
        return ticket

    def wait(self, ticket: Ticket, stream: int) -> List[int]:
        """Enqueue on ``stream`` the waits for every F rank's answer to ``ticket``; returns the A-device addresses
        of the E partial sums (bf16 [n_tokens, hidden] each). The caller adds them."""
        self.check()
        g, slot, seq = self.g, ticket.slot, ticket.seq
        for f in range(g.num_f):
            cu.wait32(stream, self._done(f, slot), seq)
        ticket.reply = self._watchdog.watch(stream, [(self._done(f, slot), seq) for f in range(g.num_f)],
                                            f"reply to layer {ticket.layer_id} slot {slot} seq {seq}")
        ticket.result_ptrs = [self._result + g.seg_off(f, slot) for f in range(g.num_f)]
        return ticket.result_ptrs
