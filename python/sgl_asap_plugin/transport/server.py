"""F side of the transport: one FfnServer per F rank owns a RecvArena and a doorbell page (layout.py; memory.py
shares them), discovers messages by polling the doorbell page, and answers each with a partial sum written into
the sender's ResultArena.

Threads of an F rank process:
* the handshake thread accepts A ranks (one JSON line in, one out), validates them and maps their ResultArena;
* the serving thread runs serve_forever(): poll, hand each complete job to the handler (which launches the
  experts and calls reply()), repeat. Detection and reply share this thread, so a handler must only launch GPU
  work and return; long CPU work in it delays the discovery of every other doorbell.

A job is one micro-batch of one layer from one A replica: the T rows (one per TP rank of the replica) of the
same (replica, slot) message. Jobs are served first come, first served; doorbells found in the same poll are
taken in round-robin order of the A rank, so a low rank number gives no priority.
"""

from __future__ import annotations

import collections
import ctypes
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from sgl_asap_plugin.transport import cuda_wrapper as cu
from sgl_asap_plugin.transport.channel import MAX_HANDSHAKE_BYTES
from sgl_asap_plugin.transport.layout import (
    BELL_STRIDE,
    MAX_EPOCH,
    PROTO_VERSION,
    SEQ_MASK,
    Geometry,
    HandshakeError,
    unpack_doorbell,
)
from sgl_asap_plugin.transport.memory import Shared
from sgl_asap_plugin.transport.memory import select as select_memory


@dataclass
class Peer:
    """A connected A rank: where its ResultArena is mapped and which epoch its messages must carry."""

    a: int
    result_base: int  # its ResultArena, mapped here (memory.py)
    epoch: int
    pid: int
    mapping: Optional[Shared] = None  # the mapping at result_base, released on close or reconnect
    host: Optional[str] = None  # where it runs; a reconnect comes from the same host


@dataclass
class Row:
    """One A rank's part of a job; the pointers address this F rank's RecvArena."""

    a: int
    slot: int
    seq: int
    epoch: int
    layer_id: int
    n_tokens: int
    x_ptr: int
    ids_ptr: int
    w_ptr: int


@dataclass
class Job:
    """What the handler receives: one micro-batch of one layer of one replica, ready to compute."""

    replica: int
    slot: int
    layer_id: int
    rows: List[Row]  # one per TP rank of the replica, in TP-rank order
    t_detect_ns: int


class FfnServer:
    """One F rank's end of the transport (start -> serve_forever -> close); see the module docstring."""

    def __init__(
        self,
        f_rank: int,
        geometry: Geometry,
        *,
        device: int,
        run_dir: str,
        host: str = "127.0.0.1",
        port: int = 0,
        seq_base: int = 0,
        memory: str = "auto",
        model: Optional[dict] = None,
        info: Optional[dict] = None,
        trace_len: int = 4096,
    ):
        """``memory``: how the RecvArena and the doorbell page are shared (memory.py: auto, ipc or fabric); every
        A rank must resolve to the same. ``run_dir``: where the ipc backend puts the doorbell file (/dev/shm).
        ``model``: fingerprint every A rank must send in its hello (None skips the check).
        ``info``: extra fields of the handshake answer (what this rank serves, for A's logs).
        ``trace_len``: how many recent jobs ``trace`` keeps as (replica, slot, layer_id)."""
        self.f, self.g, self.seq_base = f_rank, geometry, seq_base
        self.memory, self.mem = memory, None  # the requested backend, the one start() chose
        self.model, self.info = model, dict(info or {})
        self.device, self.run_dir, self.host, self._port_req = (
            cu.Device(device),
            run_dir,
            host,
            port,
        )
        self.stats: collections.Counter = collections.Counter()
        self.trace: collections.deque = collections.deque(maxlen=trace_len)
        self.peers: Dict[int, Peer] = {}
        self._pending: Dict[
            tuple, Dict[int, Row]
        ] = {}  # (replica, slot) -> rows that arrived, by TP rank
        self._lock = (
            threading.RLock()
        )  # peers / pending / last-seen are shared with the handshake thread
        self._rr = 0  # where the next poll starts scanning (round robin over (a, slot))
        self._prev_replica = None

    def start(self) -> FfnServer:
        """Allocate and zero the RecvArena and the doorbell page, then listen for A ranks."""
        g = self.g
        self.device.activate()
        self.mem = select_memory(self.memory, self.device.ordinal)
        self._recv = self.mem.device(g.recv_bytes)
        self.recv_base = self._recv.ptr
        cu.memset8(self.recv_base, 0, g.recv_bytes)
        self._page = self.mem.doorbells(
            g.host_bytes, os.path.join(self.run_dir, f"asap_f{self.f}.slots")
        )
        words = np.ctypeslib.as_array(
            (ctypes.c_uint64 * (g.host_bytes // 8)).from_address(self._page.ptr)
        )
        # word 0 of each cache line; a strided view, so the poller copies it before decoding
        self._bells = words[: g.num_a * g.slots * (BELL_STRIDE // 8)].reshape(
            -1, BELL_STRIDE // 8
        )[:, 0]
        self._bells[:] = self.seq_base
        self._last = np.full(g.num_a * g.slots, self.seq_base, dtype=np.uint32)
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self._port_req))
        self._srv.listen(64)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]  # the real port when 0 was requested
        self._hs_stop = threading.Event()
        self._hs_thread = threading.Thread(
            target=self._accept_loop, name=f"asap-handshake-f{self.f}", daemon=True
        )
        self._hs_thread.start()
        return self

    # ---- handshake (handshake thread) ----
    def _accept_loop(self) -> None:
        """Handshake thread: one JSON line in and out per A rank; a bad hello gets the reason back."""
        while not self._hs_stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            with conn, conn.makefile("rwb") as f:
                conn.settimeout(10.0)
                try:
                    reply = self._on_hello(json.loads(f.readline(MAX_HANDSHAKE_BYTES)))
                except Exception as exc:  # reject with the reason; the thread must survive any bad peer
                    reply = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
                try:
                    f.write(json.dumps(reply).encode() + b"\n")
                    f.flush()
                except OSError:
                    pass

    def _on_hello(self, hello: dict) -> dict:
        """Admit A rank ``hello['a']``: check protocol, memory, geometry and model, and that no other host
        holds the rank; map its ResultArena; new epoch."""
        g = self.g
        if hello.get("proto") != PROTO_VERSION:
            raise HandshakeError(
                f"protocol version A={hello.get('proto')} F={PROTO_VERSION}"
            )
        if hello.get("memory") != self.mem.kind:
            raise HandshakeError(
                f"transport memory A={hello.get('memory')} F={self.mem.kind} ({self.mem.reason}): "
                "give both roles the same --asap-transport-mem"
            )
        self.mem.check_peer(hello.get("peer") or {})
        theirs = hello.get("geometry", {})
        diff = [
            f"{k}: A={theirs.get(k)} F={v}"
            for k, v in g.as_dict().items()
            if theirs.get(k) != v
        ]
        if diff:
            raise HandshakeError("geometry mismatch (" + ", ".join(diff) + ")")
        if self.model is not None:
            got = hello.get("model") or {}
            diff = [
                f"{k}: A={got.get(k)} F={v}"
                for k, v in self.model.items()
                if got.get(k) != v
            ]
            if diff:
                raise HandshakeError(
                    "the A model and the experts served here differ ("
                    + ", ".join(diff)
                    + ")"
                )
        a = int(hello["a"])
        if not 0 <= a < g.num_a:
            raise HandshakeError(f"A rank {a} is outside [0, {g.num_a})")
        host = (hello.get("peer") or {}).get("host")
        with self._lock:
            old = self.peers.get(a)
        if old is not None and old.host != host:  # not a reconnect: two A instances number a rank alike
            raise HandshakeError(
                f"A rank {a} is connected from {old.host} (pid {old.pid}), and {host} claims it too: "
                "the attention instances sharing the FFN ranks need disjoint --asap-attn-base-replica ranges"
            )
        self.device.activate()
        result = self.mem.open(hello["result"])  # before the old mapping goes: this may fail
        with self._lock:
            old = self.peers.get(a)
            if old is not None:  # a restarted or reconnecting A rank
                old.mapping.release()
                self.stats["reconnects"] += 1
            peer = Peer(
                a,
                result.ptr,
                (old.epoch % MAX_EPOCH) + 1 if old else 1,
                int(hello.get("pid", 0)),
                result,
                host,
            )
            self.peers[a] = peer
            lo = a * g.slots
            self._bells[lo : lo + g.slots] = self.seq_base
            self._last[lo : lo + g.slots] = self.seq_base
            for s in range(g.slots):
                self._pending.pop(
                    (a // g.group, s), None
                )  # half-assembled jobs of the old incarnation
        return {
            "ok": True,
            "f": self.f,
            "epoch": peer.epoch,
            "seq_base": self.seq_base,
            "memory": self.mem.kind,
            "recv": self._recv.desc,
            "doorbells": self._page.desc,
            **self.info,
        }

    # ---- discovery (serving thread) ----
    def poll(self) -> List[Job]:
        """Jobs whose rows have all arrived since the last call; never blocks."""
        g, jobs = self.g, []
        with self._lock:
            words = self._bells.copy()
            seq = (words & np.uint64(SEQ_MASK)).astype(np.uint32)
            ahead = (seq - self._last).view(np.int32)  # cyclic difference: wrap-safe
            ready = np.flatnonzero(ahead > 0)
            if (
                ready.size > 1
            ):  # start after the last served index: no rank is always first
                ready = np.concatenate(
                    [ready[ready >= self._rr], ready[ready < self._rr]]
                )
            for idx in ready.tolist():
                a, s = divmod(idx, g.slots)
                self.stats["lost_messages"] += (
                    int(ahead[idx]) - 1
                )  # overwritten unread; 0 with backpressure
                self._last[idx] = seq[idx]
                self._rr = idx + 1
                row = self._row(a, s, int(words[idx]))
                if row is None:
                    continue
                rows = self._pending.setdefault((a // g.group, s), {})
                rows[a % g.group] = row
                if len(rows) == g.group:
                    job = self._job(
                        a // g.group, s, self._pending.pop((a // g.group, s))
                    )
                    if job is not None:
                        jobs.append(job)
        return jobs

    def _row(self, a: int, s: int, word: int) -> Optional[Row]:
        """Decode a doorbell; None (and a stats counter) for unknown peers, old epochs, bad sizes."""
        seq, n_tokens, layer_id, epoch = unpack_doorbell(word)
        peer = self.peers.get(a)
        if peer is None:
            self.stats["unknown_peer"] += 1
            return None
        if epoch != peer.epoch:  # sent before A's last reconnect
            self.stats["stale_epoch"] += 1
            return None
        if not 1 <= n_tokens <= self.g.n_max:
            self.stats["bad_doorbell"] += 1
            return None
        base = self.recv_base + self.g.region_off(a, s)
        return Row(
            a,
            s,
            seq,
            epoch,
            layer_id,
            n_tokens,
            base,
            base + self.g.ids_off,
            base + self.g.w_off,
        )

    def _job(self, replica: int, slot: int, rows: Dict[int, Row]) -> Optional[Job]:
        """Join one replica's T rows into a job; they must name the same layer."""
        ordered = [rows[j] for j in range(self.g.group)]
        if (
            len({r.layer_id for r in ordered}) > 1
        ):  # the TP ranks of one replica disagree: a protocol bug
            self.stats["group_mismatch"] += 1
            return None
        self.stats["jobs"] += 1
        if self._prev_replica is not None and replica != self._prev_replica:
            self.stats["replica_switches"] += (
                1  # > 0 means replicas interleave instead of taking turns
            )
        self._prev_replica = replica
        self.trace.append((replica, slot, ordered[0].layer_id))
        return Job(replica, slot, ordered[0].layer_id, ordered, time.monotonic_ns())

    # ---- answer (serving thread) ----
    def reply(self, row: Row, src_ptr: int, nbytes: int, stream: int) -> None:
        """Enqueue on ``stream``: copy the partial sum (nbytes at src_ptr) into the A rank's segment, then its
        done flag. Dropped if that A rank reconnected since the message was sent."""
        peer, g = self.peers.get(row.a), self.g
        if peer is None or peer.epoch != row.epoch:
            self.stats["dropped_replies"] += 1
            return
        cu.memcpy_async(
            peer.result_base + g.seg_off(self.f, row.slot), src_ptr, nbytes, stream
        )
        cu.write32(stream, peer.result_base + g.done_off(self.f, row.slot), row.seq)

    def serve_forever(
        self,
        handler: Callable[[Job], None],
        stop: threading.Event,
        spin_s: float = 0.005,
        idle_sleep_s: float = 0.0002,
    ) -> None:
        """Poll and hand every complete job to ``handler`` until ``stop`` is set.

        After the last job the poller spins for ``spin_s`` before it starts napping ``idle_sleep_s`` per round:
        a napping poller adds up to one nap (sleep(200us) lasts ~253 us here) to the detection of the next
        doorbell, and the next layer's doorbell of a busy A arrives within a few ms.
        """
        sys.setswitchinterval(
            1e-5
        )  # a spinning poller would otherwise starve the handshake thread
        self.device.activate()
        last_job = time.monotonic()
        while not stop.is_set():
            jobs = self.poll()
            for job in jobs:
                handler(job)
            if jobs:
                last_job = time.monotonic()
            elif idle_sleep_s and time.monotonic() - last_job > spin_s:
                time.sleep(idle_sleep_s)

    def close(self) -> None:
        """Stop the handshake thread, unmap the A arenas, free the doorbell page and the RecvArena."""
        self._hs_stop.set()
        self._hs_thread.join(timeout=2.0)
        self._srv.close()
        self.device.activate()
        for peer in self.peers.values():
            peer.mapping.release()
        self.peers.clear()
        self._bells = None  # a view of the doorbell page, which release() unmaps
        self._page.release()  # ipc: also removes the /dev/shm file
        self._recv.release()
