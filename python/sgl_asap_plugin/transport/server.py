"""F side: owns a RecvArena, discovers messages by polling the slot file, replies with partial sums."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport import streams as st
from sgl_asap_plugin.transport.arena import Geometry
from sgl_asap_plugin.transport.handshake import HandshakeServer
from sgl_asap_plugin.transport.hostslots import HostSlots
from sgl_asap_plugin.transport.protocol import MAX_EPOCH, PROTO_VERSION, Doorbell, HandshakeError


@dataclass
class Peer:
    a: int
    result_base: int  # A's ResultArena, opened through IPC
    epoch: int
    pid: int


@dataclass
class Row:
    a: int
    j: int  # TP row inside the A replica
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
    group: int  # A replica
    slot: int
    layer_id: int
    rows: List[Row]  # one per TP row, ordered by j
    t_detect_ns: int


class FfnServer:
    def __init__(self, f_rank: int, geometry: Geometry, *, device: int, run_dir: str, host: str = "127.0.0.1",
                 port: int = 0, seq_base: int = 0, info: Optional[dict] = None):
        """``info`` is what a ``describe`` request returns besides the geometry (what this F serves)."""
        self.f, self.g, self.seq_base, self.info = f_rank, geometry, seq_base, dict(info or {})
        self.device, self.run_dir, self.host, self._port_req = cu.Device(device), run_dir, host, port
        self.stats: Counter = Counter()
        self.service_order: List[tuple] = []
        self.peers: Dict[int, Peer] = {}
        self._pending: Dict[tuple, Dict[int, Row]] = {}
        self._lock = threading.RLock()

    def start(self) -> "FfnServer":
        g = self.g
        self.device.activate()
        self.recv_base = cu.mem_alloc(g.recv_bytes)
        cu.memset8(self.recv_base, 0, g.recv_bytes)
        self.recv_handle = cu.ipc_export(self.recv_base)
        self.shm_path = os.path.join(self.run_dir, f"asap_f{self.f}.slots")
        self.slots = HostSlots(self.shm_path, g, create=True)
        self.slots.reset_all(self.seq_base)
        self._last = np.full(g.num_a * g.slots, self.seq_base, dtype=np.uint32)
        self._hs = HandshakeServer(self.host, self._port_req, self._on_hello)
        self.port = self._hs.port
        return self

    # ---- handshake (runs on the handshake thread) ----
    def _on_hello(self, hello: dict) -> dict:
        g = self.g
        if hello.get("op") == "describe":  # no side effects: A asks what is served before it builds its model
            return {"ok": True, "f": self.f, "num_f": g.num_f, "proto": PROTO_VERSION, "geometry": g.as_dict(),
                    **self.info}
        if hello.get("proto") != PROTO_VERSION:
            raise HandshakeError(f"protocol version {hello.get('proto')} != {PROTO_VERSION}")
        theirs, ours = hello.get("geometry", {}), g.as_dict()
        diff = [f"{k}: A={theirs.get(k)} F={v}" for k, v in ours.items() if theirs.get(k) != v]
        if diff:
            raise HandshakeError("geometry mismatch (" + ", ".join(diff) + ")")
        a = int(hello["a"])
        if not 0 <= a < g.num_a:
            raise HandshakeError(f"A rank {a} is outside [0, {g.num_a})")
        self.device.activate()
        with self._lock:
            old = self.peers.get(a)
            if old is not None:
                cu.ipc_close(old.result_base)
                self.stats["reconnects"] += 1
            peer = Peer(a, cu.ipc_open(bytes.fromhex(hello["result_handle"])),
                        (old.epoch % MAX_EPOCH) + 1 if old else 1, int(hello.get("pid", 0)))
            self.peers[a] = peer
            self.slots.reset(a, self.seq_base)
            self._last[a * g.slots:(a + 1) * g.slots] = self.seq_base
            for s in range(g.slots):
                self._pending.pop((a // g.group, s), None)  # half-assembled jobs of the old incarnation
        return {"ok": True, "f": self.f, "epoch": peer.epoch, "seq_base": self.seq_base,
                "recv_handle": self.recv_handle.hex(), "shm_path": self.shm_path}

    # ---- discovery ----
    def poll(self) -> List[Job]:
        """Return the jobs whose rows have all arrived since the last call (non-blocking)."""
        g, jobs = self.g, []
        with self._lock:
            words = self.slots.read()
            seq = (words & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            diff = (seq - self._last).view(np.int32)
            for idx in np.flatnonzero(diff > 0):
                a, s = divmod(int(idx), g.slots)
                self.stats["lost_messages"] += int(diff[idx]) - 1  # backpressure keeps this at zero
                self._last[idx] = seq[idx]
                row = self._read_row(a, s, int(words[idx]))
                if row is None:
                    continue
                rows = self._pending.setdefault((a // g.group, s), {})
                rows[row.j] = row
                if len(rows) == g.group:
                    job = self._make_job(a // g.group, s, self._pending.pop((a // g.group, s)))
                    if job is not None:
                        jobs.append(job)
        return jobs

    def _read_row(self, a: int, s: int, word: int):
        g, peer, bell = self.g, self.peers.get(a), Doorbell.unpack(word)
        if peer is None:
            self.stats["unknown_peer"] += 1
            return None
        if bell.epoch != peer.epoch:
            self.stats["stale_epoch"] += 1
            return None
        if not 1 <= bell.n_tokens <= g.n_max:
            self.stats["bad_doorbell"] += 1
            return None
        base = self.recv_base + g.region_off(a, s)
        return Row(a=a, j=a % g.group, slot=s, seq=bell.seq, epoch=bell.epoch, layer_id=bell.layer_id,
                   n_tokens=bell.n_tokens, x_ptr=base, ids_ptr=base + g.ids_off, w_ptr=base + g.w_off)

    def _make_job(self, group: int, slot: int, rows: Dict[int, Row]):
        ordered = [rows[j] for j in range(self.g.group)]
        if len({r.layer_id for r in ordered}) > 1:
            self.stats["group_mismatch"] += 1
            return None
        self.stats["jobs"] += 1
        return Job(group, slot, ordered[0].layer_id, ordered, time.monotonic_ns())

    # ---- reply ----
    def reply(self, row: Row, src_ptr: int, nbytes: int, stream: int) -> None:
        """Enqueue on ``stream``: copy the partial sum into A's result segment, then set its done flag."""
        peer, g = self.peers.get(row.a), self.g
        if peer is None or peer.epoch != row.epoch:
            self.stats["dropped_replies"] += 1  # that A reconnected meanwhile
            return
        cu.memcpy_async(peer.result_base + g.seg_off(self.f, row.slot), src_ptr, nbytes, stream)
        st.write32(stream, peer.result_base + g.done_off(self.f, row.slot), row.seq)

    def serve_forever(self, handler, stop: threading.Event, spin_s: float = 0.005,
                      idle_sleep_s: float = 0.0002) -> None:
        """Poll, hand every complete job to ``handler``. The poller spins for ``spin_s`` after the last job and
        only then naps ``idle_sleep_s`` per round: a sleeping poller adds up to a full nap (about 250 us here,
        since sleep(200us) lasts 253 us) to the detection of the next doorbell, on every layer."""
        sys.setswitchinterval(1e-5)  # a busy poller would otherwise starve the handshake thread
        self.device.activate()
        last_job = time.monotonic()
        while not stop.is_set():
            jobs = self.poll()
            for job in jobs:
                self.service_order.append((job.group, job.slot))
                handler(job)
            if jobs:
                last_job = time.monotonic()
            elif idle_sleep_s and time.monotonic() - last_job > spin_s:
                time.sleep(idle_sleep_s)

    def close(self) -> None:
        self._hs.close()
        self.device.activate()
        for peer in self.peers.values():
            cu.ipc_close(peer.result_base)
        self.peers.clear()
        self.slots.close(unlink=True)
        cu.mem_free(self.recv_base)
