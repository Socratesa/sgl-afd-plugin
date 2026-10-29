"""Host-side supervision of A's stream waits.

A stream blocked in a stream memop never times out by itself, so a dead F would hang the GPU
stream (and torch.cuda.synchronize) forever. The watchdog polls events recorded before and
after each group of waits. When a group has been waiting longer than the timeout, the
watchdog writes the awaited values itself from a separate stream, which releases every
blocked wait, and reports the timeout. Whatever the released stream reads afterwards is
garbage: the channel is poisoned and callers must check for the error before using results.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Tuple

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport import streams as st
from sgl_asap_plugin.transport.pools import EventPool


class Phase:
    """A group of stream waits: ``pre`` is recorded before them, ``post`` after them."""

    __slots__ = ("what", "pre", "post", "release", "detail", "t_start", "state")

    def __init__(self, what: str, pre: int, post: int, release: List[Tuple[int, int]], detail: str = ""):
        self.what, self.pre, self.post, self.release, self.detail = what, pre, post, release, detail
        self.t_start = None
        self.state = "pending"  # pending | done | failed


class Watchdog(threading.Thread):
    def __init__(self, device: cu.Device, events: EventPool, timeout_s: float, on_timeout: Callable[[Phase], None]):
        super().__init__(name="asap-watchdog", daemon=True)
        self.device, self.events, self.timeout_s, self.on_timeout = device, events, timeout_s, on_timeout
        self._phases: List[Phase] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._recovery = None

    def add(self, phase: Phase) -> None:
        with self._lock:
            self._phases.append(phase)

    def run(self) -> None:
        self.device.activate()
        self._recovery = st.stream_create()
        while not self._stop_evt.is_set():
            self._scan()
            time.sleep(0.002)
        st.stream_destroy(self._recovery)

    def _scan(self) -> None:
        now = time.monotonic()
        with self._lock:
            phases = list(self._phases)
        for ph in phases:
            if ph.t_start is None and st.event_done(ph.pre):
                ph.t_start = now
            if st.event_done(ph.post):
                if ph.state == "pending":
                    ph.state = "done"
                self._retire(ph)
            elif ph.state == "pending" and ph.t_start is not None and now - ph.t_start > self.timeout_s:
                self._fail_all(ph)
                return

    def _fail_all(self, culprit: Phase) -> None:
        """Release every pending wait (not just the culprit's) so no stream stays blocked."""
        self.on_timeout(culprit)
        with self._lock:
            phases = list(self._phases)
        for ph in phases:
            if ph.state == "pending":
                ph.state = "failed"
                for addr, value in ph.release:
                    st.write32(self._recovery, addr, value)

    def _retire(self, ph: Phase) -> None:
        with self._lock:
            if ph not in self._phases:
                return
            self._phases.remove(ph)
        self.events.release(ph.pre)
        self.events.release(ph.post)

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=5.0)
