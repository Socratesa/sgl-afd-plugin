"""A side, part 1: connection lifecycle (arenas, handshake, watchdog, poisoning)."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport.arena import Geometry
from sgl_asap_plugin.transport.handshake import request
from sgl_asap_plugin.transport.hostslots import HostSlots
from sgl_asap_plugin.transport.pools import EventPool
from sgl_asap_plugin.transport.protocol import PROTO_VERSION, ChannelPoisoned, HandshakeError
from sgl_asap_plugin.transport.watchdog import Phase, Watchdog


class ChannelBase:
    def __init__(self, a: int, geometry: Geometry, endpoints: Sequence[Tuple[str, int]], *, device: int,
                 seq_base: int = 0, timeout_s: float = 30.0, connect_timeout_s: float = 60.0,
                 _backpressure: bool = True, _epoch_override: Optional[int] = None):
        """``_backpressure`` and ``_epoch_override`` exist for tests that must provoke protocol violations."""
        if len(endpoints) != geometry.num_f:
            raise ValueError(f"need one endpoint per F rank ({geometry.num_f}), got {len(endpoints)}")
        self.a, self.g, self.endpoints, self.seq_base = a, geometry, list(endpoints), seq_base
        self.device, self.timeout_s, self.connect_timeout_s = cu.Device(device), timeout_s, connect_timeout_s
        self._backpressure, self._epoch_override = _backpressure, _epoch_override
        self._poison: Optional[ChannelPoisoned] = None
        self._watchdog: Optional[Watchdog] = None
        self._events: Optional[EventPool] = None
        self._recv: List[int] = []
        self._host: List[HostSlots] = []

    def connect(self):
        g = self.g
        self.device.activate()
        self._events = EventPool()
        self._result = cu.mem_alloc(g.result_bytes)
        cu.memset32(self._result, self.seq_base, g.flags_bytes // 4)  # done flags start at seq_base
        hello = {"proto": PROTO_VERSION, "a": self.a, "geometry": g.as_dict(), "pid": os.getpid(),
                 "result_handle": cu.ipc_export(self._result).hex()}
        self._host_dev, self._epoch = [], []
        try:
            for host, port in self.endpoints:
                reply = request(host, port, hello, connect_timeout=self.connect_timeout_s)
                if reply["seq_base"] != self.seq_base:
                    raise HandshakeError(f"seq_base A={self.seq_base} F={reply['seq_base']}")
                self._recv.append(cu.ipc_open(bytes.fromhex(reply["recv_handle"])))
                slots = HostSlots(reply["shm_path"], g, create=False)
                self._host.append(slots)
                self._host_dev.append(slots.register_cuda())
                self._epoch.append(reply["epoch"])
        except BaseException:
            self.close()
            raise
        self._seq = [self.seq_base] * g.slots
        self._poison = None
        self._watchdog = Watchdog(self.device, self._events, self.timeout_s, self._on_timeout)
        self._watchdog.start()
        return self

    def reconnect(self):
        """New epoch, fresh result arena: the only way out of a poisoned channel."""
        self.close()
        return self.connect()

    def close(self) -> None:
        """Synchronize the streams that used this channel first."""
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        self.device.activate()
        for base in self._recv:
            cu.ipc_close(base)
        for slots in self._host:
            slots.close()
        self._recv, self._host = [], []
        if self._events is not None:
            self._events.close()
            cu.mem_free(self._result)
            self._events = None

    def _on_timeout(self, phase: Phase) -> None:
        self._poison = ChannelPoisoned(
            f"{phase.what} ({phase.detail}) did not complete within {self.timeout_s}s: F is dead or stuck. "
            "Slot ownership is unknown until reconnect().")

    def check(self) -> None:
        if self._poison is not None:
            raise self._poison

    def _done_addr(self, f: int, slot: int) -> int:
        return self._result + self.g.done_off(f, slot)
