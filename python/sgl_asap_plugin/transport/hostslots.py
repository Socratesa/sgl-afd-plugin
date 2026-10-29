"""The /dev/shm slot file of one F rank: one doorbell word per (A rank, slot), a cache line each.

A's GPU writes the words (64-bit memop), F's CPU polls them with plain loads. Never poll
host-mapped memory with atomics or acquire loads: that hangs (spike finding).
"""

from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport.arena import HOST_SLOT_BYTES, Geometry


class HostSlots:
    def __init__(self, path: str, geometry: Geometry, *, create: bool):
        self.path, self.g = path, geometry
        fd = os.open(path, os.O_RDWR | (os.O_CREAT | os.O_TRUNC if create else 0), 0o600)
        try:
            if create:
                os.ftruncate(fd, geometry.host_bytes)
            self.mm = mmap.mmap(fd, geometry.host_bytes)
        finally:
            os.close(fd)
        self._buf = ctypes.c_char.from_buffer(self.mm)
        self.addr = ctypes.addressof(self._buf)
        self._words = np.frombuffer(self.mm, dtype=np.uint64)
        per_slot, n = HOST_SLOT_BYTES // 8, geometry.num_a * geometry.slots
        # word 0 of every cache line is the doorbell; a strided view, so copy() it (read()) before use
        self.doorbells = self._words[: n * per_slot].reshape(n, per_slot)[:, 0]
        self._registered = False

    def register_cuda(self) -> int:
        """Make the mapping writable by this process's GPU -> device base address."""
        base = cu.host_register(self.addr, self.g.host_bytes)
        self._registered = True
        return base

    def read(self) -> np.ndarray:
        return self.doorbells.copy()

    def reset_all(self, value: int) -> None:
        self.doorbells[:] = value

    def reset(self, a: int, value: int) -> None:
        lo = a * self.g.slots
        self.doorbells[lo:lo + self.g.slots] = value

    def close(self, unlink: bool = False) -> None:
        if self._registered:
            cu.host_unregister(self.addr)
            self._registered = False
        self.doorbells = self._words = None  # numpy views pin the mmap
        del self._buf
        self.mm.close()
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
