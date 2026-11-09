"""Where the transport's shared memory lives and how a peer maps it: two backends under one protocol (layout.py).

Every message goes into memory owned by its receiver: F rank f owns its RecvArena (device) and its doorbell page
(host memory its CPU polls), A rank a its ResultArena (device). The owner allocates it and sends a descriptor (a
small dict) in the handshake; the peer opens the descriptor and gets an address its GPU writes through NVLink.

* ``ipc``: legacy CUDA IPC (cuMemAlloc + cuIpcGetMemHandle) for device memory; the doorbell page is a /dev/shm file
  that the peer maps and registers with cuMemHostRegister. Within one OS whose processes see the same GPUs (same
  CUDA_VISIBLE_DEVICES): H100 / B200 nodes, or one GB200 tray.
* ``fabric``: CUDA VMM allocations created exportable as fabric handles (CU_MEM_HANDLE_TYPE_FABRIC). The doorbell
  page is memory of the host NUMA node next to F's GPU (CU_MEM_LOCATION_TYPE_HOST_NUMA), mapped for F's CPU, which
  polls it with plain loads as before. Any GPUs of one NVLink domain, across OS instances: GB200 / GB300 trays,
  the same tray or another one. Needs the IMEX service and an IMEX channel (/dev/nvidia-caps-imex-channels).

The handle type of an allocation is fixed when it is created and every arena is shared with all peers, so one
deployment uses one backend. ``auto`` takes fabric on a GPU of a multi-node NVLink fabric where a trial allocation
works, else ipc. On GB200 both move data at the same NVLink bandwidth (net/README.md).
"""

from __future__ import annotations

import ctypes
import mmap
import os
import socket
from typing import Callable, Optional, Tuple

from sgl_asap_plugin.transport import cuda_wrapper as cu
from sgl_asap_plugin.transport.layout import HandshakeError, TransportError, align_up

KINDS = ("auto", "ipc", "fabric")


class Shared:
    """Shared memory as this process sees it. ``ptr``: the device address, except for an own doorbell page, where
    it is the CPU address (the one the poller reads). ``desc`` (owner only): what a peer opens it from."""

    def __init__(self, ptr: int, release: Callable[[], None], desc: Optional[dict] = None):
        self.ptr, self.desc, self._release = ptr, desc, release

    def release(self) -> None:
        """Free it (owner) or unmap it (peer); idempotent."""
        release, self._release = self._release, None
        if release is not None:
            release()


class IpcMemory:
    """Legacy CUDA IPC and a /dev/shm file: one OS (module docstring)."""

    kind = "ipc"

    def __init__(self, ordinal: int, reason: str = "requested"):
        self.ordinal, self.reason = ordinal, reason

    def identity(self) -> dict:
        return {"host": socket.gethostname()}

    def check_peer(self, peer: dict) -> None:
        mine = socket.gethostname()
        if peer.get("host") not in (None, mine):
            raise HandshakeError(
                f"the ipc transport needs one host, but the peer runs on {peer['host']} and this rank on "
                f"{mine}: between GB200 trays use --asap-transport-mem fabric"
            )

    def device(self, nbytes: int) -> Shared:
        ptr = cu.mem_alloc(nbytes)
        try:
            handle = cu.ipc_export(ptr).hex()
        except BaseException:
            cu.mem_free(ptr)
            raise
        return Shared(ptr, lambda: cu.mem_free(ptr), {"ipc": handle})

    def doorbells(self, nbytes: int, path: str) -> Shared:
        """A zeroed file of nbytes at ``path`` (in /dev/shm), mapped here; release() also removes it."""
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, nbytes)
            mm = mmap.mmap(fd, nbytes)
        finally:
            os.close(fd)

        def release():
            mm.close()
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        return Shared(_address(mm), release, {"path": path})

    def open(self, desc: dict) -> Shared:
        ptr = cu.ipc_open(bytes.fromhex(desc["ipc"]))
        return Shared(ptr, lambda: cu.ipc_close(ptr))

    def open_doorbells(self, desc: dict, nbytes: int) -> Shared:
        """mmap the owner's /dev/shm file and register it, so that this GPU can write the doorbells."""
        fd = os.open(desc["path"], os.O_RDWR)
        try:
            mm = mmap.mmap(fd, nbytes)
        finally:
            os.close(fd)
        addr = _address(mm)
        try:
            ptr = cu.host_register(addr, nbytes)
        except BaseException:
            mm.close()
            raise

        def release():
            cu.host_unregister(addr)
            mm.close()

        return Shared(ptr, release)


def _address(mm: mmap.mmap) -> int:
    buf = ctypes.c_char.from_buffer(mm)
    addr = ctypes.addressof(buf)
    del buf  # the address stays valid while the mapping is open; a live export would block mm.close()
    return addr


class FabricMemory:
    """CUDA VMM allocations shared as fabric handles: one NVLink domain (module docstring)."""

    kind = "fabric"

    def __init__(
        self, ordinal: int, domain: Optional[Tuple[str, int]] = None, reason: str = "requested"
    ):
        """``domain``: this GPU's (ClusterUUID, CliqueId), see nvlink_domain."""
        self.ordinal, self.reason = ordinal, reason
        self.numa = cu.attribute(cu.ATTR_HOST_NUMA_ID, ordinal)
        self.unit = max(
            cu.fabric_granularity(ordinal), cu.fabric_granularity(ordinal, self.numa)
        )
        self.domain = domain if domain is not None else nvlink_domain(ordinal)

    def identity(self) -> dict:
        ident = {"host": socket.gethostname()}
        if self.domain is not None:
            ident["cluster"], ident["clique"] = self.domain
        return ident

    def check_peer(self, peer: dict) -> None:
        theirs = (peer.get("cluster"), peer.get("clique"))
        if self.domain is not None and None not in theirs and theirs != self.domain:
            raise HandshakeError(
                f"the peer's GPU is in NVLink domain {theirs}, this rank's in {self.domain}: fabric "
                "memory is shared within one NVLink domain (same ClusterUUID and CliqueId)"
            )

    def device(self, nbytes: int) -> Shared:
        return self._create(nbytes, None)

    def doorbells(self, nbytes: int, path: str) -> Shared:
        """Memory of the host NUMA node next to this GPU, mapped for that node's CPUs (``ptr``) and this GPU.
        ``path`` is the ipc backend's."""
        return self._create(nbytes, self.numa)

    def open(self, desc: dict) -> Shared:
        size = int(desc["bytes"])
        handle = cu.fabric_import(bytes.fromhex(desc["fabric"]))
        return self._mapped(handle, size, None)

    def open_doorbells(self, desc: dict, nbytes: int) -> Shared:
        return self.open(desc)

    def _create(self, nbytes: int, numa: Optional[int]) -> Shared:
        size = align_up(nbytes, self.unit)
        handle = cu.fabric_create(size, self.ordinal, numa)
        try:
            raw = cu.fabric_export(handle)
        except BaseException:
            cu.vmm_release(handle)
            raise
        shared = self._mapped(handle, size, numa)
        shared.desc = {"fabric": raw.hex(), "bytes": size}
        return shared

    def _mapped(self, handle: int, size: int, numa: Optional[int]) -> Shared:
        """Map an allocation handle for this GPU (and with ``numa``, that node's CPUs); release() unmaps it and
        drops the handle. The handle is dropped here if the mapping fails."""
        try:
            ptr = cu.vmm_map(handle, size, self.ordinal, numa)
        except BaseException:
            cu.vmm_release(handle)
            raise

        def release():
            cu.vmm_unmap(ptr, size)
            cu.vmm_release(handle)

        return Shared(ptr, release)

def nvlink_domain(ordinal: int) -> Optional[Tuple[str, int]]:
    """(ClusterUUID hex, CliqueId) of the NVLink fabric that device ``ordinal`` joined, from NVML; None if it
    joined none or NVML (pynvml) cannot tell. GPUs with the same pair reach each other over NVLink."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            info = pynvml.c_nvmlGpuFabricInfo_t()
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(cu.pci_bus_id(ordinal))
            pynvml.nvmlDeviceGetGpuFabricInfo(handle, ctypes.byref(info))
        finally:
            pynvml.nvmlShutdown()
    except (pynvml.NVMLError, AttributeError):
        return None
    completed = getattr(pynvml, "NVML_GPU_FABRIC_STATE_COMPLETED", 3)
    if info.state != completed or not any(info.clusterUuid):
        return None
    return bytes(info.clusterUuid).hex(), int(info.cliqueId)


def fabric_unavailable(ordinal: int) -> Optional[str]:
    """Why device ``ordinal`` cannot use the fabric backend, None if it can. Its context must be current."""
    if not cu.attribute(cu.ATTR_HANDLE_TYPE_FABRIC, ordinal, default=0):
        return "the GPU does not support fabric handles"
    if not cu.attribute(cu.ATTR_HOST_NUMA_MULTINODE_IPC, ordinal, default=0):
        return "host memory (the doorbell page) cannot be shared across nodes"
    if cu.attribute(cu.ATTR_HOST_NUMA_ID, ordinal, default=-1) < 0:
        return "the GPU has no host NUMA node"
    domain = nvlink_domain(ordinal)
    if domain is None:
        return "NVML reports no completed NVLink fabric for the GPU (or pynvml is missing)"
    try:  # what the IMEX channel and the container's NUMA support decide
        mem = FabricMemory(ordinal, domain)
        for numa in (None, mem.numa):
            own = mem._create(mem.unit, numa)
            try:
                mem.open(own.desc).release()
            finally:
                own.release()
    except cu.CUDAError as exc:
        return (
            f"a trial fabric allocation failed ({exc}); check the IMEX channel "
            "(/dev/nvidia-caps-imex-channels) and the container's NUMA support"
        )
    return None


def select(kind: str, ordinal: int):
    """The backend of ``kind`` (``KINDS``) on device ``ordinal``, whose context must be current. ``reason`` of
    the result says why it was chosen."""
    if kind not in KINDS:
        raise ValueError(f"transport memory must be one of {KINDS}, got {kind!r}")
    if kind == "ipc":
        return IpcMemory(ordinal)
    why = fabric_unavailable(ordinal)
    if why is None:
        return FabricMemory(ordinal, reason="requested" if kind == "fabric" else "auto")
    if kind == "fabric":
        raise TransportError(f"the fabric transport is not usable on GPU {ordinal}: {why}")
    return IpcMemory(ordinal, reason=f"auto: {why}")
