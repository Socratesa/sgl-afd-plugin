"""The CUDA driver calls of the transport, through ctypes on libcuda.

Raw driver memory instead of torch tensors, because the arenas are shared (memory.py: CUDA IPC, or VMM allocations
exported as fabric handles) with processes that never see our torch allocator, and must not move or be cached. No
torch import: any process with a CUDA context can run the transport. Every call acts on the calling thread's
current context; ``Device.activate`` makes the device's primary context current, which is the one torch uses too.

Stream memory operations (write32 / write64 / wait32) are the signalling primitive: they use no SM, block only
their own stream, and their value is fixed when they are enqueued (so they cannot advance across CUDA graph
replays). A write has a full memory barrier: everything earlier in the stream is visible before the value is.
(cuda.bindings.driver offers the same calls, but its IPC handle type cannot be turned into bytes without ctypes,
and a ctypes call is ~1 us cheaper; sglang's multimodal IPC transport binds these memops with ctypes too.)
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_size_t,
    c_ubyte,
    c_uint,
    c_uint32,
    c_uint64,
    c_ushort,
    c_void_p,
)
from typing import Optional

_lib = ctypes.CDLL("libcuda.so.1")

_IPC_LAZY_PEER = 1  # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
_HOST_REGISTER = 1 | 2  # CU_MEMHOSTREGISTER_PORTABLE | DEVICEMAP
_WAIT_GEQ = 0  # CU_STREAM_WAIT_VALUE_GEQ: (int32)(*addr - value) >= 0, wrap-safe
_NOT_READY = 600  # CUDA_ERROR_NOT_READY
_EVENT_DISABLE_TIMING = 2
_STREAM_NON_BLOCKING = 1
_ALLOC_PINNED = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
_HANDLE_FABRIC = 8  # CU_MEM_HANDLE_TYPE_FABRIC
_LOC_DEVICE, _LOC_HOST_NUMA = 1, 3  # CU_MEM_LOCATION_TYPE_DEVICE / _HOST_NUMA
_ACCESS_RW = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
_GRANULARITY_MIN = 0  # CU_MEM_ALLOC_GRANULARITY_MINIMUM
ATTR_HANDLE_TYPE_FABRIC = 128  # CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
ATTR_HOST_NUMA_ID = 134  # CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
ATTR_HOST_NUMA_MULTINODE_IPC = 143  # CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED


class CUDAError(RuntimeError):
    def __init__(self, name: str, code: int):
        text = c_char_p()
        _lib.cuGetErrorName(code, byref(text))
        super().__init__(
            f"{name} failed: {code} {text.value.decode() if text.value else ''}"
        )
        self.code = code


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", c_ubyte * 64)]  # c_char would stop at the first NUL byte


# The VMM structs of cuda.h (CUmemLocation, CUmemAllocationProp, CUmemAccessDesc); a fabric handle is 64 bytes.
class _MemLocation(ctypes.Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", c_ubyte),
        ("gpuDirectRDMACapable", c_ubyte),
        ("usage", c_ushort),
        ("reserved", c_ubyte * 4),
    ]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", c_int)]


_FabricHandle = c_ubyte * 64


def _bind(name: str, *argtypes):
    """A callable for driver function ``name`` that raises CUDAError on a non-zero result."""
    fn = getattr(_lib, name)
    fn.argtypes, fn.restype = list(argtypes), c_int

    def call(*args):
        code = fn(*args)
        if code != 0:
            raise CUDAError(name, code)

    return call


_init = _bind("cuInit", c_uint)
_device_get = _bind("cuDeviceGet", POINTER(c_int), c_int)
_primary_retain = _bind("cuDevicePrimaryCtxRetain", POINTER(c_void_p), c_int)
_ctx_set = _bind("cuCtxSetCurrent", c_void_p)
_alloc = _bind("cuMemAlloc_v2", POINTER(c_uint64), c_size_t)
_free = _bind("cuMemFree_v2", c_uint64)
_memset8 = _bind("cuMemsetD8_v2", c_uint64, c_ubyte, c_size_t)
_memset32 = _bind("cuMemsetD32_v2", c_uint64, c_uint32, c_size_t)
_memcpy_async = _bind("cuMemcpyAsync", c_uint64, c_uint64, c_size_t, c_void_p)
_ipc_get = _bind("cuIpcGetMemHandle", POINTER(_IpcHandle), c_uint64)
_ipc_open = _bind("cuIpcOpenMemHandle_v2", POINTER(c_uint64), _IpcHandle, c_uint)
_ipc_close = _bind("cuIpcCloseMemHandle", c_uint64)
_host_register = _bind("cuMemHostRegister_v2", c_void_p, c_size_t, c_uint)
_host_unregister = _bind("cuMemHostUnregister", c_void_p)
_host_devptr = _bind(
    "cuMemHostGetDevicePointer_v2", POINTER(c_uint64), c_void_p, c_uint
)
_stream_create = _bind("cuStreamCreate", POINTER(c_void_p), c_uint)
_stream_destroy = _bind("cuStreamDestroy_v2", c_void_p)
_stream_sync = _bind("cuStreamSynchronize", c_void_p)
_write32 = _bind("cuStreamWriteValue32_v2", c_void_p, c_uint64, c_uint32, c_uint)
_write64 = _bind("cuStreamWriteValue64_v2", c_void_p, c_uint64, c_uint64, c_uint)
_wait32 = _bind("cuStreamWaitValue32_v2", c_void_p, c_uint64, c_uint32, c_uint)
_event_create = _bind("cuEventCreate", POINTER(c_void_p), c_uint)
_event_record = _bind("cuEventRecord", c_void_p, c_void_p)
_event_destroy = _bind("cuEventDestroy_v2", c_void_p)
_event_query = _lib.cuEventQuery  # NOT_READY is an answer, not an error
_event_query.argtypes, _event_query.restype = [c_void_p], c_int
_attribute = _bind("cuDeviceGetAttribute", POINTER(c_int), c_int, c_int)
_pci_bus_id = _bind("cuDeviceGetPCIBusId", c_char_p, c_int, c_int)
_granularity = _bind(
    "cuMemGetAllocationGranularity", POINTER(c_size_t), POINTER(_AllocationProp), c_int
)
_create = _bind(
    "cuMemCreate", POINTER(c_uint64), c_size_t, POINTER(_AllocationProp), c_uint64
)
_release = _bind("cuMemRelease", c_uint64)
_export = _bind("cuMemExportToShareableHandle", c_void_p, c_uint64, c_int, c_uint64)
_import = _bind("cuMemImportFromShareableHandle", POINTER(c_uint64), c_void_p, c_int)
_reserve = _bind(
    "cuMemAddressReserve", POINTER(c_uint64), c_size_t, c_size_t, c_uint64, c_uint64
)
_unreserve = _bind("cuMemAddressFree", c_uint64, c_size_t)
_map = _bind("cuMemMap", c_uint64, c_size_t, c_size_t, c_uint64, c_uint64)
_unmap = _bind("cuMemUnmap", c_uint64, c_size_t)
_set_access = _bind(
    "cuMemSetAccess", c_uint64, c_size_t, POINTER(_AccessDesc), c_size_t
)


class DeviceBuffer:
    """Raw device memory as a CUDA-array-interface object: ``torch.as_tensor(DeviceBuffer(ptr, n), device=...)``
    aliases it as a uint8 tensor without copying or owning it (the arena stays owned by the transport)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
        }


class Device:
    """The primary context of one device, retained for the lifetime of the object."""

    def __init__(self, ordinal: int):
        _init(0)
        dev, ctx = c_int(), c_void_p()
        _device_get(byref(dev), ordinal)
        _primary_retain(byref(ctx), dev.value)
        self.ordinal, self.ctx = ordinal, ctx.value

    def activate(self) -> None:
        _ctx_set(self.ctx)


# ---- device memory and CUDA IPC ----
def mem_alloc(nbytes: int) -> int:
    ptr = c_uint64()
    _alloc(byref(ptr), nbytes)
    return ptr.value


def mem_free(ptr: int) -> None:
    _free(ptr)


def memset8(ptr: int, value: int, nbytes: int) -> None:
    _memset8(ptr, value, nbytes)


def memset32(ptr: int, value: int, count: int) -> None:
    _memset32(ptr, value & 0xFFFFFFFF, count)


def memcpy_async(dst: int, src: int, nbytes: int, stream: int) -> None:
    _memcpy_async(dst, src, nbytes, stream)


def ipc_export(dptr: int) -> bytes:
    """64-byte handle of a cuMemAlloc'ed buffer; another process maps it with ipc_open."""
    handle = _IpcHandle()
    _ipc_get(byref(handle), dptr)
    return bytes(handle.reserved)


def ipc_open(raw: bytes) -> int:
    """Map another process's buffer into this context (lazy peer access) -> device address."""
    handle = _IpcHandle()
    ctypes.memmove(byref(handle), raw, 64)
    ptr = c_uint64()
    _ipc_open(byref(ptr), handle, _IPC_LAZY_PEER)
    return ptr.value


def ipc_close(dptr: int) -> None:
    _ipc_close(dptr)


# ---- host memory the GPU can write ----
def host_register(host_ptr: int, nbytes: int) -> int:
    """Pin an existing host mapping (e.g. an mmap of /dev/shm) for device access -> its device address."""
    _host_register(host_ptr, nbytes, _HOST_REGISTER)
    dev = c_uint64()
    _host_devptr(byref(dev), host_ptr, 0)
    return dev.value


def host_unregister(host_ptr: int) -> None:
    _host_unregister(host_ptr)


# ---- streams, stream memory operations, events ----
def stream_create() -> int:
    """A non-blocking stream: it does not synchronize with the legacy default stream."""
    handle = c_void_p()
    _stream_create(byref(handle), _STREAM_NON_BLOCKING)
    return handle.value


def stream_destroy(stream: int) -> None:
    _stream_destroy(stream)


def stream_sync(stream: int) -> None:
    _stream_sync(stream)


def write32(stream: int, addr: int, value: int) -> None:
    """Enqueue ``*(u32*)addr = value`` after everything earlier in ``stream`` is visible."""
    _write32(stream, addr, value & 0xFFFFFFFF, 0)


def write64(stream: int, addr: int, value: int) -> None:
    """Enqueue one atomic 64-bit store, same ordering as write32."""
    _write64(stream, addr, value & 0xFFFFFFFFFFFFFFFF, 0)


def wait32(stream: int, addr: int, value: int) -> None:
    """Enqueue 'block ``stream`` until (int32)(*addr - value) >= 0'. Never times out by itself (see Watchdog)."""
    _wait32(stream, addr, value & 0xFFFFFFFF, _WAIT_GEQ)


def event_create() -> int:
    handle = c_void_p()
    _event_create(byref(handle), _EVENT_DISABLE_TIMING)
    return handle.value


def event_record(event: int, stream: int) -> None:
    _event_record(event, stream)


def event_done(event: int) -> bool:
    """Non-blocking: has the stream passed this event? (NOT_READY means not yet.)"""
    code = _event_query(event)
    if code == 0:
        return True
    if code == _NOT_READY:
        return False
    raise CUDAError("cuEventQuery", code)


def event_destroy(event: int) -> None:
    _event_destroy(event)


# ---- device attributes ----
def attribute(attr: int, ordinal: int, default: Optional[int] = None) -> int:
    """cuDeviceGetAttribute of device ``ordinal``; ``default`` if the driver does not know the attribute."""
    dev, value = c_int(), c_int()
    _device_get(byref(dev), ordinal)
    try:
        _attribute(byref(value), attr, dev.value)
    except CUDAError:
        if default is None:
            raise
        return default
    return value.value


def pci_bus_id(ordinal: int) -> str:
    dev, buf = c_int(), ctypes.create_string_buffer(32)
    _device_get(byref(dev), ordinal)
    _pci_bus_id(buf, len(buf), dev.value)
    return buf.value.decode()


# ---- CUDA VMM memory shared through fabric handles (memory.py) ----
def _fabric_prop(ordinal: int, numa: Optional[int]) -> _AllocationProp:
    """Pinned memory exportable as a fabric handle: on device ``ordinal``, or with ``numa`` on that host NUMA node."""
    prop = _AllocationProp()
    prop.type, prop.requestedHandleTypes = _ALLOC_PINNED, _HANDLE_FABRIC
    if numa is None:
        prop.location.type, prop.location.id = _LOC_DEVICE, ordinal
    else:
        prop.location.type, prop.location.id = _LOC_HOST_NUMA, numa
    return prop


def fabric_granularity(ordinal: int, numa: Optional[int] = None) -> int:
    """The size unit of fabric_create (2 MiB on GB200)."""
    size = c_size_t()
    _granularity(byref(size), byref(_fabric_prop(ordinal, numa)), _GRANULARITY_MIN)
    return size.value


def fabric_create(nbytes: int, ordinal: int, numa: Optional[int] = None) -> int:
    """Physical memory of nbytes, a multiple of fabric_granularity (placement: _fabric_prop) -> allocation handle."""
    handle = c_uint64()
    _create(byref(handle), nbytes, byref(_fabric_prop(ordinal, numa)), 0)
    return handle.value


def fabric_export(handle: int) -> bytes:
    """The 64-byte fabric handle of an allocation, which any process of the NVLink domain can import."""
    raw = _FabricHandle()
    _export(byref(raw), handle, _HANDLE_FABRIC, 0)
    return bytes(raw)


def fabric_import(raw: bytes) -> int:
    """Another process's allocation, from its fabric handle -> an allocation handle of this process."""
    buf = _FabricHandle.from_buffer_copy(raw)
    handle = c_uint64()
    _import(byref(handle), byref(buf), _HANDLE_FABRIC)
    return handle.value


def vmm_map(handle: int, nbytes: int, ordinal: int, numa: Optional[int] = None) -> int:
    """Map an allocation at a fresh address that device ``ordinal`` (with ``numa``, also the CPUs of that host NUMA
    node) can read and write -> the address."""
    where = [(_LOC_DEVICE, ordinal)] + ([] if numa is None else [(_LOC_HOST_NUMA, numa)])
    descs = (_AccessDesc * len(where))()
    for i, (loc_type, loc_id) in enumerate(where):
        descs[i].location.type, descs[i].location.id = loc_type, loc_id
        descs[i].flags = _ACCESS_RW
    ptr = c_uint64()
    _reserve(byref(ptr), nbytes, 0, 0, 0)
    try:
        _map(ptr.value, nbytes, 0, handle, 0)
        try:
            _set_access(ptr.value, nbytes, descs, len(where))
        except BaseException:
            _unmap(ptr.value, nbytes)
            raise
    except BaseException:
        _unreserve(ptr.value, nbytes)
        raise
    return ptr.value


def vmm_unmap(ptr: int, nbytes: int) -> None:
    _unmap(ptr, nbytes)
    _unreserve(ptr, nbytes)


def vmm_release(handle: int) -> None:
    """Drop this process's handle; the memory lives on while a mapping of it or another process's handle does."""
    _release(handle)
