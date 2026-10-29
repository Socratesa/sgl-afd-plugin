"""ctypes bindings for the CUDA driver calls the transport needs.

No torch import: any process that owns a CUDA context can use the transport. Calls act on
the calling thread's current context; ``Device.activate()`` makes the device's primary
context current (torch uses the same primary context).
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_ubyte, c_uint, c_uint32, c_uint64, c_void_p

_lib = ctypes.CDLL("libcuda.so.1")

IPC_LAZY_PEER = 1  # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
HOST_PORTABLE_DEVICEMAP = 1 | 2
HOSTALLOC_PORTABLE_DEVICEMAP = 1 | 2


class CUDAError(RuntimeError):
    def __init__(self, name, code):
        text = c_char_p()
        _lib.cuGetErrorName(code, byref(text))
        super().__init__(f"{name} failed: {code} {text.value.decode() if text.value else ''}")
        self.code = code


class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", c_ubyte * 64)]  # c_char would truncate at the first NUL


def bind(name, *argtypes):
    """Return a callable for driver function ``name`` that raises CUDAError on failure."""
    fn = getattr(_lib, name)
    fn.argtypes = list(argtypes)
    fn.restype = c_int

    def call(*args):
        code = fn(*args)
        if code != 0:
            raise CUDAError(name, code)

    return call


_init = bind("cuInit", c_uint)
_device_get = bind("cuDeviceGet", POINTER(c_int), c_int)
_primary_retain = bind("cuDevicePrimaryCtxRetain", POINTER(c_void_p), c_int)
_ctx_set = bind("cuCtxSetCurrent", c_void_p)
_alloc = bind("cuMemAlloc_v2", POINTER(c_uint64), c_size_t)
_free = bind("cuMemFree_v2", c_uint64)
_memset8 = bind("cuMemsetD8_v2", c_uint64, c_ubyte, c_size_t)
_memset32 = bind("cuMemsetD32_v2", c_uint64, c_uint32, c_size_t)
_memcpy_async = bind("cuMemcpyAsync", c_uint64, c_uint64, c_size_t, c_void_p)
_dtoh = bind("cuMemcpyDtoH_v2", c_void_p, c_uint64, c_size_t)
_ipc_get = bind("cuIpcGetMemHandle", POINTER(IpcHandle), c_uint64)
_ipc_open = bind("cuIpcOpenMemHandle_v2", POINTER(c_uint64), IpcHandle, c_uint)
_ipc_close = bind("cuIpcCloseMemHandle", c_uint64)
_host_alloc = bind("cuMemHostAlloc", POINTER(c_void_p), c_size_t, c_uint)
_host_free = bind("cuMemFreeHost", c_void_p)
_host_register = bind("cuMemHostRegister_v2", c_void_p, c_size_t, c_uint)
_host_unregister = bind("cuMemHostUnregister", c_void_p)
_host_devptr = bind("cuMemHostGetDevicePointer_v2", POINTER(c_uint64), c_void_p, c_uint)


class Device:
    """A device's primary context, retained for the lifetime of the object."""

    def __init__(self, ordinal: int):
        _init(0)
        dev, ctx = c_int(), c_void_p()
        _device_get(byref(dev), ordinal)
        _primary_retain(byref(ctx), dev.value)
        self.ordinal, self.ctx = ordinal, ctx.value

    def activate(self) -> None:
        _ctx_set(self.ctx)


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


def copy_to_host(host_ptr: int, dptr: int, nbytes: int) -> None:
    _dtoh(host_ptr, dptr, nbytes)


def ipc_export(dptr: int) -> bytes:
    handle = IpcHandle()
    _ipc_get(byref(handle), dptr)
    return bytes(handle.reserved)


def ipc_open(raw: bytes) -> int:
    handle = IpcHandle()
    ctypes.memmove(byref(handle), raw, 64)
    ptr = c_uint64()
    _ipc_open(byref(ptr), handle, IPC_LAZY_PEER)
    return ptr.value


def ipc_close(dptr: int) -> None:
    _ipc_close(dptr)


def host_alloc(nbytes: int):
    """Pinned, device-mapped host memory -> (host_ptr, device_ptr)."""
    ptr = c_void_p()
    _host_alloc(byref(ptr), nbytes, HOSTALLOC_PORTABLE_DEVICEMAP)
    dev = c_uint64()
    _host_devptr(byref(dev), ptr.value, 0)
    return ptr.value, dev.value


def host_free(host_ptr: int) -> None:
    _host_free(host_ptr)


def host_register(host_ptr: int, nbytes: int) -> int:
    """Register an existing host mapping (e.g. /dev/shm) for device access -> device pointer."""
    _host_register(host_ptr, nbytes, HOST_PORTABLE_DEVICEMAP)
    dev = c_uint64()
    _host_devptr(byref(dev), host_ptr, 0)
    return dev.value


def host_unregister(host_ptr: int) -> None:
    _host_unregister(host_ptr)
