"""Streams, events and stream memory operations (the signalling primitives of the transport).

``write32`` / ``wait32`` are CUDA stream memory operations: they cost no SM, do not block
other streams, and compare cyclically (GEQ is wrap-around safe). The value is fixed when the
operation is enqueued, so sequence numbers advance per call, not per CUDA-graph replay.
"""

from __future__ import annotations

from ctypes import POINTER, byref, c_int, c_uint, c_uint32, c_uint64, c_void_p

from sgl_asap_plugin.transport import cuda as cu

WAIT_GEQ = 0
_NOT_READY = 600
_EVENT_DISABLE_TIMING = 2
_STREAM_NON_BLOCKING = 1

_stream_create = cu.bind("cuStreamCreate", POINTER(c_void_p), c_uint)
_stream_destroy = cu.bind("cuStreamDestroy_v2", c_void_p)
_stream_sync = cu.bind("cuStreamSynchronize", c_void_p)
_write32 = cu.bind("cuStreamWriteValue32_v2", c_void_p, c_uint64, c_uint32, c_uint)
_wait32 = cu.bind("cuStreamWaitValue32_v2", c_void_p, c_uint64, c_uint32, c_uint)
_write64 = cu.bind("cuStreamWriteValue64_v2", c_void_p, c_uint64, c_uint64, c_uint)
_event_create = cu.bind("cuEventCreate", POINTER(c_void_p), c_uint)
_event_record = cu.bind("cuEventRecord", c_void_p, c_void_p)
_event_sync = cu.bind("cuEventSynchronize", c_void_p)
_event_destroy = cu.bind("cuEventDestroy_v2", c_void_p)
_event_query = cu._lib.cuEventQuery
_event_query.argtypes = [c_void_p]
_event_query.restype = c_int


def stream_create() -> int:
    handle = c_void_p()
    _stream_create(byref(handle), _STREAM_NON_BLOCKING)
    return handle.value


def stream_destroy(stream: int) -> None:
    _stream_destroy(stream)


def stream_sync(stream: int) -> None:
    _stream_sync(stream)


def write32(stream: int, addr: int, value: int) -> None:
    """Enqueue ``*addr = value`` (full memory barrier: earlier copies in the stream are visible first)."""
    _write32(stream, addr, value & 0xFFFFFFFF, 0)


def write64(stream: int, addr: int, value: int) -> None:
    """Enqueue a 64-bit ``*addr = value`` (one atomic word; same barrier as write32)."""
    _write64(stream, addr, value & 0xFFFFFFFFFFFFFFFF, 0)


def wait32(stream: int, addr: int, value: int) -> None:
    """Enqueue 'block this stream until (int32)(*addr - value) >= 0'."""
    _wait32(stream, addr, value & 0xFFFFFFFF, WAIT_GEQ)


def event_create() -> int:
    handle = c_void_p()
    _event_create(byref(handle), _EVENT_DISABLE_TIMING)
    return handle.value


def event_record(event: int, stream: int) -> None:
    _event_record(event, stream)


def event_done(event: int) -> bool:
    code = _event_query(event)
    if code == 0:
        return True
    if code == _NOT_READY:
        return False
    raise cu.CUDAError("cuEventQuery", code)


def event_sync(event: int) -> None:
    _event_sync(event)


def event_destroy(event: int) -> None:
    _event_destroy(event)
