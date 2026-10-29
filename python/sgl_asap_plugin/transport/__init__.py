"""A<->F transport over NVLink: static CUDA-IPC arenas, one-sided copies, stream-memop flags, backpressure.

Independent of sglang and torch, so it is tested on its own (tests/test_transport.py).
* layout.py   what both sides agree on: memory layout (with diagrams), doorbell word, errors
* channel.py  A side: AfdChannel (connect, send, wait) and the watchdog that turns a dead F into an error
* server.py   F side: FfnServer (handshake, polling, job assembly, reply)
* cuda.py     the CUDA driver calls (ctypes)

Submodules load lazily: importing the package or ``layout`` does not need libcuda.
"""

import importlib

_EXPORTS = {
    "Geometry": "layout",
    "TransportError": "layout",
    "HandshakeError": "layout",
    "ChannelPoisoned": "layout",
    "AfdChannel": "channel",
    "Ticket": "channel",
    "FfnServer": "server",
    "Job": "server",
    "Row": "server",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
