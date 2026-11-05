"""A<->F transport over NVLink: static CUDA-IPC arenas, one-sided copies, stream-memop flags, backpressure.

The communication library of the plugin. Independent of sglang and torch, as DeepEP is: the attention side
wraps it in ``srt/layers/moe/token_dispatcher/asap.py`` (``AsapBuffer``), the FFN side in
``srt/managers/scheduler.py``. Tested on its own (tests/test_gpu.py: TestPingPong, TestBackpressure, TestFaults,
TestPerf).
* layout.py        what both sides agree on: memory layout (with diagrams), doorbell word, errors
* channel.py       A side: AttnChannel (connect, send, wait) and the watchdog that turns a dead F into an error
* server.py        F side: FfnServer (handshake, polling, job assembly, reply)
* cuda_wrapper.py  the CUDA driver calls (ctypes), cf. sglang's device_communicators/cuda_wrapper.py

The plugin uses only AttnChannel.{connect, send, wait, check, close} on A and FfnServer.{start, serve_forever,
reply, close} plus Job / Row on F; another backend (RDMA, PR-18) provides the same methods. Submodules load
lazily: importing the package or ``layout`` does not need libcuda.
"""

import importlib

_EXPORTS = {
    "Geometry": "layout",
    "TransportError": "layout",
    "HandshakeError": "layout",
    "ChannelPoisoned": "layout",
    "AttnChannel": "channel",
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
