"""A<->F transport: static arenas, single-sided writes, flags with backpressure (NVLink via CUDA IPC).

Independent of sglang and torch. See roadmap 02b for the protocol and 02b section 12 for slot
ownership, epochs and failure handling. Submodules load lazily, so importing this package
(or ``arena`` / ``protocol``) does not need libcuda.
"""

import importlib

_EXPORTS = {
    "Geometry": "arena",
    "AfdChannel": "channel",
    "Ticket": "ticket",
    "FfnServer": "server",
    "Job": "server",
    "Row": "server",
    "Doorbell": "protocol",
    "TransportError": "protocol",
    "HandshakeError": "protocol",
    "ProtocolError": "protocol",
    "ChannelPoisoned": "protocol",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
