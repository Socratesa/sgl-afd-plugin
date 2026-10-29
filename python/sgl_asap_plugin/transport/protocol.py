"""Wire protocol: the per-message doorbell and the error types.

A message is announced by ONE 64-bit stream memory write into the F rank's slot file. It is
issued after the payload copies in the same stream, and the memop's barrier makes the payload
visible first. Everything that never changes per message (protocol version, hidden size, top-k,
dtype, capacities) is agreed once in the handshake, so the doorbell only carries what varies:

    bits  0-31  seq        per (A rank, slot) message number; also the ready flag
    bits 32-48  n_tokens   rows in this message, 1 .. MAX_TOKENS
    bits 49-58  layer_id   0 .. MAX_LAYER
    bits 59-63  epoch      connection generation from the handshake, 1 .. MAX_EPOCH

With one word, F's CPU can never see a flag without its metadata. It also keeps host-to-host
copies out of the stream: a host-to-host cuMemcpyAsync blocks the calling thread until every
earlier operation of the stream has finished, which defeats a stream parked in a wait.
"""

from __future__ import annotations

from dataclasses import dataclass

SEQ_MASK = 0xFFFFFFFF
MAX_TOKENS = (1 << 17) - 1
MAX_LAYER = (1 << 10) - 1
MAX_EPOCH = (1 << 5) - 1
PROTO_VERSION = 2


class TransportError(RuntimeError):
    """Base class of every transport failure."""


class HandshakeError(TransportError):
    """The peer rejected the connection or could not be reached."""


class ProtocolError(TransportError):
    """A message violates the wire protocol."""


class ChannelPoisoned(TransportError):
    """A wait timed out: the channel's slots have unknown ownership until it reconnects."""


@dataclass(frozen=True)
class Doorbell:
    seq: int
    n_tokens: int
    layer_id: int
    epoch: int

    def pack(self) -> int:
        if not (1 <= self.n_tokens <= MAX_TOKENS and 0 <= self.layer_id <= MAX_LAYER
                and 1 <= self.epoch <= MAX_EPOCH):
            raise ValueError(f"doorbell field out of range: {self}")
        return (self.seq & SEQ_MASK) | (self.n_tokens << 32) | (self.layer_id << 49) | (self.epoch << 59)

    @classmethod
    def unpack(cls, word: int) -> "Doorbell":
        return cls(seq=word & SEQ_MASK, n_tokens=(word >> 32) & MAX_TOKENS, layer_id=(word >> 49) & MAX_LAYER,
                   epoch=word >> 59)
