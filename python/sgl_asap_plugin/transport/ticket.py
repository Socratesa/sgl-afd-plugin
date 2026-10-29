"""Handle of one in-flight A->F message."""

from __future__ import annotations

import time
from typing import List, Optional

from sgl_asap_plugin.transport.watchdog import Phase


class Ticket:
    """One in-flight message. ``result_ptrs`` (one A-device pointer per F rank) is set by wait_combine."""

    def __init__(self, channel, slot: int, seq: int, layer_id: int, n_tokens: int):
        self.channel, self.slot, self.seq, self.layer_id, self.n_tokens = channel, slot, seq, layer_id, n_tokens
        self.send_phase: Optional[Phase] = None
        self.combine_phase: Optional[Phase] = None
        self.result_ptrs: List[int] = []

    def synchronize(self, timeout_s: float = 60.0) -> None:
        """Block the host until the stream passed this message's waits; raise ChannelPoisoned if one timed out."""
        phase = self.combine_phase or self.send_phase
        deadline = time.monotonic() + timeout_s
        while phase is not None and phase.state == "pending":
            if time.monotonic() > deadline:
                raise TimeoutError(f"layer {self.layer_id} slot {self.slot} seq {self.seq} still pending")
            time.sleep(0.0001)
        self.channel.check()
