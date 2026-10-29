"""A side, part 2: send a micro-batch to every F rank, then wait for their partial sums.

Everything is enqueued on the caller's stream and nothing blocks the host: send() and
wait_combine() only enqueue copies, flag writes and stream waits, so the CPU can run ahead
while the GPU waits for F. Rules for the caller:

* x/ids/w must stay untouched until the send's copies have executed (same-stream order does it).
* the result pointers of a slot must be consumed (stream-ordered) before the next send of
  that slot: F overwrites them once it answers that next message.
* after a timeout the results are garbage; check() raises ChannelPoisoned.
* a parked wait also delays cudaMalloc, first-use kernels and anything else that needs a
  device-wide sync until F answers: warm those paths up before serving.
"""

from __future__ import annotations

from typing import List

from sgl_asap_plugin.transport import cuda as cu
from sgl_asap_plugin.transport import streams as st
from sgl_asap_plugin.transport.arena import ACT_BYTES, TOPK_ID_BYTES
from sgl_asap_plugin.transport.channel_base import ChannelBase
from sgl_asap_plugin.transport.protocol import SEQ_MASK, Doorbell
from sgl_asap_plugin.transport.ticket import Ticket
from sgl_asap_plugin.transport.watchdog import Phase


class AfdChannel(ChannelBase):
    def send(self, layer_id: int, slot: int, x_ptr: int, ids_ptr: int, w_ptr: int, n_tokens: int,
             stream: int) -> Ticket:
        self.check()
        g = self.g
        if not (0 <= slot < g.slots and 1 <= n_tokens <= g.n_max):
            raise ValueError(f"slot {slot} / n_tokens {n_tokens} outside the geometry {g}")
        seq = self._seq[slot] = (self._seq[slot] + 1) & SEQ_MASK
        x_n, i_n = n_tokens * g.hidden * ACT_BYTES, n_tokens * g.topk * TOPK_ID_BYTES
        pre = self._events.acquire()
        st.event_record(pre, stream)
        if self._backpressure:  # the previous message of this slot must be answered before its buffers are reused
            for f in range(g.num_f):
                st.wait32(stream, self._done_addr(f, slot), seq - 1)
        for f in range(g.num_f):
            epoch = self._epoch[f] if self._epoch_override is None else self._epoch_override
            dst = self._recv[f] + g.region_off(self.a, slot)
            cu.memcpy_async(dst, x_ptr, x_n, stream)
            cu.memcpy_async(dst + g.ids_off, ids_ptr, i_n, stream)
            cu.memcpy_async(dst + g.w_off, w_ptr, i_n, stream)
            bell = Doorbell(seq, n_tokens, layer_id, epoch).pack()  # after the payload, in the same stream
            st.write64(stream, self._host_dev[f] + g.host_doorbell_off(self.a, slot), bell)
        ticket = Ticket(self, slot, seq, layer_id, n_tokens)
        post = self._events.acquire()
        st.event_record(post, stream)
        ticket.send_phase = Phase("send", pre, post, [(self._done_addr(f, slot), seq - 1) for f in range(g.num_f)],
                                  f"layer {layer_id} slot {slot} seq {seq}")
        self._watchdog.add(ticket.send_phase)
        return ticket

    def wait_combine(self, ticket: Ticket, stream: int) -> List[int]:
        """Enqueue the waits for every F rank's reply; returns the A-device pointers of their partial sums."""
        self.check()
        g, slot = self.g, ticket.slot
        pre = self._events.acquire()
        st.event_record(pre, stream)
        for f in range(g.num_f):
            st.wait32(stream, self._done_addr(f, slot), ticket.seq)
        post = self._events.acquire()
        st.event_record(post, stream)
        detail = f"layer {ticket.layer_id} slot {slot} seq {ticket.seq}"
        ticket.combine_phase = Phase("combine", pre, post,
                                     [(self._done_addr(f, slot), ticket.seq) for f in range(g.num_f)], detail)
        self._watchdog.add(ticket.combine_phase)
        ticket.result_ptrs = [self._result + g.seg_off(f, slot) for f in range(g.num_f)]
        return ticket.result_ptrs
