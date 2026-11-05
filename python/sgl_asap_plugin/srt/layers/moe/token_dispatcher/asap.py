"""Token dispatcher of an attention rank whose routed experts run on the FFN ranks.

``AsapDispatcher.dispatch()`` enqueues the send of one MoE layer's tokens (hidden
states, top-k ids, top-k weights) to every FFN rank; ``combine()`` enqueues the
waits for their replies and returns the sum of their partial outputs. The experts
run on the FFN ranks in between, and the CPU never waits for the FFN ranks.
The model runs its shared experts on the compute stream between the two calls.
Sends use the buffer's send stream; under two-batch overlap
(``srt/batch_overlap/operations_strategy.py``) each micro-batch uses its own slot.

``AsapBuffer`` is the rank's connection to the FFN ranks, as ``DeepEPBuffer`` is to
DeepEP: the transport channel (``transport/``) plus tensor views of the replies.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

import torch
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker

from sgl_asap_plugin.srt.server_args import (
    bootstrap_endpoints,
    get_asap_args,
    transport_geometry,
)

logger = logging.getLogger(__name__)

# The slot of a whole batch; micro-batch i of two-batch overlap uses slot i.
SLOT = 0


class AsapDispatchOutput(NamedTuple):
    ticket: Optional[object]  # the pending send, None when ``hidden_states`` is final
    hidden_states: Optional[torch.Tensor]
    keepalive: tuple = ()  # the tensors the send's copies read, until the answer


class AsapDispatcher(BaseDispatcher):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        micro_batch: Optional[int] = None,
    ) -> AsapDispatchOutput:
        """``micro_batch``: index of a two-batch-overlap micro-batch, None for a whole
        batch."""
        if hidden_states.shape[0] == 0:
            return AsapDispatchOutput(
                None, hidden_states.new_empty(hidden_states.shape)
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError(
                f"ASAP sends explicit top-k ids and weights, got {type(topk_output).__name__}"
            )
        buffer = AsapBuffer.get()
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        if hidden_states.shape[0] > buffer.n_max:
            # Only a mixed batch: chunked prefill keeps a prefill within n_max tokens.
            output = buffer.exchange(self.layer_id, hidden_states, ids, weights)
            return AsapDispatchOutput(None, output)
        ticket, keepalive = buffer.send(
            self.layer_id, hidden_states, ids, weights, micro_batch
        )
        return AsapDispatchOutput(ticket, None, keepalive)

    def combine(self, combine_input: AsapDispatchOutput) -> torch.Tensor:
        if combine_input.hidden_states is not None:
            return combine_input.hidden_states
        return AsapBuffer.get().recv(combine_input.ticket)


class AsapBuffer:
    """The attention rank's channel to the FFN ranks, with tensor views of its replies."""

    _buffer: Optional[AsapBuffer] = None

    def __init__(self, channel, device: int):
        self.channel, self.device = channel, device
        self.n_max = channel.g.n_max
        self._segments = {}  # result address -> [n_max, hidden] view
        self.send_stream = torch.cuda.Stream(device)

    @classmethod
    def get(cls) -> AsapBuffer:
        if cls._buffer is None:
            raise RuntimeError("This attention rank is not connected to the FFN ranks")
        return cls._buffer

    @classmethod
    def connect(cls, model_runner) -> AsapBuffer:
        """Called after the model is loaded and before the KV cache is sized, so that
        the result buffers exist before SGLang hands the free memory to the KV pool."""
        from sglang.srt.runtime_context import get_parallel

        from sgl_asap_plugin.transport.channel import AttnChannel

        layers = sorted(
            (
                m
                for m in model_runner.model.modules()
                if isinstance(getattr(m, "dispatcher", None), AsapDispatcher)
            ),
            key=lambda m: m.layer_id,
        )
        if not layers:
            raise RuntimeError(
                "--asap-ffn-size is set, but the model has no AsapMoE layer"
            )
        args, parallel = get_asap_args(), get_parallel()
        rank = (parallel.dp_rank or 0) * args.asap_attn_tp_size + parallel.tp_rank
        model = {
            "hidden": layers[0].hidden_size,
            "topk": layers[0].top_k,
            "n_experts": layers[0].num_experts,
            "moe_layers": [m.layer_id for m in layers],
        }
        channel = AttnChannel(
            rank,
            transport_geometry(args, model["hidden"], model["topk"]),
            bootstrap_endpoints(args),
            device=model_runner.gpu_id,
            hello={"model": model},
            timeout_s=float(os.environ.get("ASAP_WAITING_TIMEOUT", "60")),
            connect_timeout_s=float(os.environ.get("ASAP_BOOTSTRAP_TIMEOUT", "1800")),
        ).connect()
        cls._buffer = cls(channel, model_runner.gpu_id)
        served = [
            {k: r.get(k) for k in ("f", "experts", "quant_method")}
            for r in channel.replies
        ]
        logger.info(
            f"[ASAP] attention rank {rank} connected to {len(served)} FFN ranks: {served}"
        )
        return cls._buffer

    def send(self, layer_id: int, x, ids, weights, micro_batch=None):
        """Enqueue one message -> (ticket, the tensors its copies read). A whole
        batch sends on slot 0 and micro-batch i on slot i. Both use the send stream,
        after the work that produced their input; the caller
        keeps the tensors until the answer, which comes after the copies."""
        if x.dtype != torch.bfloat16:
            raise TypeError(f"ASAP carries bf16 activations, got {x.dtype}")
        x = x.contiguous()
        ids = ids.to(torch.int32).contiguous()
        weights = weights.to(torch.float32).contiguous()
        stream = torch.cuda.current_stream()
        slot = SLOT if micro_batch is None else micro_batch
        self.send_stream.wait_stream(stream)
        stream = self.send_stream
        ticket = self.channel.send(
            layer_id,
            slot,
            x.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            x.shape[0],
            stream.cuda_stream,
        )
        return ticket, (x, ids, weights)

    def recv(self, ticket) -> torch.Tensor:
        """The sum of the FFN ranks' partial outputs, as a new tensor: the next reply
        overwrites the result segments."""
        ptrs = self.channel.wait(ticket, torch.cuda.current_stream().cuda_stream)
        parts = [self._segment(p)[: ticket.n_tokens] for p in ptrs]
        if len(parts) == 1:
            return parts[0].clone()
        output = parts[0].float()
        for part in parts[1:]:
            output += part
        return output.to(torch.bfloat16)

    def exchange(self, layer_id: int, x, ids, weights) -> torch.Tensor:
        """send + recv in pieces of at most n_max tokens."""
        pieces = []
        for i in range(0, x.shape[0], self.n_max):
            end = i + self.n_max
            ticket, _ = self.send(layer_id, x[i:end], ids[i:end], weights[i:end])
            pieces.append(self.recv(ticket))
        return torch.cat(pieces)

    def _segment(self, ptr: int) -> torch.Tensor:
        view = self._segments.get(ptr)
        if view is None:
            from sgl_asap_plugin.transport.cuda_wrapper import DeviceBuffer

            g = self.channel.g
            raw = DeviceBuffer(ptr, g.n_max * g.hidden * 2)
            view = torch.as_tensor(raw, device=f"cuda:{self.device}")
            view = self._segments[ptr] = view.view(torch.bfloat16).view(
                g.n_max, g.hidden
            )
        return view
