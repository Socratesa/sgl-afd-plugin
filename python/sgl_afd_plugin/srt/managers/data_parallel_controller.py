"""AFD overrides for SGLang's DataParallelController."""

import sys
import logging

import zmq

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process as original_run_data_parallel_controller_process,
)
from sglang.srt.managers.io_struct import sock_recv, sock_send, wrap_as_pickle
from sglang.srt.runtime_context import (
    derive_attention_widths,
    get_parallel,
    get_serving,
)
from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA
from sglang.srt.utils.network import NetworkAddress, get_zmq_socket

from sgl_afd_plugin.afd.config import get_afd
from sgl_afd_plugin.afd.utils.common import override_obj, override_list

logger = logging.getLogger(__name__)

ROLE_HANDSHAKE_PORT_DELTA = DP_ATTENTION_HANDSHAKE_PORT_DELTA + 1


def _gather_node_roles(controller) -> dict[int, str]:
    """Gather node roles to node 0 from all other nodes."""
    if get_parallel().dist_init_addr is None:
        na = NetworkAddress(
            get_serving().host or "127.0.0.1",
            get_serving().port + ROLE_HANDSHAKE_PORT_DELTA,
        )
    else:
        na = NetworkAddress.parse(get_parallel().dist_init_addr)
        na = NetworkAddress(na.host, na.port + ROLE_HANDSHAKE_PORT_DELTA)
    endpoint = na.to_tcp()

    node_role = {get_parallel().node_rank: get_afd().role}
    if get_parallel().node_rank == 0:
        return node_role | _collect_roles_as_server(
            controller, endpoint, get_parallel().nnodes - 1
        )
    _report_role_as_client(controller, endpoint, node_role)
    return node_role


def _collect_roles_as_server(controller, endpoint, expected_clients) -> dict[int, str]:
    rep_socket = get_zmq_socket(controller.context, zmq.REP, endpoint, True)
    roles = {}
    try:
        connected_clients = 0
        while connected_clients < expected_clients:
            roles.update(sock_recv(rep_socket))
            sock_send(rep_socket, wrap_as_pickle("ok"))
            connected_clients += 1
        return roles
    finally:
        rep_socket.close()


def _report_role_as_client(controller, endpoint, node_role) -> None:
    req_socket = get_zmq_socket(controller.context, zmq.REQ, endpoint, False)
    req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000) # 10 miniutes timeout
    req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)
    try:
        sock_send(req_socket, wrap_as_pickle(node_role))
        sock_recv(req_socket)
    except zmq.Again:
        raise RuntimeError("Timeout waiting for role handshake from node 0")
    finally:
        req_socket.close()


def _ffn_slots_from(roles: dict[int, str]) -> frozenset[int]:
    ps = get_parallel()
    _, attn_tp_size = derive_attention_widths(
        tp_size=ps.tp_size,
        attn_cp_size=ps.attn_cp_size,
        dp_size=ps.dp_size,
        enable_dp_attention=True,
    )
    dp_group = attn_tp_size * ps.attn_cp_size
    tp_size_per_node = max(ps.tp_size // max(ps.nnodes // ps.pp_size, 1), 1)
    return frozenset(
        tp_rank // dp_group
        for tp_rank in range(ps.tp_size)
        if roles.get(tp_rank // tp_size_per_node) == "ffn"
    )


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------


def _after_data_parallel_controller_init(result, self, *args, **kwargs):
    """Exclude FFN ranks from request dispatching."""
    ffn_slots = _ffn_slots_from(_gather_node_roles(self))

    self.ffn_slots = ffn_slots
    self.routing_active = [
        i for i in self._active_workers if i not in self.ffn_slots
    ]
    self.dp_budget.ffn_slots = ffn_slots

    logger.info(
        "[AFD][DPC] ffn slots=%s, active dp workers=%s",
        sorted(ffn_slots),
        self.routing_active,
    )


def _around_dispatching_with_trace(original_fn, self, req, refresh_load_budget=True):
    with override_obj(self, "_active_workers", self.routing_active):
        return original_fn(self, req, refresh_load_budget)


def _around_dp_budget_dispatch(original_fn, self, method, estimated_tokens=0):
    with (
        override_list(self.total_requests, self.ffn_slots, sys.maxsize),
        override_list(self.total_tokens, self.ffn_slots, sys.maxsize),
    ):
        return original_fn(self, method, estimated_tokens)


def run_data_parallel_controller_process(*args, **kwargs):
    """
    Run the DataParallelController process.

    The original function doesn't load plugins. This wrapper loads plugins
    first, then calls the original function. Thus hooks can override
    DataParallelController and its dependencies.
    """
    from sglang.srt.plugins import load_plugins

    load_plugins()

    return original_run_data_parallel_controller_process(*args, **kwargs)


def install():
    HookRegistry.register(
        "sglang.srt.managers.data_parallel_controller.DataParallelController"
        ".__init__",
        _after_data_parallel_controller_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.managers.data_parallel_controller.DataParallelController"
        ".dispatching_with_trace",
        _around_dispatching_with_trace,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.data_parallel_controller.DPBudget.dispatch",
        _around_dp_budget_dispatch,
        HookType.AROUND,
    )
