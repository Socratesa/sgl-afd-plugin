"""Port choice of the ASAP roles (cf. ``sglang/srt/utils/network.py``)."""

from __future__ import annotations

import logging

from sglang.srt.utils.network import try_bind_socket

logger = logging.getLogger(__name__)

# The fixed ports of an ASAP deployment, all below the kernel's ephemeral range (10000-61000
# on this host), where no outgoing connection can take them:
#   9500 + r   process-group port of A replica r (here)
#   9590       process-group port of F (--nccl-port in scripts/launch_asap.sh)
#   9600 + f   handshake port of F rank f (--asap-bootstrap-addr)
REPLICA_PORT_BASE = 9500


def get_free_port_below_ephemeral(low=22000, high=29999, attempts=64):
    """REPLACE ``sglang.srt.utils.network.get_free_port_below_ephemeral`` (``low`` and ``high`` unused).

    SGLang asks for one process-group port per data-parallel replica of A, and the replica binds it
    only later. It picks at random from [low, high], below the kernel's ephemeral range, but when
    that range starts below ``low`` (10000 on this host) it falls back to bind(0): an ephemeral
    port, which an outgoing connection can take before the replica binds it (EADDRINUSE, about every
    other start of two replicas).

    Replica r gets ``REPLICA_PORT_BASE + r``: SGLang holds the port of a replica before it asks for
    the next replica's (``DataParallelController.launch_dp_schedulers``), so the first free port from
    the base up is the next replica's.
    """
    for port in range(REPLICA_PORT_BASE, REPLICA_PORT_BASE + attempts):
        try:
            try_bind_socket("0.0.0.0", port, reuse_addr=False).close()
        except OSError:
            continue
        logger.info(f"[ASAP] replica process-group port {port}")
        return port
    raise RuntimeError(
        f"no free port in [{REPLICA_PORT_BASE}, {REPLICA_PORT_BASE + attempts}) "
        "for the process group of an A replica"
    )
