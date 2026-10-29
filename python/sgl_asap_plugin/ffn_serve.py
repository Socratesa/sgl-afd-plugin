"""Entry point of the F (expert) role: ``python -m sgl_asap_plugin.ffn_serve``.

It takes the same flags as ``sglang serve`` (``--model-path``, ``--tp-size E --ep-size E``,
``--quantization`` ...) plus ``--asap-*``, resolves them through sglang's own pipeline,
validates the deployment and computes the static layout (expert partition, arena sizes).

M0 scope: validation, layout and process lifecycle. The expert model, the transport and the
service loop arrive in PR-03/PR-05; the running worker is a placeholder that only proves
the launch contract (readiness line, SIGTERM -> clean exit).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from sgl_asap_plugin import config, layout, validation
from sgl_asap_plugin.errors import AsapConfigError


def build_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(prog="python -m sgl_asap_plugin.ffn_serve")
    ServerArgs.add_cli_args(parser)
    config.add_cli_args(parser)
    parser.add_argument("--asap-dry-run", action="store_true",
                        help="Validate, print the layout and exit without starting the worker.")
    return parser


def plan(ns) -> dict:
    """Resolve + validate the launch and return the static layout summary."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.server_args import ServerArgs

    ns.asap_role = ns.asap_role or "ffn"
    asap = config.AsapConfig.from_namespace(ns)
    if asap.role != "ffn":
        raise AsapConfigError("[ASAP] ffn_serve only runs the ffn role")
    view = resolving_view(ServerArgs.from_cli_args(ns))
    validation.check_ffn(view, asap)
    shape = layout.read_moe_shape(view.model_path, view.json_model_override_args)
    part = layout.partition_experts(shape.n_experts, view.ep_size)
    arena = layout.arena_plan(asap.num_replicas, asap.replica_tp, view.ep_size, asap.slots, asap.n_max,
                              shape.hidden, shape.topk)
    return {
        "role": "ffn",
        "ep_size": view.ep_size,
        "n_experts": shape.n_experts,
        "padded_experts": part.padded,
        "padding_experts": part.padding,
        "experts_per_rank": part.per_rank,
        "expert_ranges": [list(r) for r in part.ranges()],
        "moe": {"hidden": shape.hidden, "topk": shape.topk, "layers": shape.layers},
        "asap": {"rdv_addr": asap.rdv_addr, "num_replicas": asap.num_replicas, "replica_tp": asap.replica_tp,
                 "n_max": asap.n_max, "slots": asap.slots},
        "arena": {"bytes_per_token": arena.bytes_per_token, "region_bytes": arena.region_bytes,
                  "f_recv_arena_bytes": arena.f_recv_arena_bytes,
                  "a_result_arena_bytes": arena.a_result_arena_bytes},
    }


def serve_forever(summary: dict) -> None:
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(f"[ASAP] ffn worker started (M0 placeholder, no transport yet): rdv={summary['asap']['rdv_addr']}",
          flush=True)
    stop.wait()
    print("[ASAP] ffn worker stopped", flush=True)


def main(argv=None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        summary = plan(ns)
    except AsapConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    print("[ASAP] ffn layout " + json.dumps(summary, sort_keys=True), flush=True)
    if not ns.asap_dry_run:
        serve_forever(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
