"""Deployment rules, checked against sglang's resolved ServerArgs (``resolving_view``).

A (attn) role: D plain data-parallel replicas that never wait for each other, so attention
data parallelism, multi-node and anything else that couples replicas is rejected.
F (ffn) role: one expert service whose ranks are pure expert parallelism.

The functions only read attributes, so tests can feed them ``types.SimpleNamespace``.
All problems are collected and reported together.
"""

from __future__ import annotations

from sgl_asap_plugin.config import AsapConfig
from sgl_asap_plugin.errors import AsapConfigError

_NO_A2A = ("none", None)


class _Reader:
    """Attribute reader that reports a missing sglang field instead of crashing."""

    def __init__(self, view, problems):
        self._view, self._problems = view, problems

    def __call__(self, name):
        try:
            return getattr(self._view, name)
        except AttributeError:
            self._problems.append(f"sglang no longer exposes server arg {name!r} (version drift?)")
            return None


def _finish(role, problems):
    if problems:
        raise AsapConfigError(f"[ASAP] invalid server args for role {role!r}:\n  - " + "\n  - ".join(problems))


def _common(get, problems):
    if get("pp_size") not in (None, 1):
        problems.append(f"pipeline parallelism is not supported (--pp-size 1, got {get('pp_size')})")
    if get("nnodes") not in (None, 1):
        problems.append(f"the MVP transport is CUDA IPC, so A and F must share one node (--nnodes 1, got {get('nnodes')})")
    if get("moe_a2a_backend") not in _NO_A2A:
        problems.append(f"--moe-a2a-backend must be 'none' (experts are served remotely or by the F service "
                        f"alone), got {get('moe_a2a_backend')!r}")
    if get("enable_eplb"):
        problems.append("--enable-eplb is not supported")


def check_attn(view, asap: AsapConfig) -> None:
    problems = []
    get = _Reader(view, problems)
    if get("attn_dp_size") not in (None, 1) or get("enable_dp_attention"):
        problems.append("attention data parallelism (--enable-dp-attention / --attn-dp-size) couples the replicas "
                        "through per-step collectives; use plain replicas instead: --dp-size D")
    _common(get, problems)
    if get("ep_size") not in (None, 1):
        problems.append(f"--ep-size must be 1 on the attention role (experts live on the F service), got {get('ep_size')}")
    if get("expert_distribution_recorder_mode") not in (None,):
        problems.append("--expert-distribution-recorder-mode is not supported (the attention role has no local experts)")
    if get("disaggregation_mode") == "decode":
        problems.append("ASAP serves the prefill phase only; --disaggregation-mode decode is not allowed "
                        "(use null or prefill)")
    dp, tp = get("dp_size"), get("tp_size")
    if asap.num_replicas is not None and dp is not None and asap.num_replicas != dp:
        problems.append(f"--asap-num-replicas {asap.num_replicas} must equal --dp-size {dp}")
    if asap.replica_tp is not None and tp is not None and asap.replica_tp != tp:
        problems.append(f"--asap-replica-tp {asap.replica_tp} must equal --tp-size {tp}")
    _finish("attn", problems)


def check_ffn(view, asap: AsapConfig) -> None:
    problems = []
    get = _Reader(view, problems)
    tp, ep = get("tp_size"), get("ep_size")
    if tp is not None and ep is not None and tp != ep:
        problems.append(f"the F service is pure expert parallelism: set --ep-size equal to --tp-size (= E), "
                        f"got tp={tp} ep={ep}")
    if get("dp_size") not in (None, 1) or get("attn_dp_size") not in (None, 1) or get("enable_dp_attention"):
        problems.append("the F service has no data-parallel dimension (--dp-size 1, no attention DP)")
    _common(get, problems)
    if get("disaggregation_mode") not in (None, "null"):
        problems.append(f"--disaggregation-mode must be null on the F service, got {get('disaggregation_mode')!r}")
    _finish("ffn", problems)


def check_for_role(view, asap: AsapConfig) -> None:
    if asap.role == "attn":
        check_attn(view, asap)
    elif asap.role == "ffn":
        check_ffn(view, asap)
    else:
        raise AsapConfigError(f"[ASAP] unknown role {asap.role!r}")
