"""Every sglang symbol the plugin attaches to, why, and when (the README hook table mirrors SPECS).

Three tiers:
* base (every process sglang starts): adds the --asap-* flags and watches ServerArgs.from_cli_args; nothing else.
* role (an ASAP deployment, --asap-role / ASAP_ROLE): resolve and validate the server args, log each rank.
* afd (attention role with --asap-ffn-ranks): the routed experts move to the F service (attn.py, checkpoint.py).

The launcher activates the role tiers from the parsed CLI, before ServerArgs resolution runs, and exports the
config; child processes (DP controller, schedulers) activate them from the inherited ASAP_ROLE in register().

sglang's plugin loader only logs a hook that fails to apply, so a renamed target would silently turn a feature
off. Each spec therefore names the leading parameters it expects: ``preflight`` checks them before
registration and ``ensure_installed`` checks that the patches were really applied. Once the user opted in,
failures are loud: SystemExit in register() (the loader swallows Exception, not SystemExit), exceptions in the
launcher.
"""

from __future__ import annotations

import inspect
import logging
import os
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sgl_asap_plugin.config import ENV_ROLE, AsapConfig, AsapConfigError, AsapInstallError

logger = logging.getLogger("sgl_asap_plugin")

BASE, ROLE, AFD = "base", "role", "afd"
TESTED_SGLANG = ("0.5.22",)


@dataclass(frozen=True)
class HookSpec:
    target: str  # dotted path of the sglang function or method
    type: HookType
    params: Tuple[str, ...]  # expected leading parameter names of the target
    tier: str
    fn: Callable  # signature per HookRegistry.register: BEFORE(*a), AFTER(result, *a), AROUND(orig, *a), REPLACE(*a)
    why: str


# ---- hook bodies: thin, and they import heavy modules lazily (hooks are declared in every process) ----
def _add_cli_args(result, parser):
    from sgl_asap_plugin import config

    config.add_cli_args(parser)


def _activate_from_cli(cls, args):
    """Launcher: the role tiers must be registered before ServerArgs resolution runs. Works for --config file.yaml
    too, because it reads the parsed namespace, not argv."""
    cfg = AsapConfig.from_namespace(args)
    if cfg is None:
        return None
    _activate(cfg)
    HookRegistry.apply_hooks()
    ensure_installed(active_specs())
    cfg.export_env()  # children inherit the config
    _warn_if_untested()
    return None


def _resolve_server_args(result, server_args):
    """Runs right after handle_dwdp and before cuda-graph resolution, the slot DWDP uses for the same purpose:
    the attention role in AFD mode cannot capture CUDA graphs, because the stream waits on F are enqueued with
    fixed sequence numbers (graph replay would reuse them)."""
    cfg = AsapConfig.from_env()
    if cfg is not None and cfg.afd:
        from sglang.srt.arg_groups.overrides import declare_resolution

        declare_resolution(server_args, "asap_afd_attention", disable_cuda_graph=True)


def _check_server_args(server_args):
    """Before sglang's own checks, so a coupled configuration is reported in ASAP terms."""
    from sglang.srt.arg_groups.overrides import resolving_view

    from sgl_asap_plugin import config

    cfg = AsapConfig.from_env()
    if cfg is None:
        return None
    ensure_installed(active_specs())  # a hook that failed to apply must stop the launch
    view = resolving_view(server_args)
    if cfg.role == "attn":
        config.check_attn(view, cfg)
    else:
        config.check_ffn(view, cfg)
    logger.info("[ASAP] server args ok: role=%s afd=%s hooks=%s sglang=%s", cfg.role, cfg.afd, hooks_applied(),
                _sglang_version())
    return None


def _on_model_loaded(result, model_runner):
    from sgl_asap_plugin import attn

    attn.on_model_loaded(model_runner, hooks_applied())


def _remote_experts_class(quant_config):
    from sgl_asap_plugin.attn import RemoteExperts

    return RemoteExperts


def _moe_forward(original, *args, **kwargs):
    from sgl_asap_plugin.attn import moe_forward

    return moe_forward(original, *args, **kwargs)


def _attn_multi_thread_iterator(*args, **kwargs):
    from sgl_asap_plugin.checkpoint import attn_multi_thread_iterator

    return attn_multi_thread_iterator(*args, **kwargs)


def _attn_single_thread_iterator(*args, **kwargs):
    from sgl_asap_plugin.checkpoint import attn_single_thread_iterator

    return attn_single_thread_iterator(*args, **kwargs)


_WEIGHT_UTILS = "sglang.srt.model_loader.weight_utils."
_ITER_PARAMS = ("hf_weights_files", "disable_mmap", "prefetch", "prefetch_num_threads", "drop_cache_after_load")

SPECS: List[HookSpec] = [
    HookSpec("sglang.srt.server_args.ServerArgs.add_cli_args", HookType.AFTER, ("parser",), BASE, _add_cli_args,
             "adds the --asap-* flags"),
    HookSpec("sglang.srt.server_args.ServerArgs.from_cli_args", HookType.BEFORE, ("cls", "args"), BASE,
             _activate_from_cli, "on --asap-role: activates the role tiers before ServerArgs resolution, exports ASAP_*"),
    HookSpec("sglang.srt.arg_groups.parallel_hook.handle_dwdp", HookType.AFTER, ("server_args",), ROLE,
             _resolve_server_args, "AFD attention role: resolves disable_cuda_graph=True before cuda-graph resolution"),
    HookSpec("sglang.srt.arg_groups.validation_hook.check_server_args", HookType.BEFORE, ("server_args",), ROLE,
             _check_server_args, "checks the deployment rules of the role (config.check_attn / check_ffn)"),
    HookSpec("sglang.srt.model_executor.model_runner.ModelRunner.load_model", HookType.AFTER, ("self",), ROLE,
             _on_model_loaded, "logs each rank; in AFD mode connects it to F before the KV pool takes the memory"),
    HookSpec("sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class", HookType.REPLACE, ("quant_config",), AFD,
             _remote_experts_class, "MoE layers get RemoteExperts: no expert tensor is allocated"),
    HookSpec("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward", HookType.AROUND,
             ("self", "hidden_states", "forward_batch", "gemm_output_zero_allocator", "input_ids", "input_ids_global",
              "skip_shared_experts", "return_moe_output"), AFD, _moe_forward,
             "layers holding RemoteExperts send to F instead of running experts"),
    HookSpec(_WEIGHT_UTILS + "buffered_multi_thread_safetensors_weights_iterator", HookType.REPLACE,
             ("hf_weights_files", "max_workers") + _ITER_PARAMS[1:], AFD, _attn_multi_thread_iterator,
             "routed-expert and MTP tensors are skipped unread (~147 of 166.9 GB)"),
    HookSpec(_WEIGHT_UTILS + "safetensors_weights_iterator", HookType.REPLACE, _ITER_PARAMS, AFD,
             _attn_single_thread_iterator, "same, single-thread path"),
]

_registered: Dict[str, HookSpec] = {}


def _sglang_version():
    import importlib.metadata

    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return None


def _warn_if_untested() -> None:
    if _sglang_version() not in TESTED_SGLANG:
        logger.warning("[ASAP] sglang %s is not in the tested set %s; relying on the hook preflight",
                       _sglang_version(), TESTED_SGLANG)


def _problem(spec: HookSpec):
    """Why ``spec`` cannot attach to this sglang build, or None."""
    owner_path, _, attr = spec.target.rpartition(".")
    try:
        raw = inspect.getattr_static(pkgutil.resolve_name(owner_path), attr)
    except Exception as exc:  # ImportError / AttributeError / ValueError from resolve_name
        return f"{spec.target}: cannot resolve ({type(exc).__name__}: {exc})"
    fn = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    try:
        names = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as exc:
        return f"{spec.target}: signature unreadable ({exc})"
    if names[: len(spec.params)] != list(spec.params):
        return f"{spec.target}: expected leading parameters {list(spec.params)}, found {names[: len(spec.params)]}"
    return None


def preflight(specs: List[HookSpec]) -> None:
    problems = [p for p in map(_problem, specs) if p]
    if problems:
        raise AsapInstallError(f"[ASAP] hook targets do not match sglang {_sglang_version()}:\n  - "
                               + "\n  - ".join(problems))


def register_tier(tier: str) -> List[HookSpec]:
    """Preflight, then queue every not yet registered spec of ``tier`` (idempotent). sglang's loader, or
    HookRegistry.apply_hooks(), patches the targets afterwards."""
    todo = [s for s in SPECS if s.tier == tier and s.target not in _registered]
    preflight(todo)
    for spec in todo:
        HookRegistry.register(spec.target, spec.fn, spec.type)
        _registered[spec.target] = spec
    return todo


def active_specs() -> List[HookSpec]:
    return list(_registered.values())


def missing_patches(specs: List[HookSpec]) -> List[str]:
    return sorted({s.target for s in specs} - HookRegistry._patched)


def hooks_applied() -> str:
    """'applied/registered', e.g. '9/9'."""
    specs = active_specs()
    return f"{len(specs) - len(missing_patches(specs))}/{len(specs)}"


def ensure_installed(specs: List[HookSpec]) -> None:
    missing = missing_patches(specs)
    if missing:
        raise AsapInstallError("[ASAP] hooks registered but not applied (look for 'Failed to apply hooks' in the "
                               "log):\n  - " + "\n  - ".join(missing))


def _activate(cfg: AsapConfig) -> None:
    if cfg.role != "attn":
        raise AsapConfigError(f"[ASAP] role {cfg.role!r} does not run under sglang serve; start the expert "
                              "service with: python -m sgl_asap_plugin.ffn")
    register_tier(ROLE)
    if cfg.afd:
        register_tier(AFD)


def register_plugin() -> None:
    """Body of the entry point; sglang's loader applies the registered hooks afterwards."""
    try:
        register_tier(BASE)
    except AsapInstallError as exc:  # not opted in yet: version drift must never break plain sglang
        logger.warning("%s\n[ASAP] the plugin stays disabled in this process.", exc)
        return
    if os.environ.get(ENV_ROLE):
        try:
            _activate(AsapConfig.from_env())
        except (AsapConfigError, AsapInstallError) as exc:
            raise SystemExit(f"[ASAP] {ENV_ROLE}={os.environ[ENV_ROLE]} but the plugin cannot start: {exc}")
        _warn_if_untested()
