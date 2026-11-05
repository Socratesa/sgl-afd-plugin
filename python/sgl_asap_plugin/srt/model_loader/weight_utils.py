"""AROUND hooks on SGLang's safetensors iterators: skip, unread, the tensors a role
does not load.

SGLang's iterators read every tensor and the model's ``load_weights`` drops the ones
it does not need. An attention rank holds no routed expert, and an FFN rank holds
1/E of them and nothing else, so either would read most of the checkpoint for
nothing (DeepSeek-V4-Flash: 147 of 166.9 GB are routed experts). The role picks a
filter (``skip_routed_experts`` or ``read_routed_experts_only``); the iterators then
select tensors by checkpoint name before any byte is read, and are SGLang's own
otherwise. A routed expert is ``.experts.<E>.`` in a name, its decoder layer
``layers.<L>.`` (``layers.3.ffn.experts.17.w1.weight`` in DeepSeek-V4's format,
``model.layers.3.mlp.experts.17.gate_proj.weight`` in Hugging Face's).
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import os
import re
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch

_ROUTED_EXPERT = re.compile(r"\.experts\.(\d+)\.")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# The tensors this process reads; None: all of them.
_keep: Optional[Callable[[str], bool]] = None


def skip_routed_experts() -> None:
    """Attention ranks: every tensor but the routed experts."""
    global _keep
    _keep = _is_not_a_routed_expert


def read_routed_experts_only(experts: range, num_layers: int) -> None:
    """FFN ranks: the routed experts in ``experts`` of the decoder layers below
    ``num_layers`` (not those of an MTP layer), and nothing else."""
    global _keep

    def keep(name: str) -> bool:
        expert, layer = _ROUTED_EXPERT.search(name), _LAYER.search(name)
        return (
            expert is not None
            and layer is not None
            and int(layer.group(1)) < num_layers
            and int(expert.group(1)) in experts
        )

    _keep = keep


def _is_not_a_routed_expert(name: str) -> bool:
    return _ROUTED_EXPERT.search(name) is None


def buffered_multi_thread_safetensors_weights_iterator(
    original,
    hf_weights_files,
    max_workers,
    disable_mmap=False,
    prefetch=False,
    prefetch_num_threads=4,
    drop_cache_after_load=False,
):
    """AROUND hook. Prefetching is moot when most of every shard is skipped."""
    if _keep is None:
        return original(
            hf_weights_files,
            max_workers,
            disable_mmap=disable_mmap,
            prefetch=prefetch,
            prefetch_num_threads=prefetch_num_threads,
            drop_cache_after_load=drop_cache_after_load,
        )
    return filtered_safetensors_iterator(
        hf_weights_files,
        _keep,
        max_workers=max_workers,
        disable_mmap=disable_mmap,
        drop_cache_after_load=drop_cache_after_load,
    )


def safetensors_weights_iterator(
    original,
    hf_weights_files,
    disable_mmap=False,
    prefetch=False,
    prefetch_num_threads=4,
    drop_cache_after_load=False,
):
    """AROUND hook."""
    if _keep is None:
        return original(
            hf_weights_files,
            disable_mmap=disable_mmap,
            prefetch=prefetch,
            prefetch_num_threads=prefetch_num_threads,
            drop_cache_after_load=drop_cache_after_load,
        )
    return filtered_safetensors_iterator(
        hf_weights_files,
        _keep,
        max_workers=1,
        disable_mmap=disable_mmap,
        drop_cache_after_load=drop_cache_after_load,
    )


def filtered_safetensors_iterator(
    files: Sequence[str],
    keep: Callable[[str], bool],
    *,
    max_workers: int = 4,
    disable_mmap: bool = False,
    drop_cache_after_load: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """``(name, tensor)`` of the kept tensors. A thread pool reads up to
    max_workers + 1 shards ahead, like SGLang's buffered iterator."""
    import safetensors
    import safetensors.torch

    def load(path):
        if disable_mmap:
            with open(path, "rb") as f:
                tensors = safetensors.torch.load(f.read())
            return {k: v for k, v in tensors.items() if keep(k)}
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys() if keep(k)}

    paths = iter(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = collections.deque(
            (p, pool.submit(load, p)) for p in itertools.islice(paths, max_workers + 1)
        )
        while pending:
            path, future = pending.popleft()
            tensors = future.result()
            next_path = next(paths, None)
            if next_path is not None:
                pending.append((next_path, pool.submit(load, next_path)))
            for name in sorted(tensors):
                yield name, tensors[name]
            del tensors
            if drop_cache_after_load:
                _drop_page_cache(path)


def _drop_page_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass
