"""Which checkpoint tensors each role reads, and a safetensors iterator that skips the others unread.

A DeepSeek-V4 shard mixes attention and expert tensors, so the choice is made per tensor name before its bytes
are read. sglang's iterators read every tensor and the model's load_weights drops the unknown ones afterwards,
so an A rank would read ~147 GB of the 166.9 GB checkpoint for nothing, and an F rank everything but its share.

Raw (checkpoint) names, before DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format:
    layers.<L>.ffn.experts.<E>.w{1,2,3}.{weight,scale}   routed experts: F rank owning E reads them, A never
    layers.<L>.ffn.shared_experts.*                      shared expert: A
    mtp.<k>.*                                            MTP head: nobody (no speculative decoding)
    everything else (attention, norms, gate, embed, head): A
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import json
import os
import re
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

ROUTED_EXPERT_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.")


def keep_for_attn(name: str) -> bool:
    """An A rank reads everything except routed experts and the MTP head."""
    return not (ROUTED_EXPERT_RE.match(name) or name.startswith("mtp."))


def expert_selector(layers: int, lo: int, hi: int) -> Callable[[str], bool]:
    """Routed experts lo <= e < hi of layers < ``layers``: the share of one F rank."""

    def keep(name: str) -> bool:
        m = ROUTED_EXPERT_RE.match(name)
        return bool(m) and int(m.group(1)) < layers and lo <= int(m.group(2)) < hi

    return keep


def select_files(model_dir: str, keep: Callable[[str], bool]) -> Tuple[List[str], int]:
    """(shards holding at least one kept tensor, number of kept tensors), from the safetensors index."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]
    kept = [name for name in weight_map if keep(name)]
    return [os.path.join(model_dir, s) for s in sorted({weight_map[n] for n in kept})], len(kept)


def filtered_safetensors_iterator(files: Sequence[str], keep: Callable[[str], bool], *, max_workers: int = 4,
                                  disable_mmap: bool = False,
                                  drop_cache_after_load: bool = False) -> Iterator[Tuple[str, "torch.Tensor"]]:
    """(name, tensor) for every kept tensor, shards read by a thread pool with at most max_workers + 1 shards in
    flight (bounded host memory), like sglang's buffered_multi_thread_safetensors_weights_iterator."""
    import safetensors
    import safetensors.torch

    def load(path: str):
        if disable_mmap:
            with open(path, "rb") as f:
                return {k: v for k, v in safetensors.torch.load(f.read()).items() if keep(k)}
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys() if keep(k)}

    paths = iter(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = collections.deque((p, pool.submit(load, p)) for p in itertools.islice(paths, max_workers + 1))
        while pending:
            path, future = pending.popleft()
            tensors = future.result()
            nxt = next(paths, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(load, nxt)))
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


def attn_multi_thread_iterator(hf_weights_files, max_workers, disable_mmap=False, prefetch=False,
                               prefetch_num_threads=4, drop_cache_after_load=False):
    """Replaces weight_utils.buffered_multi_thread_safetensors_weights_iterator on an AFD A rank (same args;
    prefetching is moot because the skipped tensors are most of every shard)."""
    return filtered_safetensors_iterator(hf_weights_files, keep_for_attn, max_workers=max_workers,
                                         disable_mmap=disable_mmap, drop_cache_after_load=drop_cache_after_load)


def attn_single_thread_iterator(hf_weights_files, disable_mmap=False, prefetch=False, prefetch_num_threads=4,
                                drop_cache_after_load=False):
    """Replaces weight_utils.safetensors_weights_iterator on an AFD A rank."""
    return filtered_safetensors_iterator(hf_weights_files, keep_for_attn, max_workers=1,
                                         disable_mmap=disable_mmap, drop_cache_after_load=drop_cache_after_load)
