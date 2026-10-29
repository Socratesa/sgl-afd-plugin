"""Checkpoint tensor selection for the two roles, and a name-filtering safetensors iterator.

A DeepSeek-V4 shard mixes attention and expert tensors, so selection has to happen per tensor
name, before the tensor is read: an A rank skips every routed expert (about 147 GB of the
166.9 GB checkpoint) and an F rank reads only its own experts.
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import json
import os
import re
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

# raw checkpoint names: layers.<L>.ffn.experts.<E>.w{1,2,3}.{weight,scale}; mtp.<k>.ffn.experts...
EXPERT_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.")
MTP_PREFIX = "mtp."


def keep_for_attn(name: str) -> bool:
    """An A rank holds everything except routed experts and the MTP head (no speculative decoding)."""
    return not (EXPERT_RE.match(name) or name.startswith(MTP_PREFIX))


def expert_selector(layers: int, lo: int, hi: int) -> Callable[[str], bool]:
    """Names of routed experts ``lo <= e < hi`` in layers ``< layers``: exactly an F rank's share."""

    def keep(name: str) -> bool:
        m = EXPERT_RE.match(name)
        return bool(m) and int(m.group(1)) < layers and lo <= int(m.group(2)) < hi

    return keep


def filtered_safetensors_iterator(hf_weights_files: Sequence[str], keep: Callable[[str], bool], *,
                                  max_workers: int = 4, disable_mmap: bool = False,
                                  drop_cache_after_load: bool = False) -> Iterator[Tuple[str, "torch.Tensor"]]:
    """Like sglang's buffered multi-thread safetensors iterator, but only tensors with keep(name) are read.

    At most max_workers + 1 shards are in flight, so host memory stays bounded.
    """
    import safetensors
    import safetensors.torch

    def load(path: str) -> Dict[str, "torch.Tensor"]:
        if disable_mmap:
            with open(path, "rb") as f:
                return {k: v for k, v in safetensors.torch.load(f.read()).items() if keep(k)}
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys() if keep(k)}

    files = iter(hf_weights_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: collections.deque = collections.deque()
        for path in itertools.islice(files, max_workers + 1):
            pending.append((path, pool.submit(load, path)))
        while pending:
            path, future = pending.popleft()
            tensors = future.result()
            nxt = next(files, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(load, nxt)))
            for name in sorted(tensors):
                yield name, tensors[name]
            del tensors
            if drop_cache_after_load:
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass


def attn_weights_iterator(hf_weights_files, max_workers=4, disable_mmap=False, prefetch=False,
                          prefetch_num_threads=4, drop_cache_after_load=False):
    """Replacement for sglang's two safetensors iterators on an A rank (same keyword names)."""
    return filtered_safetensors_iterator(hf_weights_files, keep_for_attn, max_workers=max_workers,
                                         disable_mmap=disable_mmap, drop_cache_after_load=drop_cache_after_load)


def weight_map(model_dir: str) -> Dict[str, str]:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def select_files(model_dir: str, keep: Callable[[str], bool]) -> Tuple[List[str], int]:
    """Shard paths that contain at least one tensor with keep(name), and the number of such tensors."""
    wanted = {shard for name, shard in weight_map(model_dir).items() if keep(name)}
    count = sum(1 for name in weight_map(model_dir) if keep(name))
    return [os.path.join(model_dir, s) for s in sorted(wanted)], count
