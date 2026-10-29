"""Tensors that alias raw device memory owned by the transport arenas."""

from __future__ import annotations

import torch


class _DeviceMemory:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (int(ptr), False),
                                         "version": 2}


def tensor_at(ptr: int, dtype: torch.dtype, shape, device: int) -> torch.Tensor:
    """A tensor over ``prod(shape)`` elements at device address ``ptr`` (no copy, no ownership)."""
    count = 1
    for s in shape:
        count *= int(s)
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    return torch.as_tensor(_DeviceMemory(ptr, nbytes), device=f"cuda:{device}").view(dtype).view(*shape)
