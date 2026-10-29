"""A torch reference of DeepSeek-V4's routed experts, straight from the checkpoint files.

MXFP4 weights are dequantized exactly (fp4 e2m1 value * 2^(e8m0 - 127) per 32 inputs) and the expert
runs as in the model's reference implementation (inference/model.py): swiglu with the clamp, the
intermediate rounded to bf16, routing weights applied after the down projection.
"""

import os

import torch

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def dequant(packed_i8: torch.Tensor, scale_e8m0: torch.Tensor, device) -> torch.Tensor:
    b = packed_i8.to(device).view(torch.uint8).to(torch.int64)
    lut = LUT.to(device)
    vals = torch.stack([lut[b & 0xF], lut[b >> 4]], dim=-1).reshape(b.shape[0], -1)  # low nibble first
    scale = torch.pow(2.0, scale_e8m0.to(device).view(torch.uint8).to(torch.float32) - 127.0)
    return (vals * scale.repeat_interleave(32, dim=1)).to(torch.bfloat16)


def load_reference(model_dir: str, pool, layers: int, device):
    """{(layer, expert): (w1, w2, w3)} as bf16 on ``device`` for the experts in ``pool``."""
    from sgl_asap_plugin.model.weights import weight_map
    from safetensors import safe_open

    wm = weight_map(model_dir)
    refs = {}
    for layer in range(layers):
        for e in pool:
            tensors = {}
            for k in ("w1", "w2", "w3"):
                name = f"layers.{layer}.ffn.experts.{e}.{k}"
                with safe_open(os.path.join(model_dir, wm[name + ".weight"]), framework="pt", device="cpu") as f:
                    tensors[k] = dequant(f.get_tensor(name + ".weight"), f.get_tensor(name + ".scale"), device)
            refs[(layer, e)] = (tensors["w1"], tensors["w2"], tensors["w3"])
    return refs


def reference_moe(refs, layer: int, x, ids, w, limit: float = 10.0, lo: int = 0, hi: int = 1 << 30) -> torch.Tensor:
    """sum_k w_k * expert_{ids_k}(x) in fp32, for ids in the reference pool (and in [lo, hi))."""
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for (lay, e), (w1, w2, w3) in refs.items():
        if lay != layer or not lo <= e < hi:
            continue
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        xe = xf[rows]
        gate = (xe @ w1.float().T).clamp(max=limit)
        up = (xe @ w3.float().T).clamp(-limit, limit)
        h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        out.index_add_(0, rows, (h @ w2.float().T) * w[rows, slots].unsqueeze(1))
    return out
