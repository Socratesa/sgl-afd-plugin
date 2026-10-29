"""Helpers shared by the tests: fresh-interpreter runs, the model location, a torch reference of the experts."""

import json
import os
import subprocess
import sys

MODEL_DIR = os.environ.get("ASAP_TEST_MODEL", "/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731")
HAS_MODEL = os.path.isfile(os.path.join(MODEL_DIR, "config.json"))
# 4-layer, 128-expert truncation of DeepSeek-V4-Flash: resolves like the real model but is cheap.
TINY_OVERRIDE = '{"num_hidden_layers": 4, "n_routed_experts": 128, "num_hash_layers": 0}'


def run_python(code, env=None, timeout=900):
    """Run ``code`` in a fresh interpreter (sglang patches are process-global and must not leak)."""
    full_env = dict(os.environ)
    full_env.pop("ASAP_ROLE", None)
    full_env["SGLANG_PLUGINS"] = "sgl_asap_plugin"  # keep the old AFD plugin installed in the venv out
    full_env.update(env or {})
    return subprocess.run([sys.executable, "-c", code], env=full_env, capture_output=True, text=True, timeout=timeout)


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON line in output:\n{stdout}")


# ---- torch reference of DeepSeek-V4's routed experts, straight from the checkpoint ----
_FP4 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]  # e2m1 code -> value


def dequant_mxfp4(packed, scale_e8m0, device):
    """MXFP4 -> bf16 exactly: two e2m1 codes per byte (low nibble first), one 2^(e - 127) scale per 32 inputs."""
    import torch

    b = packed.to(device).view(torch.uint8).to(torch.int64)
    lut = torch.tensor(_FP4, dtype=torch.float32, device=device)
    vals = torch.stack([lut[b & 0xF], lut[b >> 4]], dim=-1).reshape(b.shape[0], -1)
    scale = torch.pow(2.0, scale_e8m0.to(device).view(torch.uint8).to(torch.float32) - 127.0)
    return (vals * scale.repeat_interleave(32, dim=1)).to(torch.bfloat16)


def load_reference_experts(model_dir, pool, layers, device):
    """{(layer, expert): (w1, w2, w3)} in bf16 for the experts in ``pool`` of layers < ``layers``."""
    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    refs = {}
    for layer in range(layers):
        for e in pool:
            w = {}
            for k in ("w1", "w2", "w3"):
                name = f"layers.{layer}.ffn.experts.{e}.{k}"
                with safe_open(os.path.join(model_dir, weight_map[name + ".weight"]), framework="pt") as f:
                    w[k] = dequant_mxfp4(f.get_tensor(name + ".weight"), f.get_tensor(name + ".scale"), device)
            refs[(layer, e)] = (w["w1"], w["w2"], w["w3"])
    return refs


def reference_moe(refs, layer, x, ids, w, limit=10.0, lo=0, hi=1 << 30):
    """sum_k w_k * expert_{ids_k}(x) in fp32 over the reference experts in [lo, hi), as inference/model.py does it:
    swiglu with the clamp, the intermediate rounded to bf16, the routing weight applied after the down projection."""
    import torch

    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for (lay, e), (w1, w2, w3) in refs.items():
        if lay != layer or not lo <= e < hi:
            continue
        rows, slots = torch.where(ids == e)
        if rows.numel():
            gate = (xf[rows] @ w1.float().T).clamp(max=limit)
            up = (xf[rows] @ w3.float().T).clamp(-limit, limit)
            h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
            out.index_add_(0, rows, (h @ w2.float().T) * w[rows, slots].unsqueeze(1))
    return out
