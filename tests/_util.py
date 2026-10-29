"""Shared helpers for the subprocess-based tests."""

import json
import os
import subprocess
import sys

MODEL_DIR = os.environ.get("ASAP_TEST_MODEL", "/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731")
HAS_MODEL = os.path.isfile(os.path.join(MODEL_DIR, "config.json"))
# 4-layer, 128-expert truncation of DeepSeek-V4-Flash: resolves like the real model but is cheap.
TINY_OVERRIDE = '{"num_hidden_layers": 4, "n_routed_experts": 128, "num_hash_layers": 0}'


def run_python(code, env=None, timeout=900):
    """Run ``code`` in a fresh interpreter (global sglang patches must not leak into the test run)."""
    full_env = dict(os.environ)
    full_env.pop("ASAP_ROLE", None)
    full_env["SGLANG_PLUGINS"] = "sgl_asap_plugin"  # keep the old AFD plugin out of the way
    full_env.update(env or {})
    return subprocess.run([sys.executable, "-c", code], env=full_env, capture_output=True, text=True,
                          timeout=timeout)


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON line in output:\n{stdout}")
