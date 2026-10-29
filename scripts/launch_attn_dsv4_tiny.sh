#!/usr/bin/env bash
# A-role smoke launch: D plain-DP replicas of a 4-layer DeepSeek-V4-Flash (dummy weights, FP8 experts).
#   GPUS=0,1 D=2 T=1 PORT=31000 bash scripts/launch_attn_dsv4_tiny.sh
set -euo pipefail
MODEL=${MODEL:-/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
export SGLANG_PLUGINS=sgl_asap_plugin            # keep the old AFD plugin out of the way
export SGLANG_DSV4_FP4_EXPERTS=0                 # Hopper: no DeepGEMM fp4, emulate a converted FP8 checkpoint
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 NCCL_DEBUG=WARN
unset NCCL_DEBUG_FILE ASAP_ROLE
exec python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "${PORT:-31000}" \
  --asap-role attn --dp-size "${D:-2}" --tp-size "${T:-1}" \
  --load-format dummy --mem-fraction-static 0.55 --disable-cuda-graph \
  --json-model-override-args '{"num_hidden_layers": 4, "n_routed_experts": 128, "num_hash_layers": 0}'
