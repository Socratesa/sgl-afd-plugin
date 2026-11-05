#!/bin/bash
# ASAP deployment on one node: start F (experts), start A (attention replicas), wait for both /health.
#   bash scripts/run_asap.sh                                      # 2A4F: A on GPUs 0-1, F on GPUs 4-7, DBO on, chunk 16k
#   A_GPUS="0 1 2 3" bash scripts/run_asap.sh                     # 4A4F
#   DBO=0 bash scripts/run_asap.sh                                # no dual-batch overlap
#   PER_DP_CHUNK=8192 bash scripts/run_asap.sh                    # chunk 8k per A replica
#   A_EXTRA="--asap-tbo-min-tokens 0" bash scripts/run_asap.sh    # split every batch (F_EXTRA for F)
#   DBO=0 F_EXTRA="--moe-runner-backend flashinfer_mxfp4 --asap-ffn-cuda-graph-max-tokens 0" bash scripts/run_asap.sh
#                                                                    # the setup of docs/BENCH.md
# The defaults follow the fastest setup of docs/BENCH_MOE.md and docs/DBO.md: F runs the humming experts with CUDA
# graphs for every job size, A splits prefill batches of at least 6144 tokens into two micro-batches (DBO). 
#
# Fixed ports (a second deployment on the same host needs another set):
#   A http 30000 | F /health 30100 | F process group 9590 | F rank f handshake 9600+f | A replica r process group 9500+r
# The A_GPUS and F_GPUS lists are one CUDA_VISIBLE_DEVICES: A and F must see the same GPUs (CUDA IPC), A takes the
# first len(A_GPUS) of them, F the rest (--base-gpu-id), so each role's GPUs are consecutive within that list.

set -euo pipefail

ps -ef | grep sglang | awk '{print $2}' | xargs kill -9 || true

VENV=/root/paddlejob/inference-public/lihaode/sgl_afd/.venv
MODEL=/root/paddlejob/workspace/models/DeepSeek-V4-Flash-0731
A_GPUS=(${A_GPUS:-0 1})
F_GPUS=(${F_GPUS:-2 3})
DBO=${DBO:-1}
# Tokens one A replica (one DP rank) prefills per step, not a total over the replicas. It is the replica's
# --chunked-prefill-size = --max-prefill-tokens, and the cap of one dispatch (--asap-max-dispatch-tokens) and of one F job.
PER_DP_CHUNK=${PER_DP_CHUNK:-16384}
PREFILL_MAX_RUNNING=${PREFILL_MAX_RUNNING:-512}
D=${#A_GPUS[@]}
E=${#F_GPUS[@]}
LOG_DIR=${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/asap_${D}A${E}F_$(date +%Y%m%d-%H%M%S)}

export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${A_GPUS[*]},${F_GPUS[*]}")
export SGLANG_PLUGINS=sgl_asap_plugin SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 PYTHONDONTWRITEBYTECODE=1
DBO_FLAGS=()
if [ "$DBO" = 1 ]; then
  DBO_FLAGS=(
    --enable-two-batch-overlap
    # micro-batches in flight per A replica (receive slots per replica on every F rank); DBO needs at least 2
    --asap-max-micro-batches 2
    # 【TODO Test】split only EXTEND batches of at least this many tokens: smaller ones are CPU-launch bound, a split doubles that
    --asap-tbo-min-tokens 12288
    # SGLang's: if splitting between sequences leaves the smaller side under this share, a sequence is cut in the middle (halves 48-52%)
    --tbo-token-distribution-threshold 0.48
  )
fi
mkdir -p "$LOG_DIR"

# F's CUDA graphs: --asap-ffn-cuda-graph-max-tokens (default 0 = eager) is the largest job a graph covers; the buckets are the
# powers of two below it plus the value itself (16384: 1, 2, 4 ... 8192, 16384), every layer in each, and a bigger job runs
# eagerly. A job is one micro-batch of one layer, so it never exceeds PER_DP_CHUNK, and the plugin rejects a value above --asap-max-dispatch-tokens.

# Hopper --moe-runner-backend humming \
# 在 humming 0.1.12 上会 misaligned address：indexed kernel block_M >= 16 时 regs_c 缺 alignas(16)，每 rank 25-72 token 就崩, not use humming
# SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype": "float8e4m3", "input_scale_group_size": 128}' \  
# GB   --moe-runner-backend flashinfer_mxfp4 \
python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30100 --nccl-port 9590 \
  --tp-size "$E" --ep-size "$E" --base-gpu-id "$D" \
  --asap-max-dispatch-tokens "$PER_DP_CHUNK" --asap-ffn-cuda-graph-max-tokens "$PER_DP_CHUNK" \
  --mem-fraction-static 0.85 \
  --moe-runner-backend flashinfer_mxfp4 \
  --asap-role ffn --asap-attn-dp-size "$D" --asap-attn-tp-size 1 --asap-bootstrap-addr 127.0.0.1:9600 \
  ${NVTX_LAYER_FLAGS:-} \
  ${F_EXTRA:-} \
  > "$LOG_DIR/ffn.log" 2>&1 &
F_PID=$!

#  A 的 D 个 DP 副本各要一个进程组端口，--nccl-port 指定不了，副本会抢同一个端口。
#  【TODO FIX】改了插件的 srt/utils/network.py，让副本 r 固定用 9500+r，不再随机挑。
# shared_expert 输入 bf16，Fp8LinearMethod 内部量化为 fp8；
# 【TODO optimize】 w4a8 模式可以尝试在 A 侧量化，传输 fp8 到 F，当前 transport 不支持
python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 \
  --dp-size "$D" --tp-size 1 --base-gpu-id 0 --load-balance-method total_tokens \
  --context-length 132072 --allow-auto-truncate \
  --max-running-requests "${PREFILL_MAX_RUNNING}" \
  --asap-max-dispatch-tokens "$PER_DP_CHUNK" --chunked-prefill-size "$PER_DP_CHUNK" --max-prefill-tokens "$PER_DP_CHUNK" \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --fp8-gemm-backend deep_gemm \
  "${DBO_FLAGS[@]}" \
  --extra-metric-labels "{\"model_name\": \"lhd_dsv4\"}" \
  --enable-metrics --enable-mfu-metrics --enable-metrics-for-all-schedulers --enable-cache-report \
  --enable-request-time-stats-logging --export-metrics-to-file --export-metrics-to-file-dir "$LOG_DIR/metrics_${D}a${E}f" \
  --asap-role attn --asap-ffn-size "$E" --asap-bootstrap-addr 127.0.0.1:9600 \
  --asap-min-batch-tokens 12288 --asap-max-wait-ms 100 \
  ${NVTX_LAYER_FLAGS:-} \
  ${A_EXTRA:-} \
  > "$LOG_DIR/attn.log" 2>&1 &
A_PID=$!

trap 'kill -TERM $A_PID $F_PID 2>/dev/null || true; wait' EXIT
trap 'exit 130' INT TERM
echo "F: $LOG_DIR/ffn.log (pid $F_PID)"
echo "A: $LOG_DIR/attn.log (pid $A_PID), http://127.0.0.1:30000"
until curl -sf -m 5 http://127.0.0.1:30000/health > /dev/null && curl -sf -m 5 http://127.0.0.1:30100/health > /dev/null; do
  kill -0 $F_PID 2>/dev/null || { echo "F exited, see $LOG_DIR/ffn.log"; exit 1; }
  kill -0 $A_PID 2>/dev/null || { echo "A exited, see $LOG_DIR/attn.log"; exit 1; }
  sleep 5
done
echo "ready: ${D}A${E}F, $PER_DP_CHUNK tokens per A replica, DBO=$DBO"
wait -n $A_PID $F_PID || true
kill -0 $F_PID 2>/dev/null && echo "A exited, see $LOG_DIR/attn.log" || echo "F exited, see $LOG_DIR/ffn.log"
