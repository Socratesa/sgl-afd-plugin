#!/bin/bash
# Colocated baseline of run_asap.sh: stock SGLang (plugin off), EP ranks that each run attention (attn-tp 1, attention
# DP = EP) and a share of the experts. W4A16 experts (MXFP4 weights, humming, bf16 activations from DeepEP's bf16 dispatch)
# and fp8 KV cache, like run_asap.sh; the metrics flags are the same too, so the two deployments are compared on equal
# terms. No two-batch overlap, and no CUDA graphs on the attention side (neither has them in run_asap.sh).
#   bash scripts/run_ep.sh                                  # EP4: GPUs 0-3, 16k tokens per rank, 64k in all
#   EP=8 bash scripts/run_ep.sh                             # EP8: GPUs 0-7, 128k in all
#   PER_DP_CHUNK=8192 bash scripts/run_ep.sh                # 8k tokens per rank (the same PER_DP_CHUNK as run_asap.sh)
#   MOE_BACKEND=flashinfer_mxfp4 bash scripts/run_ep.sh     # GB: MXFP4 MoE, no DeepEP (a2a defaults to none)
# CUDA_VISIBLE_DEVICES, when set, picks other GPUs than the first EP.

set -euo pipefail

ps -ef | grep sglang | awk '{print $2}' | xargs kill -9 || true

VENV=/root/paddlejob/inference-public/lihaode/sgl_afd/.venv
MODEL=/root/paddlejob/workspace/models/DeepSeek-V4-Flash-0731
EP=${EP:-4}
MOE_BACKEND=${MOE_BACKEND:-deep_gemm}  # deep_gemm (DeepEP) or flashinfer_mxfp4 (no a2a)
PER_DP_CHUNK=${PER_DP_CHUNK:-8192}  # tokens one DP rank (= one EP rank) prefills per step; --chunked-prefill-size is the total, EP * PER_DP_CHUNK
PREFILL_MAX_RUNNING=${PREFILL_MAX_RUNNING:-512}
LOG_DIR=${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/baseline_ep${EP}_$(date +%Y%m%d-%H%M%S)}

export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((EP - 1)))}
export SGLANG_PLUGINS=none SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$LOG_DIR"

# flashinfer_mxfp4 only registers ("none", "flashinfer_mxfp4"); DeepEP normal
# dispatch has no topk_output, so the DeepEP flags stay off for that backend.
case "$MOE_BACKEND" in
  deep_gemm)
    # fp8 dispatch: W4A8 量化发生在 DeepEP 里，deep_gemm 拿到 fp8 + group-128 scale
    MOE_ARGS=(--moe-a2a-backend deepep --deepep-dispatcher-output-dtype fp8 --deepep-mode normal)
    ;;
  flashinfer_mxfp4)
    MOE_ARGS=()
    ;;
  *)
    echo "MOE_BACKEND must be deep_gemm or flashinfer_mxfp4, got: $MOE_BACKEND" >&2
    exit 1
    ;;
esac

# --cuda-graph-backend-prefill 默认是 breakable，暂时关掉和 asap 对齐
python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 --nccl-port 9590 \
  --tp-size "$EP" --attn-dp-size "$EP" --ep-size "$EP" --load-balance-method total_tokens \
  --context-length 132072 --allow-auto-truncate \
  --max-running-requests "${PREFILL_MAX_RUNNING}" \
  --chunked-prefill-size $((EP * PER_DP_CHUNK)) --max-prefill-tokens "$PER_DP_CHUNK" \
  --kv-cache-dtype fp8_e4m3  --mem-fraction-static 0.85 \
  --fp8-gemm-backend deep_gemm  --moe-runner-backend "$MOE_BACKEND" \
  "${MOE_ARGS[@]}" \
  --enable-prefill-delayer --prefill-delayer-max-delay-passes 50 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --extra-metric-labels "{\"model_name\": \"${MODEL}\"}" \
  --enable-metrics --enable-mfu-metrics --enable-metrics-for-all-schedulers --enable-cache-report \
  --enable-request-time-stats-logging --export-metrics-to-file --export-metrics-to-file-dir "$LOG_DIR/metrics_ep$EP" \
  ${EXTRA:-} \
  > "$LOG_DIR/baseline_${MOE_BACKEND}.log" 2>&1 &
PID=$!

trap 'kill -TERM $PID 2>/dev/null || true; wait' EXIT
trap 'exit 130' INT TERM
echo "log: $LOG_DIR/baseline_${MOE_BACKEND}.log (pid $PID), http://127.0.0.1:30000"
until curl -sf -m 5 http://127.0.0.1:30000/health > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "server exited, see $LOG_DIR/baseline_${MOE_BACKEND}.log"; exit 1; }
  sleep 5
done
echo "ready: EP$EP, moe=$MOE_BACKEND, $PER_DP_CHUNK tokens per rank ($((EP * PER_DP_CHUNK)) in all)"
wait $PID || true
echo "server exited, see $LOG_DIR/baseline_${MOE_BACKEND}.log"
