#!/bin/bash
# Start an ASAP deployment on one node: the F service first, then the A replicas (they wait for F to listen).
#   A_GPUS="0 1" F_GPUS="4 5 6 7" bash scripts/launch_asap.sh       # 2 replicas, 4 F ranks
# A and F see the same GPU set (CUDA IPC needs it) and pick their GPUs with --base-gpu-id, so each role's GPUs
# must be consecutive within that set. Logs go to $LOG_DIR; Ctrl-C stops both roles.
set -euo pipefail
MODEL=${MODEL:-/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731}
A_GPUS=(${A_GPUS:-0})
F_GPUS=(${F_GPUS:-4 5 6 7})
N_MAX=${N_MAX:-8192}            # --asap-n-max = --chunked-prefill-size = --max-prefill-tokens
PORT=${PORT:-30000}             # A's HTTP port
RDV_PORT=${RDV_PORT:-9600}      # F rank f listens on RDV_PORT + f
LOG_DIR=${LOG_DIR:-/tmp/asap_logs}
D=${#A_GPUS[@]}
E=${#F_GPUS[@]}
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${A_GPUS[*]},${F_GPUS[*]}")
export SGLANG_PLUGINS=sgl_asap_plugin SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 PYTHONDONTWRITEBYTECODE=1

python -m sgl_asap_plugin.ffn --model-path "$MODEL" --tp-size "$E" --ep-size "$E" --base-gpu-id "$D" \
  --asap-num-replicas "$D" --asap-replica-tp 1 --asap-rdv-addr "127.0.0.1:$RDV_PORT" --asap-n-max "$N_MAX" \
  > "$LOG_DIR/ffn.log" 2>&1 &
F_PID=$!
python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dp-size "$D" --tp-size 1 --base-gpu-id 0 --load-balance-method total_tokens \
  --asap-role attn --asap-ffn-ranks "$E" --asap-rdv-addr "127.0.0.1:$RDV_PORT" --asap-n-max "$N_MAX" \
  --chunked-prefill-size "$N_MAX" --max-prefill-tokens "$N_MAX" ${A_EXTRA:-} \
  > "$LOG_DIR/attn.log" 2>&1 &
A_PID=$!
trap 'kill -TERM $A_PID 2>/dev/null; wait $A_PID 2>/dev/null; kill -TERM $F_PID 2>/dev/null; wait $F_PID' INT TERM EXIT
echo "F: $LOG_DIR/ffn.log (pid $F_PID)  A: $LOG_DIR/attn.log (pid $A_PID), http://127.0.0.1:$PORT"
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $A_PID 2>/dev/null && kill -0 $F_PID 2>/dev/null || { echo "a role exited, see $LOG_DIR"; exit 1; }
  sleep 5
done
echo "ready: $D replica(s), $E F rank(s)"
wait $A_PID
