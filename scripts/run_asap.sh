#!/bin/bash
# ASAP deployment on one node: start F (experts), start A (attention replicas), wait for both /health, start the smg router.
#   bash scripts/run_asap.sh                                      # 2A2F: A on GPUs 0-1, F on GPUs 2-3, DBO on, chunk 16k
#   A_GPUS="0 1" F_GPUS="4 5 6 7" bash scripts/run_asap.sh        # 2A4F
#   A_GPUS="0 1 2 3" F_GPUS="4 5 6 7" bash scripts/run_asap.sh    # 4A4F
#   DBO=0 bash scripts/run_asap.sh                                # no dual-batch overlap
#   PER_DP_CHUNK=8192 bash scripts/run_asap.sh                    # chunk 8k per A replica
#   A_EXTRA="--asap-tbo-min-tokens 0" bash scripts/run_asap.sh    # split every batch (F_EXTRA for F)
#   DBO=0 F_EXTRA="--moe-runner-backend flashinfer_mxfp4 --asap-ffn-cuda-graph-max-tokens 0" bash scripts/run_asap.sh
#                                                                 # the setup of docs/BENCH.md
#   ROUTER=0 bash scripts/run_asap.sh                             # engines only (the router is smg.sh)
# The defaults follow the fastest setup of docs/BENCH_MOE.md and docs/DBO.md: F runs the experts with CUDA graphs for every
# job size, A splits prefill batches of at least 12288 tokens into two micro-batches (DBO).
#
# Several trays: run_asap_cross.sh. It runs this script once per instance (ROLE=ffn on the F tray, ROLE=attn on each A tray)
# and sets the knobs below; $LOG_DIR/ready appears when everything is up (written by whoever starts the router).
#
# Fixed ports (a second deployment on the same host needs another set):
#   A http 30000 | F /health 30100 | F process group 9590 | F rank f handshake 9600+f | A replica r process group 9500+r
#   | router 20010
# The A_GPUS and F_GPUS lists are one CUDA_VISIBLE_DEVICES: A and F must see the same GPUs (CUDA IPC), A takes the
# first len(A_GPUS) of them, F the rest (--base-gpu-id), so each role's GPUs are consecutive within that list.
# Profiling: nsys_wrap.sh sets LAUNCH_WRAP, see lib.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# One tray's share of a larger deployment, set by run_asap_cross.sh; the defaults are the whole deployment on this tray.
ROLE=${ROLE:-both}                  # both | attn | ffn: which of A and F this run starts
HOST=${HOST:-127.0.0.1}             # the address the servers bind
BOOTSTRAP=${BOOTSTRAP:-$HOST:9600} # where F rank 0 listens; F rank f on port + f
D_TOTAL=${D_TOTAL:-}                # attention replicas in all, over every A instance (default: this run's)
A_BASE=${A_BASE:-0}                 # number of this A instance's first replica among the D_TOTAL
A_TAG=${A_TAG:-}                    # suffix of this A instance (A0, A1, ...; attn0.log) when there are several
TRANSPORT_MEM=${TRANSPORT_MEM:-}    # --asap-transport-mem; empty: the plugin's default (auto)
KILL=${KILL:-1}                     # 0: leave the sglang servers on this tray alone (the caller killed them)
ROUTER=${ROUTER:-1}
case "$ROLE" in both | attn | ffn) ;; *) echo "ROLE must be both, attn or ffn, got: $ROLE" >&2; exit 1 ;; esac
[ "$KILL" = 0 ] || kill_servers 127.0.0.1

VENV=${VENV:-/root/paddlejob/inference-public/lihaode/sgl_afd/.venv}
MODEL=${MODEL:-/root/paddlejob/workspace/models/DeepSeek-V4-Flash-0731}
PLUGINS=sgl_asap_plugin
A_GPUS=(${A_GPUS:-0 1})
F_GPUS=(${F_GPUS:-2 3})
DBO=${DBO:-1}
# Tokens one A replica (one DP rank) prefills per step, not a total over the replicas. It is the replica's
# --chunked-prefill-size = --max-prefill-tokens, and the cap of one dispatch (--asap-max-dispatch-tokens) and of one F job.
PER_DP_CHUNK=${PER_DP_CHUNK:-16384}
PREFILL_MAX_RUNNING=${PREFILL_MAX_RUNNING:-512}
D=${#A_GPUS[@]}
E=${#F_GPUS[@]}
D_TOTAL=${D_TOTAL:-$D}
DEPLOY=${D_TOTAL}a${E}f
# Prometheus model_name: asap_dsv4_<topology>, e.g. asap_dsv4_2a2f / asap_dsv4_6a2f. METRIC_MODEL overrides it.
METRIC_MODEL=${METRIC_MODEL:-asap_dsv4_${DEPLOY}}
A_CSV=$(IFS=,; echo "${A_GPUS[*]}")
F_CSV=$(IFS=,; echo "${F_GPUS[*]}")
TRANSPORT_ARGS=()
[ -z "$TRANSPORT_MEM" ] || TRANSPORT_ARGS=(--asap-transport-mem "$TRANSPORT_MEM")
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/asap_${D}A${E}F_$(date +%Y%m%d-%H%M%S)}
# A and F on this tray share one CUDA_VISIBLE_DEVICES; a run that starts only one of them sees only its own GPUs
A_CVD=$A_CSV,$F_CSV F_CVD=$A_CSV,$F_CSV F_BASE=$D
[ "$ROLE" != attn ] || A_CVD=$A_CSV
[ "$ROLE" != ffn ] || { F_CVD=$F_CSV F_BASE=0; }
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
rm -f "$LOG_DIR/ready"
PIDS=() NAMES=() LOGS=() URLS=()

# F's CUDA graphs: --asap-ffn-cuda-graph-max-tokens (default 0 = eager) is the largest job a graph covers; the buckets are the
# powers of two below it plus the value itself (16384: 1, 2, 4 ... 8192, 16384), every layer in each, and a bigger job runs
# eagerly. A job is one micro-batch of one layer, so it never exceeds PER_DP_CHUNK, and the plugin rejects a value above --asap-max-dispatch-tokens.

if [ "$ROLE" != attn ]; then
  # Hopper --moe-runner-backend humming \
  # 在 humming 0.1.12 上会 misaligned address：indexed kernel block_M >= 16 时 regs_c 缺 alignas(16)，每 rank 25-72 token 就崩, not use humming
  # SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype": "float8e4m3", "input_scale_group_size": 128}' \
  # GB   --moe-runner-backend flashinfer_mxfp4 \
  launch F "$HOST" "$F_CVD" "$F_CSV" "$LOG_DIR/ffn.log" \
    python -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port 30100 --nccl-port 9590 \
    --tp-size "$E" --ep-size "$E" --base-gpu-id "$F_BASE" \
    --asap-max-dispatch-tokens "$PER_DP_CHUNK" --asap-ffn-cuda-graph-max-tokens "$PER_DP_CHUNK" \
    --mem-fraction-static 0.85 \
    --moe-runner-backend flashinfer_mxfp4 \
    --asap-role ffn --asap-attn-dp-size "$D_TOTAL" --asap-attn-tp-size 1 \
    --asap-bootstrap-addr "$BOOTSTRAP" "${TRANSPORT_ARGS[@]}" \
    ${F_EXTRA:-}
  PIDS+=($!) NAMES+=(F) LOGS+=("$LOG_DIR/ffn.log")
  echo "F: $E ranks on $HOST GPUs $F_CSV -> $LOG_DIR/ffn.log (pid $!)"
fi

#  A 的 D 个 DP 副本各要一个进程组端口，--nccl-port 指定不了，副本会抢同一个端口。
#  【TODO FIX】改了插件的 srt/utils/network.py，让副本 r 固定用 9500+r，不再随机挑。
# shared_expert 输入 bf16，Fp8LinearMethod 内部量化为 fp8；
# 【TODO optimize】 w4a8 模式可以尝试在 A 侧量化，传输 fp8 到 F，当前 transport 不支持
if [ "$ROLE" != ffn ]; then
  launch "A$A_TAG" "$HOST" "$A_CVD" "$A_CSV" "$LOG_DIR/attn$A_TAG.log" \
    python -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port 30000 \
    --dp-size "$D" --tp-size 1 --base-gpu-id 0 --load-balance-method total_tokens \
    --context-length 132072 --allow-auto-truncate \
    --max-running-requests "${PREFILL_MAX_RUNNING}" \
    --asap-max-dispatch-tokens "$PER_DP_CHUNK" --chunked-prefill-size "$PER_DP_CHUNK" --max-prefill-tokens "$PER_DP_CHUNK" \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --fp8-gemm-backend deep_gemm \
    "${DBO_FLAGS[@]}" \
    --extra-metric-labels "{\"model_name\": \"${METRIC_MODEL}\"}" \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --asap-role attn --asap-ffn-size "$E" --asap-attn-dp-size "$D_TOTAL" --asap-attn-base-replica "$A_BASE" \
    --asap-bootstrap-addr "$BOOTSTRAP" "${TRANSPORT_ARGS[@]}" \
    --asap-min-batch-tokens 8192 --asap-max-wait-ms 20 \
    --enable-hierarchical-cache --hicache-ratio 1 \
    --tokenizer-worker-num 32 \
    --detokenizer-worker-num 16 \
    --tokenizer-backend fastokens \
    ${A_EXTRA:-}
  PIDS+=($!) NAMES+=("A$A_TAG") LOGS+=("$LOG_DIR/attn$A_TAG.log") URLS+=("http://$HOST:30000")
  echo "A$A_TAG: replicas $A_BASE-$((A_BASE + D - 1)) of $D_TOTAL on $HOST GPUs $A_CSV -> $LOG_DIR/attn$A_TAG.log (pid $!)"
fi

trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true; stop_router; wait' EXIT
trap 'exit 130' INT TERM
healthy() {
  [ "$ROLE" = attn ] || curl --noproxy '*' -sf -m 5 "http://$HOST:30100/health" > /dev/null || return 1
  [ "$ROLE" = ffn ] || curl --noproxy '*' -sf -m 5 "http://$HOST:30000/health" > /dev/null || return 1
}
wait_ready
echo "ready: ${ROLE/both/${D}A${E}F}, $PER_DP_CHUNK tokens per A replica, DBO=$DBO"
if [ "$ROLE" = both ]; then
  # the router knows D DP ranks once it has discovered every replica
  if [ "$ROUTER" = 1 ]; then start_router "$D_TOTAL" "${URLS[@]}" && mark_ready || exit 1
  else echo "router: WORKER_URLS=\"${URLS[*]}\" EXPECT_RANKS=$D_TOTAL LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh"; fi
fi
wait -n "${PIDS[@]}" || true
for k in "${!PIDS[@]}"; do
  kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited, see ${LOGS[k]}"; break; }
done
