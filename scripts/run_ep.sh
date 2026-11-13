#!/bin/bash
# Colocated baseline of run_asap.sh: stock SGLang plus the plugin's base hooks (ERNIE 5.5's MXFP4
# experts and their SiTU), EP ranks that each run attention (attn-tp 1, attention
# DP = EP) and a share of the experts. W4A16 experts (MXFP4 weights, humming, bf16 activations from DeepEP's bf16 dispatch)
# and fp8 KV cache, like run_asap.sh; the metrics flags are the same too, so the two deployments are compared on equal
# terms. No two-batch overlap, and no CUDA graphs on the attention side (neither has them in run_asap.sh).
# After the engine is healthy it starts the smg router (:20010); Ctrl-C, or the engine exiting, stops both.
#   bash scripts/run_ep.sh                                  # EP4: GPUs 0-3, 8k tokens per rank, 32k in all
#   EP=8 bash scripts/run_ep.sh                             # EP8: GPUs 0-7
#   PER_DP_CHUNK=16384 bash scripts/run_ep.sh               # 16k tokens per rank
#   MOE_BACKEND=megamoe bash scripts/run_ep.sh            # megamoe or flashinfer_mxfp4
#   ROUTER=0 bash scripts/run_ep.sh                         # engine only (the router is smg.sh)
# CUDA_VISIBLE_DEVICES, when set, picks other GPUs than the first GPUS_PER_NODE.
#
# Several trays: run_ep_cross.sh (one EP group whose ranks span the trays) and run_ep_pair.sh (one EP per tray behind one
# router). They run this script once per tray and set the knobs below; $LOG_DIR/ready appears when everything is up (written
# by whoever starts the router).
# Ports: http HOST:30000 | nccl 9590 | router 20010. Profiling: nsys_wrap.sh sets LAUNCH_WRAP, see lib.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

EP=${EP:-4}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_mxfp4}  # megamoe or flashinfer_mxfp4
PER_DP_CHUNK=${PER_DP_CHUNK:-8192}  # tokens one DP rank (= one EP rank) prefills per step; --chunked-prefill-size is the total, EP * PER_DP_CHUNK
PREFILL_MAX_RUNNING=${PREFILL_MAX_RUNNING:-512}
# One tray's share of a larger deployment, set by run_ep_cross.sh / run_ep_pair.sh; the defaults are the whole server here.
HOST=${HOST:-127.0.0.1}              # the address the engine binds
GPUS_PER_NODE=${GPUS_PER_NODE:-$EP}  # the EP ranks on this tray; fewer than EP when the group spans trays
NNODES=${NNODES:-1}                  # trays of the group; NODE_RANK is this one (0 serves http), DIST_INIT_ADDR is NODE0:9580
NODE_RANK=${NODE_RANK:-0}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
INSTANCE=${INSTANCE:-EP}             # name of this engine for nsys_wrap.sh (EP, or EP0, EP1, ...); also in the log file name
KILL=${KILL:-1}                      # 0: leave the sglang servers on this tray alone (the caller killed them)
ROUTER=${ROUTER:-1}
[ "$KILL" = 0 ] || kill_servers 127.0.0.1

VENV=${VENV:-/path/to/venv}
MODEL=${MODEL:-/path/to/model}
# The plugin's base hooks only: ERNIE 5.5's MXFP4 experts (expert_dtype) and their
# SiTU on flashinfer_mxfp4. No --asap-role, so the role hooks stay unregistered;
# any other activation (DeepSeek-V4) falls through to SGLang's own method.
PLUGINS=sgl_asap_plugin
DEPLOY=ep$EP
# Prometheus model_name: asap_dsv4_<topology>, e.g. asap_dsv4_ep4 / asap_dsv4_ep8 / asap_dsv4_ep4x2. METRIC_MODEL overrides it.
METRIC_MODEL=${METRIC_MODEL:-asap_dsv4_${DEPLOY}}
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/baseline_ep${EP}_$(date +%Y%m%d-%H%M%S)}
DEVS=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
SFX=${INSTANCE#EP}
ENGINE_LOG=$LOG_DIR/baseline_${MOE_BACKEND}${SFX:+_$SFX}.log
DIST=()
[ "$NNODES" = 1 ] || DIST=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR")
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/ready"

# flashinfer_mxfp4 only registers ("none", "flashinfer_mxfp4"); DeepEP normal
# dispatch has no topk_output, so the DeepEP flags stay off for that backend.
case "$MOE_BACKEND" in
  megamoe)
    # fp8 dispatch: W4A8 量化发生在 DeepEP 里，deep_gemm 拿到 fp8 + group-128 scale
    # MOE_ARGS=(--moe-a2a-backend deepep --deepep-dispatcher-output-dtype fp8 --deepep-mode normal)
    MOE_ARGS=(--moe-runner-backend deep_gemm --moe-a2a-backend megamoe)
    ;;
  flashinfer_mxfp4)
    MOE_ARGS=(--moe-runner-backend flashinfer_mxfp4)
    ;;
  *)
    echo "MOE_BACKEND must be deep_gemm(megamoe) or flashinfer_mxfp4, got: $MOE_BACKEND" >&2
    exit 1
    ;;
esac

# --cuda-graph-backend-prefill 默认是 breakable，暂时关掉和 asap 对齐
launch "$INSTANCE" "$HOST" "$DEVS" "$DEVS" "$ENGINE_LOG" \
  python -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port 30000 --nccl-port 9590 \
  "${DIST[@]}" \
  --tp-size "$EP" --attn-dp-size "$EP" --ep-size "$EP" --load-balance-method total_tokens \
  --context-length 132072 --allow-auto-truncate \
  --max-running-requests "${PREFILL_MAX_RUNNING}" \
  --chunked-prefill-size $((EP * PER_DP_CHUNK)) --max-prefill-tokens "$PER_DP_CHUNK" \
  --kv-cache-dtype fp8_e4m3  --mem-fraction-static 0.85 \
  --fp8-gemm-backend deep_gemm  \
  "${MOE_ARGS[@]}" \
  --enable-prefill-delayer --prefill-delayer-max-delay-passes 32 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --extra-metric-labels "{\"model_name\": \"${METRIC_MODEL}\"}" \
  --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --tokenizer-worker-num 32 \
  --detokenizer-worker-num 16 \
  --tokenizer-backend fastokens \
  --enable-dp-lm-head \
  --trust-remote-code \
  ${EXTRA:-}
PIDS=($!) NAMES=("$INSTANCE") LOGS=("$ENGINE_LOG")

trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true; stop_router; wait' EXIT
trap 'exit 130' INT TERM
echo "$INSTANCE: $HOST GPUs $DEVS (pid $!) -> $ENGINE_LOG, http://$HOST:30000"
# only node 0 of a group serves http; the others just run
healthy() { [ "$NODE_RANK" != 0 ] || curl --noproxy '*' -sf -m 5 "http://$HOST:30000/health" > /dev/null; }
wait_ready
if [ "$NODE_RANK" = 0 ]; then
  echo "ready: EP$EP, moe=$MOE_BACKEND, $PER_DP_CHUNK tokens per rank ($((EP * PER_DP_CHUNK)) in all)"
  if [ "$NNODES" = 1 ]; then  # a group over several trays gets its router from run_ep_cross.sh
    # ready is written by whoever starts the router: with ROUTER=0 that is run_ep_pair.sh, once both engines are healthy
    if [ "$ROUTER" = 1 ]; then start_router "$EP" "http://$HOST:30000" && mark_ready || exit 1
    else echo "router: WORKER_URLS=\"http://$HOST:30000\" EXPECT_RANKS=$EP LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh"; fi
  fi
fi
wait -n "${PIDS[@]}" || true
echo "$INSTANCE exited, see $ENGINE_LOG"
