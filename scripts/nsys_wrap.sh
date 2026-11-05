#!/bin/bash
# Wrap any service script with Nsight Systems and capture on demand.
#
# The service and every process it forks (A/F ranks, workers, ...) run inside one
# nsys session whose collection is deferred: model load, warmup and CUDA-graph
# capture are not recorded. `start` opens a capture, `stop` writes a .nsys-rep and
# leaves the service running, and the pair can be repeated — each capture lands in
# its own file.
#
#   bash scripts/nsys_wrap.sh scripts/run_asap.sh          # launch, capture deferred
#   bash scripts/nsys_wrap.sh start                        # begin a capture
#   bash scripts/nsys_wrap.sh stop                         # write logs/nsys/<session>_<ts>_1.nsys-rep
#   bash scripts/nsys_wrap.sh status
#   bash scripts/nsys_wrap.sh shutdown                     # end the session (service exits)
#
# A capture contains:
#   GPU kernels, the full CUDA API plus memcpy/memset, CUDA graphs expanded to nodes,
#   GPU memory usage, NCCL (API, collectives, group and GPU ops)
#   NVTX ranges — sglang scheduler stages and operation spans (needs the `nvtx`
#     package, installed in the venv), per-layer module markers, and the torch
#     autograd NVTX ranges with tensor shapes that nsys injects at torch import
#   cuBLAS, cuDNN, cuSOLVER, cuSPARSE, the OS runtime and Python GIL state
#   CPU backtraces (DWARF) and Python backtraces on CUDA API calls
#   GPU hardware metrics for every device at 10 kHz
#
# NVTX only shows up if the service emits it, so the wrapper enables:
#   SGLANG_ENABLE_NVTX_SCHEDULER, SGLANG_ENABLE_NVTX_OPERATIONS, SGLANG_NSYS_PROFILING
#   A_EXTRA="--enable-layerwise-nvtx-marker" F_EXTRA="--enable-layerwise-nvtx-marker" \
#     bash scripts/nsys_wrap.sh scripts/run_asap.sh
#
# Knobs: NSYS_SESSION, NSYS_OUT (directory), NSYS_GPU_METRICS=0, NSYS_STATS=false,
# NSYS_TRACE (replaces the API list), NSYS_EXTRA (extra flags passed to `nsys profile`).

set -uo pipefail

VENV=${VENV:-/root/paddlejob/inference-public/lihaode/sgl_afd/.venv}
SESSION=${NSYS_SESSION:-sgl_nsys}
OUT=${NSYS_OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/nsys}
GPU_METRICS=${NSYS_GPU_METRICS:-1}
STATS=${NSYS_STATS:-true}
STAMP=${NSYS_STAMP:-$(date +%Y%m%d-%H%M%S)}
TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas,cusolver,cusparse,nccl,python-gil}
GRAPH_TRACE=${NSYS_GRAPH_TRACE:-node}
CUDA_MEM=${NSYS_CUDA_MEM:-true}
PYTORCH=${NSYS_PYTORCH:-autograd-shapes-nvtx}
PY_SAMPLING=${NSYS_PY_SAMPLING:-true}
PY_BACKTRACE=${NSYS_PY_BACKTRACE:-cuda}
BACKTRACE=${NSYS_BACKTRACE:-dwarf}
SAMPLE=${NSYS_SAMPLE:-process-tree}
CPUCTXSW=${NSYS_CPUCTXSW:-process-tree}
NCCL_TRACE=${NSYS_NCCL_TRACE:-all}

# `start` and `stop` make every process of the session enable or disable CUPTI activity
# tracing while its threads keep launching work. Under load that deadlocks on nsys 2026.3.1
# (seen with F's serving thread inside cuGraphLaunch holding a libcuda lock while it waits
# for a CUPTI mutex that the [NSys Comms] thread holds in cuptiActivityEnable_v2 while it
# waits for that libcuda lock). F then never answers, A's 60 s transport watchdog fires and
# A exits. So both commands first pause A's scheduler between batches and let the GPUs
# drain, and resume it once the processes are done with CUPTI.
#   NSYS_ADMIN_URL   A's HTTP server (default http://127.0.0.1:30000); empty = no pause
#   NSYS_QUIESCE_MAX seconds to wait for idle GPUs after the pause (default 30)
ADMIN_URL=${NSYS_ADMIN_URL-http://127.0.0.1:30000}
QUIESCE_MAX=${NSYS_QUIESCE_MAX:-30}
paused=0

quiesce() {
  [ -n "$ADMIN_URL" ] || return 0
  if ! curl -sf -m 10 -X POST "$ADMIN_URL/pause_generation" \
    -H 'Content-Type: application/json' -d '{"mode":"in_place"}' >/dev/null; then
    echo "warning: cannot pause $ADMIN_URL, running $1 on a live pipeline" >&2
    return 0
  fi
  paused=1
  local idle=0 _
  for _ in $(seq 1 $((QUIESCE_MAX * 2))); do
    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | grep -qv '^ *0$'; then
      idle=0
    else
      idle=$((idle + 1))
    fi
    [ "$idle" -ge 3 ] && { echo "A scheduler paused, GPUs idle"; return 0; }
    sleep 0.5
  done
  echo "warning: GPUs still busy ${QUIESCE_MAX}s after the pause" >&2
}

resume() {
  [ "$paused" = 1 ] || return 0
  paused=0
  curl -sf -m 10 -X POST "$ADMIN_URL/continue_generation" \
    -H 'Content-Type: application/json' -d '{}' >/dev/null &&
    echo "A scheduler resumed" ||
    echo "warning: continue_generation failed, POST $ADMIN_URL/continue_generation by hand" >&2
}

export SGLANG_ENABLE_NVTX_SCHEDULER=${SGLANG_ENABLE_NVTX_SCHEDULER:-1}
export SGLANG_ENABLE_NVTX_OPERATIONS=${SGLANG_ENABLE_NVTX_OPERATIONS:-1}
export SGLANG_NSYS_PROFILING=${SGLANG_NSYS_PROFILING:-1}
# Picked up by scripts/run_asap.sh only. The hooks run on every forward whether or
# not nsys is collecting, so a plain run of that script must not set this.
export NVTX_LAYER_FLAGS=${NVTX_LAYER_FLAGS:---enable-layerwise-nvtx-marker}
export PATH="$VENV/bin:$PATH"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found on PATH" >&2
  exit 1
fi

usage() {
  cat >&2 <<EOF
usage:
  $(basename "$0") <service-script> [script args...]   launch under nsys, capture deferred
  $(basename "$0") start                               begin one capture
  $(basename "$0") stop                                end it and write the report; service keeps running
  $(basename "$0") status                              list the live session
  $(basename "$0") shutdown                            end the nsys session
session=$SESSION  reports=$OUT/${SESSION}_<launch-time>_N.nsys-rep
EOF
}

cmd=${1:-}
case "$cmd" in
  start)
    # `nsys start --output` names every capture but the first, which keeps the name
    # given to `nsys profile --output`. Both use the same timestamp from this launch.
    mkdir -p "$OUT"
    if [ -e "$OUT/${SESSION}_${STAMP}_1.nsys-rep" ]; then
      rep="$OUT/${SESSION}_${STAMP}_$(date +%H%M%S).nsys-rep"
    else
      rep="$OUT/${SESSION}_${STAMP}_1.nsys-rep"
    fi
    echo "capturing session '$SESSION' -> $rep"
    trap 'resume; exit 130' INT TERM
    quiesce start
    nsys start --session="$SESSION" --output="$rep" --force-overwrite=true
    rc=$?
    resume
    exit $rc
    ;;
  stop)
    echo "stopping session '$SESSION' (report is written, service keeps running)"
    trap 'resume; exit 130' INT TERM
    quiesce stop
    nsys stop --session="$SESSION" &
    stop_pid=$!
    # The processes are done with CUPTI once the session leaves Collection; what follows
    # is report generation inside nsys, which can take minutes.
    while kill -0 "$stop_pid" 2>/dev/null; do
      state=$(nsys sessions list 2>/dev/null | awk -v s="$SESSION" '$5 == s {print $3}')
      [ "$state" != Collection ] && break
      sleep 0.5
    done
    resume
    wait "$stop_pid"
    exit $?
    ;;
  status)
    exec nsys sessions list
    ;;
  shutdown)
    nsys stop --session="$SESSION" || true
    exec nsys shutdown --session="$SESSION"
    ;;
  ""|-h|--help|help)
    usage
    exit 0
    ;;
esac

script=$1
shift
if [ ! -f "$script" ]; then
  echo "service script not found: $script" >&2
  usage
  exit 1
fi

mkdir -p "$OUT"
# The first capture keeps this name; `nsys start --output` names the rest, and only
# substitutes %n when it appears in the profile command's output.
output="$OUT/${SESSION}_${STAMP}_%n"

profile=(
  nsys profile
  --session-new="$SESSION"
  --start-later=true
  --output="$output"
  --force-overwrite=true
  --trace="$TRACE"
  --cuda-graph-trace="$GRAPH_TRACE"
  --cuda-memory-usage="$CUDA_MEM"
  --nccl-trace="$NCCL_TRACE"
  --pytorch="$PYTORCH"
  --python-sampling="$PY_SAMPLING"
  --python-sampling-frequency=1000
  --python-backtrace="$PY_BACKTRACE"
  --backtrace="$BACKTRACE"
  --sample="$SAMPLE"
  --cpuctxsw="$CPUCTXSW"
  --stats="$STATS"
)
if [ "$GPU_METRICS" = 1 ]; then
  profile+=(--gpu-metrics-devices=all --gpu-metrics-frequency=10000 --gpu-metrics-set=gb10x)
fi
# shellcheck disable=SC2206
[ -n "${NSYS_EXTRA:-}" ] && profile+=(${NSYS_EXTRA})

echo "session : $SESSION"
echo "traces  : $TRACE"
echo "reports : $OUT/${SESSION}_${STAMP}_N.nsys-rep"
echo "service : $script $*"
echo "capture : deferred — run '$(basename "$0") start', then '$(basename "$0") stop'"

exec "${profile[@]}" bash "$script" "$@"
