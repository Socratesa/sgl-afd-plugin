#!/bin/bash
# Nsight Systems for the sglang servers that run_asap.sh / run_ep.sh start. It is not part of how they start: it knows no
# script names and reads no script. The only link is the launchers' generic hook LAUNCH_WRAP (lib.sh), which says "run each
# server as `$LAUNCH_WRAP HOST NAME -- ARGV...` on its tray".
#
# nsys cannot attach to a process that is already running (the CUDA/CUPTI injection happens at process start), so the servers
# are born under nsys with collection deferred, and `start` / `stop` then open and close captures whenever you like:
#
#   bash scripts/nsys_wrap.sh run scripts/run_asap.sh          # any launcher: run_asap_cross.sh, run_ep.sh, run_ep_pair.sh, ...
#   bash scripts/nsys_wrap.sh start [TAG]                      # begin a capture on every session (any time after the servers are up)
#   bash scripts/nsys_wrap.sh stop                             # end it; the reports are written, the services keep running
#   bash scripts/nsys_wrap.sh capture [SECONDS] [TAG]          # start, sleep SECONDS (default 20), stop
#   bash scripts/nsys_wrap.sh status                           # sessions, captures, reports
#   bash scripts/nsys_wrap.sh shutdown                         # end the sessions (the services exit)
#   bash scripts/nsys_wrap.sh exec HOST NAME -- COMMAND...     # the hook itself; wraps any one command by hand
#   bash scripts/nsys_wrap.sh scripts/run_asap.sh              # same as `run`
#
# One nsys session per server instance, on the tray that runs it (A0, A1, ..., F for ASAP; EP, or EP0, EP1, ... for run_ep.sh).
# That is also the only way across trays: a session profiles the process tree it launched on its own tray, and a local nsys
# sees nothing of a process started over ssh. `start` and `stop` reach every session at once (over ssh for the other trays;
# the session list is <NSYS_OUT>/<SESSION>_<launch-time>.sessions on the shared filesystem). Each tray stamps its own clock
# and nsys does not align them: compare trays by NVTX and kernel names, not by absolute time. On one tray the sessions
# share a clock, so the A and F reports line up when opened together.
#
# Reports: <NSYS_OUT>/<SESSION>_<launch-time>_<instance>_<TAG|N|HHMMSS>[_<NSYS_SUFFIX>].nsys-rep. NSYS_SUFFIX is the last
# component, by default the deployment tag the launcher passes (6a2f, 2a2f, ep4; NSYS_SUFFIX= for none), so reports from
# different runs can be downloaded into one directory without overwriting each other. nsys ignores `start --output` for the
# first capture of a session (it is <...>_1); with a TAG, stop renames it.
#
# A capture contains (NSYS_DETAIL=light, the default): GPU kernels, the CUDA API, CUDA graphs expanded to nodes, NCCL, NVTX
# ranges (sglang scheduler stages and operation spans, per-layer module markers) and GPU hardware metrics (10 kHz). It
# leaves out everything that lengthens the CPU launch path (backtraces, sampling, context switches, osrt, torch ranges, memory
# usage), so it also fits "when does A launch what, when does F answer": A's prefill is CPU-launch bound, and the full set
# stretches exactly the gaps being measured.
# NSYS_DETAIL=full adds, on top of that:
#   torch autograd ranges with shapes, GPU memory usage, cuBLAS, cuDNN, cuSOLVER, cuSPARSE, the OS runtime and Python GIL state
#   CPU backtraces (DWARF) and Python backtraces on CUDA API calls, Python sampling at 1 kHz
# Use it only when a CPU-side question needs those; it is expensive even before the first `start`, see below.
#
# Cost before the first `start`. Deferred is not free: the launch-time hooks (the trace list, --cuda-memory-usage, --pytorch,
# --python-*) are in the process from its first instruction, whether or not a capture is open. Measured on this tray (bench-qf,
# 32 sessions, 16 concurrent, prefill only; baseline = the same deployment without the wrapper; one run each, the EP4 baseline
# twice differed by 0.9%):
#                                         EP4 (run_ep.sh)                2A2F (run_asap.sh)
#   NSYS_DETAIL=full                      duration +50%, mean TTFT +44%  +24%, +25%
#   NSYS_DETAIL=light (default)           +9%, +7%                       +4%, +5%
#   light, NSYS_GPU_METRICS=0, no NVTX    +6%, +4%                       not measured
#
# What the wrapper changes in the server it starts (inside `exec`, on the server's tray; the launchers are untouched):
#   env   SGLANG_ENABLE_NVTX_SCHEDULER, SGLANG_ENABLE_NVTX_OPERATIONS, SGLANG_NSYS_PROFILING = 1: the NVTX ranges exist only if
#         the service emits them. These hooks run on every forward whether or not nsys collects.
#   argv  NSYS_NVTX_FLAGS (default --enable-layerwise-nvtx-marker; empty for none): per-layer markers, also on every forward.
#         NSYS_FFN_GRAPH_TOKENS (default 0 = eager) as --asap-ffn-cuda-graph-max-tokens, for the F role only. F's expert CUDA
#         graphs and a CUPTI capture do not mix: some seconds into a capture (or when it starts or stops) a graph launched on
#         the F ranks stops completing, the F GPUs sit at full clock, F never answers, A's 60 s transport watchdog fires
#         (ChannelPoisoned) and A exits. So F runs its experts eagerly under this wrapper (F's kernels are slower than in a
#         normal run). <n> keeps graphs up to n tokens (expect hangs); empty leaves the launcher's value. Both are appended, so
#         they win over the launcher's own flags.
#
# Knobs: NSYS_SESSION, NSYS_OUT (directory), NSYS_STAMP, NSYS_SUFFIX, NSYS_DETAIL=light|full, NSYS_GPU_METRICS=0,
# NSYS_STATS=false (skip the stats nsys runs on every stop; slow for big reports), NSYS_TRACE (replaces the API list),
# NSYS_EXTRA (extra flags passed to `nsys profile`), NSYS_BIN, NSYS_NVTX_FLAGS, NSYS_FFN_GRAPH_TOKENS.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
VENV=${VENV:-/path/to/venv}
SESSION=${NSYS_SESSION:-sgl_nsys}
OUT=${NSYS_OUT:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/nsys}
GPU_METRICS=${NSYS_GPU_METRICS:-1}
STATS=${NSYS_STATS:-true}
DETAIL=${NSYS_DETAIL:-light}
STAMP=${NSYS_STAMP:-$(date +%Y%m%d-%H%M%S)}
NVTX_FLAGS=${NSYS_NVTX_FLAGS---enable-layerwise-nvtx-marker}
FFN_GRAPH_TOKENS=${NSYS_FFN_GRAPH_TOKENS-0}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
if [[ ${NSYS_SUFFIX:-} == *[/\\]* || ${NSYS_SUFFIX:-} == *' '* ]]; then
  echo "NSYS_SUFFIX must be a single path component without spaces, got: $NSYS_SUFFIX" >&2
  exit 1
fi
case "$DETAIL" in
  full) TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas,cusolver,cusparse,nccl,python-gil} ;;
  light) TRACE=${NSYS_TRACE:-cuda,nvtx,nccl} ;;
  *) echo "NSYS_DETAIL must be full or light, got: $DETAIL" >&2; exit 1 ;;
esac
export PATH="$VENV/bin:$PATH"

NSYS_BIN=${NSYS_BIN:-$(command -v nsys || true)}
if [ -z "$NSYS_BIN" ]; then
  echo "nsys not found on PATH on $(hostname)" >&2
  exit 1
fi

usage() {
  cat >&2 <<EOF
usage:
  $(basename "$0") run <launcher-script> [args...]     launch under nsys, capture deferred
  $(basename "$0") start [TAG]                         begin one capture on every session
  $(basename "$0") stop                                end it, write the reports; services keep running
  $(basename "$0") capture [SECONDS] [TAG]             start, sleep SECONDS (default 20), stop
  $(basename "$0") status                              sessions, captures and reports
  $(basename "$0") shutdown                            end the sessions
  $(basename "$0") exec HOST NAME -- COMMAND...        the LAUNCH_WRAP hook: one command under its own session
session=$SESSION  reports=$OUT/${SESSION}_<launch-time>_<instance>_<TAG|N>[_<NSYS_SUFFIX>].nsys-rep
EOF
}

# build_profile SESSION BASE DEVICES: the `nsys profile` command line up to the service, in PROFILE
build_profile() {
  PROFILE=(
    "$NSYS_BIN" profile
    --session-new="$1"
    --start-later=true
    --output="$2_%n"
    --force-overwrite=true
    --trace="$TRACE"
    --cuda-graph-trace=node
    --nccl-trace=all
    --stats="$STATS"
  )
  if [ "$DETAIL" = full ]; then
    PROFILE+=(
      --cuda-memory-usage=true
      --pytorch=autograd-shapes-nvtx
      --python-sampling=true
      --python-sampling-frequency=1000
      --python-backtrace=cuda
      --backtrace=dwarf
      --sample=process-tree
      --cpuctxsw=process-tree
    )
  else
    PROFILE+=(--sample=none --cpuctxsw=none --backtrace=none)
  fi
  if [ "$GPU_METRICS" = 1 ]; then
    PROFILE+=(--gpu-metrics-devices="$3" --gpu-metrics-frequency=10000 --gpu-metrics-set=gb10x)
  fi
  # shellcheck disable=SC2206
  [ -n "${NSYS_EXTRA:-}" ] && PROFILE+=(${NSYS_EXTRA})
  return 0
}

# register HOST NAME SESSION BASE SUFFIX: list a session where start/stop/shutdown find it ("-": no suffix). A name listed
# again by a later launch of the same stamp replaces the earlier line (see for_sessions).
register() {
  mkdir -p "$OUT"
  REG=$OUT/${SESSION}_${STAMP}.sessions
  echo "$1 $2 $3 $4 ${5:--}" >> "$REG"
}

# find_reg: REG = the registry of the newest launch (or of NSYS_STAMP)
find_reg() {
  if [ -n "${NSYS_STAMP:-}" ]; then REG=$OUT/${SESSION}_${NSYS_STAMP}.sessions
  else REG=$(ls -t "$OUT/${SESSION}"_*.sessions 2>/dev/null | head -1); fi
  [ -s "${REG:-}" ] || { echo "no session registry for '$SESSION' under $OUT: launch a service first (NSYS_OUT, NSYS_SESSION, NSYS_STAMP)" >&2; exit 1; }
}

# report_name BASE TAG SUFFIX: the .nsys-rep path. The suffix is last so different deployments do not clobber on download.
report_name() {
  local sfx=
  [ -n "$3" ] && sfx=_$3
  echo "${1}_${2}${sfx}.nsys-rep"
}

# for_sessions FN [ARG...]: FN HOST NAME SESSION BASE SUFFIX ARG... for every session at once; fails if any failed.
# A name that was registered twice (a relaunch with the same stamp) counts once, with its last line.
for_sessions() {
  local fn=$1 pids=() rc=0 p host name session base sfx
  shift
  while read -r host name session base sfx; do
    [ "$sfx" = - ] && sfx=
    [ -n "${NSYS_SUFFIX+x}" ] && sfx=$NSYS_SUFFIX
    "$fn" "$host" "$name" "$session" "$base" "$sfx" "$@" < /dev/null &
    pids+=($!)
  done < <(awk '{l[$2] = $0} END {for (n in l) print l[n]}' "$REG" | sort -k2,2)
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

start_one() {
  local host=$1 name=$2 session=$3 base=$4 sfx=$5 tag=${6:-} rep first=0
  if [ ! -e "$base.captures" ]; then
    first=1 rep=${base}_1.nsys-rep
  else
    rep=$(report_name "$base" "${tag:-$(date +%H%M%S)}" "$sfx")
  fi
  if on "$host" "$(printf %q "$NSYS_BIN") start --session=$session --output=$(printf %q "$rep") --force-overwrite=true"; then
    echo "$(date +%T.%N | cut -c1-12) $name: capture started on $host -> $rep"
    echo "$tag" >> "$base.captures"
    : > "$base.active"
    [ $first = 1 ] && echo "${tag:-1}" > "$base.first"
    return 0
  fi
  echo "$name: nsys start failed on $host (session $session)" >&2
  return 1
}

stop_one() {
  local host=$1 name=$2 session=$3 base=$4 sfx=$5 tag i dest
  [ -e "$base.active" ] || { echo "$name: no capture running"; return 0; }
  # returns once the report is written
  on "$host" "$(printf %q "$NSYS_BIN") stop --session=$session > $(printf %q "$base.stop.log") 2>&1" \
    || { echo "$name: nsys stop failed on $host, see $base.stop.log" >&2; return 1; }
  rm -f "$base.active"
  echo "$(date +%T.%N | cut -c1-12) $name: capture stopped"
  if [ -s "$base.first" ]; then
    tag=$(cat "$base.first")
    dest=$(report_name "$base" "$tag" "$sfx")
    for i in $(seq 30); do [ -e "${base}_1.nsys-rep" ] && break; sleep 1; done
    if [ "$dest" != "${base}_1.nsys-rep" ]; then mv -f "${base}_1.nsys-rep" "$dest" && echo "$name: $dest"
    else echo "$name: $dest"; fi
    rm -f "$base.first"
  fi
}

status_one() {
  local host=$1 name=$2 session=$3 base=$4
  echo "$name on $host: session $session, $([ -e "$base.active" ] && echo capturing || echo idle), $(cat "$base.captures" 2>/dev/null | wc -l) capture(s)"
  ls -la "$base"_*.nsys-rep 2>/dev/null | sed 's/^/    /'
}

shutdown_one() {
  local host=$1 name=$2 session=$3 base=$4 sfx=$5
  [ -e "$base.active" ] && stop_one "$host" "$name" "$session" "$base" "$sfx"
  on "$host" "$(printf %q "$NSYS_BIN") shutdown --session=$session" || true
}

# server_argv ARGV...: SERVER = ARGV plus what profiling needs from an sglang server (see the header); other commands pass as is
server_argv() {
  SERVER=("$@")
  [[ " $* " == *" sglang.launch_server "* ]] || return 0
  local prev= a is_ffn=0
  for a in "$@"; do
    [ "$prev" = --asap-role ] && [ "$a" = ffn ] && is_ffn=1
    prev=$a
  done
  # shellcheck disable=SC2206
  [ -z "$NVTX_FLAGS" ] || SERVER+=($NVTX_FLAGS)
  if [ $is_ffn = 1 ] && [ -n "$FFN_GRAPH_TOKENS" ]; then SERVER+=(--asap-ffn-cuda-graph-max-tokens "$FFN_GRAPH_TOKENS"); fi
  return 0
}

cmd=${1:-}
case "$cmd" in
  start)
    find_reg
    for_sessions start_one "${2:-}"
    exit $?
    ;;
  stop)
    find_reg
    for_sessions stop_one
    exit $?
    ;;
  capture)
    find_reg
    for_sessions start_one "${3:-}" || exit 1
    sleep "${2:-20}"
    for_sessions stop_one
    exit $?
    ;;
  status)
    find_reg
    echo "registry: $REG"
    for_sessions status_one
    while read -r host _; do echo "$host"; done < "$REG" | sort -u | while read -r host; do
      echo "[$host]"
      on "$host" "$(printf %q "$NSYS_BIN") sessions list" < /dev/null
    done
    exit 0
    ;;
  shutdown)
    find_reg
    for_sessions shutdown_one
    exit 0
    ;;
  ""|-h|--help|help)
    usage
    exit 0
    ;;
  exec)
    # the LAUNCH_WRAP hook: run COMMAND under its own session on this tray
    host=${2:?host} name=${3:?name}
    shift 3
    [ "${1:-}" = -- ] && shift
    [ $# -gt 0 ] || { usage; exit 1; }
    session=${SESSION}_$name base=$OUT/${SESSION}_${STAMP}_$name
    register "$host" "$name" "$session" "$base" "${NSYS_SUFFIX-${LAUNCH_DEPLOY:-}}"
    rm -f "$base".captures "$base".active "$base".first
    # a session left by a killed run would make --session-new fail
    "$NSYS_BIN" shutdown --session="$session" > /dev/null 2>&1 || true
    build_profile "$session" "$base" "${LAUNCH_GPUS:-${CUDA_VISIBLE_DEVICES:-all}}"
    export SGLANG_ENABLE_NVTX_SCHEDULER=${SGLANG_ENABLE_NVTX_SCHEDULER:-1}
    export SGLANG_ENABLE_NVTX_OPERATIONS=${SGLANG_ENABLE_NVTX_OPERATIONS:-1}
    export SGLANG_NSYS_PROFILING=${SGLANG_NSYS_PROFILING:-1}
    server_argv "$@"
    exec "${PROFILE[@]}" "${SERVER[@]}"
    ;;
esac

[ "$cmd" = run ] && shift
script=${1:-}
[ $# -eq 0 ] || shift
if [ ! -f "$script" ]; then
  echo "launcher script not found: $script" >&2
  usage
  exit 1
fi

mkdir -p "$OUT"
# The launcher starts every server through $LAUNCH_WRAP HOST NAME -- ARGV... (lib.sh). Servers on other trays get the
# environment the launcher builds there, so what this script reads is passed on explicitly.
fwd=(NSYS_BIN NSYS_OUT NSYS_SESSION NSYS_STAMP NSYS_SUFFIX NSYS_DETAIL NSYS_TRACE NSYS_STATS NSYS_GPU_METRICS NSYS_EXTRA
     NSYS_NVTX_FLAGS NSYS_FFN_GRAPH_TOKENS VENV)
export NSYS_BIN NSYS_OUT=$OUT NSYS_SESSION=$SESSION NSYS_STAMP=$STAMP NSYS_DETAIL=$DETAIL NSYS_STATS=$STATS NSYS_GPU_METRICS=$GPU_METRICS VENV
wrap="env"
for v in "${fwd[@]}"; do
  [ -n "${!v+x}" ] && wrap+=" $v=$(printf %q "${!v}")"
done
export LAUNCH_WRAP="$wrap bash $(printf %q "$SELF") exec"
echo "session : $SESSION (detail $DETAIL), one nsys per server instance"
[ "$DETAIL" = full ] && echo "note    : detail full keeps heavy hooks active before 'start' too (measured +24..50% duration); the default light costs +4..9%"
echo "traces  : $TRACE"
echo "launcher: $script $*"
echo "F graphs: ${FFN_GRAPH_TOKENS:+--asap-ffn-cuda-graph-max-tokens $FFN_GRAPH_TOKENS (F role only)}"
echo "reports : $OUT/${SESSION}_${STAMP}_<instance>_<TAG|N>[_<deployment>].nsys-rep (on the shared disk)"
echo "capture : deferred, run '$(basename "$0") start' then '$(basename "$0") stop' (or '$(basename "$0") capture')"
exec bash "$script" "$@"
