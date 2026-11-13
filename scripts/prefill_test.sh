#!/bin/bash
# Prefill comparison. A group is one deployment; every group goes through the same steps:
#   1. cluster  the group's launcher (define() below: run_asap.sh, run_asap_cross.sh, run_ep.sh, run_ep_pair.sh or
#               run_ep_cross.sh) in the background: engines, then the smg router on :20010 of this tray
#   2. ready    wait for $LOG_DIR/ready: engines healthy and the router knows every DP rank (no fixed sleeps)
#   3. req      req.sh, one short request through the router
#   4. bench    bench-qf-new-t6t7.sh against the router (BENCH; its knobs, e.g. NUM_PROMPTS, pass through)
#   5. stop     the cluster is stopped before the next group, whatever happened
# A group that fails records the step and the next one runs; the exit status is 1 if any failed.
#   bash scripts/prefill_test.sh                          # DEFAULT_GROUPS, in this order
#   bash scripts/prefill_test.sh asap_2a2f baseline_ep4   # these groups, in this order
#   bash scripts/prefill_test.sh list                     # every group
#   BENCH=/path/to/bench-mini.sh bash scripts/prefill_test.sh   # smoke test: 1 session
# Per group, everything lands in $LOG_ROOT/prefill_<group>_<time>/: launch.log, ffn.log/attn.log or baseline_*.log,
# pd_router.log, req.log, bench.log (the cross/pair launchers add launch_*.log). '*' in a group name is x in file names.
# The bench results go to bench-qf/results-t6t7/<time>_<group>-qf (see bench.log).
# READY_TIMEOUT: seconds to wait for ready (default 1800). Envs for one group are in define() below.
# To profile a group, start the cluster by hand under nsys_wrap.sh instead (run, then start/stop around the bench).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
BENCH=${BENCH:-/path/to/bench.sh}
LOG_ROOT=${LOG_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
# DEFAULT_GROUPS=(asap_6a2f asap_4a2f 'baseline_ep4*2' baseline_ep4 asap_2a2f baseline_ep8)
DEFAULT_GROUPS=(asap_6a2f asap_4a2f 'baseline_ep4*2')
# DEFAULT_GROUPS=(asap_6a2f asap_4a2f asap_2a2f baseline_ep8)
# DEFAULT_GROUPS=(baseline_ep4_megamoe 'baseline_ep4*2_megamoe'  baseline_ep8_megamoe)
ALL_GROUPS=(baseline_ep4 'baseline_ep4*2' baseline_ep8 baseline_ep4_megamoe 'baseline_ep4*2_megamoe'  baseline_ep8_megamoe asap_6a2f asap_4a2f asap_2a2f)
mkdir -p "$LOG_ROOT"
DRIVER_LOG=$LOG_ROOT/prefill_test_$(date +%Y%m%d-%H%M%S).log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

# the two trays of the hostfile: TRAY0 is this one (router, bench and the A of the cross deployments), TRAY1 the other
read -r TRAY0 TRAY1 <<< "$(hostfile_trays)"

# define GROUP: LAUNCHER (a script of this directory) and ENVV (the environment of its launcher). A tray has 4 GPUs, so what
# needs more than 4 goes through a cross/pair launcher; the others run on this tray only.
define() {
  case "$1" in
    baseline_ep4) LAUNCHER=run_ep.sh ENVV=(EP=4) ;;
    # two independent EP4, one per tray, one router
    'baseline_ep4*2') LAUNCHER=run_ep_pair.sh ENVV=(EP=4 NODES="$TRAY0 $TRAY1") ;;
    # one EP8 whose ranks span both trays
    baseline_ep8) LAUNCHER=run_ep_cross.sh ENVV=(NODES="$TRAY0 $TRAY1" GPUS_PER_NODE=4) ;;

    baseline_ep4_megamoe) LAUNCHER=run_ep.sh ENVV=(EP=4 MOE_BACKEND=megamoe) ;;
    # two independent EP4, one per tray, one router
    'baseline_ep4*2_megamoe') LAUNCHER=run_ep_pair.sh ENVV=(EP=4 NODES="$TRAY0 $TRAY1" MOE_BACKEND=megamoe) ;;
    # one EP8 whose ranks span both trays
    baseline_ep8_megamoe) LAUNCHER=run_ep_cross.sh ENVV=(NODES="$TRAY0 $TRAY1" GPUS_PER_NODE=4 MOE_BACKEND=megamoe) ;;


    # A0 = 4 replicas on GPUs 0-3 of this tray, A1 = 2 on GPUs 2-3 of the other, F on GPUs 0-1 of the other
    asap_6a2f) LAUNCHER=run_asap_cross.sh ENVV=(F_HOST="$TRAY1" F_GPUS="0 1" A_HOSTS="$TRAY0:0,1,2,3 $TRAY1:2,3") ;;
    # A = 4 replicas on GPUs 0-3 of this tray, F on GPUs 0-1 of the other
    asap_4a2f) LAUNCHER=run_asap_cross.sh ENVV=(F_HOST="$TRAY1" F_GPUS="0 1" A_HOSTS="$TRAY0:0,1,2,3") ;;
    # A on GPUs 0-1, F on GPUs 2-3, all on this tray
    asap_2a2f) LAUNCHER=run_asap.sh ENVV=(A_GPUS="0 1" F_GPUS="2 3") ;;
    *) return 1 ;;
  esac
}

LAUNCHER_PID=
stop_cluster() {
  if [ -n "$LAUNCHER_PID" ] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    kill -TERM "$LAUNCHER_PID" 2>/dev/null   # its EXIT trap stops the engines and the router on every tray
    for _ in $(seq 60); do kill -0 "$LAUNCHER_PID" 2>/dev/null || break; sleep 2; done
    kill -0 "$LAUNCHER_PID" 2>/dev/null && kill -9 "$LAUNCHER_PID" 2>/dev/null
    wait "$LAUNCHER_PID" 2>/dev/null
  fi
  LAUNCHER_PID=
  # a cross/pair group has engines on both trays, and an F can outlive the SIGTERM of its launcher
  kill_servers "$TRAY0" "$TRAY1"
}
trap 'log "interrupted"; stop_cluster; exit 130' INT TERM

# wait_ready DIR: 0 when DIR/ready exists, 1 when the launcher died first, 2 on timeout
wait_ready_file() {
  local t=0
  until [ -e "$1/ready" ]; do
    kill -0 "$LAUNCHER_PID" 2>/dev/null || return 1
    [ "$t" -lt "$READY_TIMEOUT" ] || return 2
    sleep 5
    t=$((t + 5))
  done
}

# step LABEL CMD...: run CMD with its output in $GDIR/LABEL.log; log how it went
step() {
  local label=$1 t0=$SECONDS
  shift
  log "$GROUP: $label ..."
  if "$@" >> "$GDIR/$label.log" 2>&1; then
    log "$GROUP: $label ok ($((SECONDS - t0)) s)"
  else
    log "$GROUP: FAIL $label ($((SECONDS - t0)) s), see $GDIR/$label.log"
    return 1
  fi
}

# bench-qf-new-t6t7.sh writes ./para.yaml, so it runs from its own directory; the group is part of its result directory
bench_cmd() { (cd "$(dirname "$BENCH")" && TIMESTAMP="$(date +%Y%m%d-%H%M%S)_$GTAG" bash "$BENCH"); }

# run_group GROUP: the steps above; FAILED_STEP is empty when all passed. GTAG is the group name in file names (no '*').
run_group() {
  GROUP=$1 FAILED_STEP= GTAG=${1//\*/x}
  GDIR=$LOG_ROOT/prefill_${GTAG}_$(date +%Y%m%d-%H%M%S)
  define "$GROUP"
  mkdir -p "$GDIR"
  log "===== $GROUP start: $LAUNCHER ${ENVV[*]} (logs $GDIR) ====="
  env LOG_DIR="$GDIR" "${ENVV[@]}" bash "$SCRIPT_DIR/$LAUNCHER" > "$GDIR/launch.log" 2>&1 &
  LAUNCHER_PID=$!
  log "$GROUP: cluster launcher pid=$LAUNCHER_PID"
  wait_ready_file "$GDIR"
  case $? in
    0) log "$GROUP: ready" ;;
    1) FAILED_STEP=cluster; log "$GROUP: FAIL cluster exited before it was ready, see $GDIR/launch.log" ;;
    *) FAILED_STEP=ready; log "$GROUP: FAIL not ready after ${READY_TIMEOUT}s, see $GDIR/launch.log" ;;
  esac
  [ -n "$FAILED_STEP" ] || step req bash "$SCRIPT_DIR/req.sh" || FAILED_STEP=req
  [ -n "$FAILED_STEP" ] || step bench bench_cmd || FAILED_STEP=bench
  stop_cluster
  log "===== $GROUP done (${FAILED_STEP:+fail at }${FAILED_STEP:-ok}) ====="
}

if [ "${1:-}" = list ]; then
  printf '%s\n' "${ALL_GROUPS[@]}"
  echo "default: ${DEFAULT_GROUPS[*]}"
  exit 0
fi
TO_RUN=("$@")
[ $# -gt 0 ] || TO_RUN=("${DEFAULT_GROUPS[@]}")
for g in "${TO_RUN[@]}"; do
  define "$g" || { echo "unknown group: $g (bash $0 list)" >&2; exit 1; }
done

log "prefill test start: ${TO_RUN[*]}"
SUMMARY=() RC=0
for g in "${TO_RUN[@]}"; do
  run_group "$g"
  if [ -z "$FAILED_STEP" ]; then SUMMARY+=("$g: ok"); else SUMMARY+=("$g: FAIL at $FAILED_STEP"); RC=1; fi
done
log "prefill test finished"
printf '  %s\n' "${SUMMARY[@]}" | tee -a "$DRIVER_LOG"
exit $RC
