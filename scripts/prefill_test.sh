#!/bin/bash
# Prefill 对照。底部每行一条实验，环境变量只作用于这一条的 launcher。
#   bash scripts/prefill_test.sh
# 一组的顺序：后台启动 launch_* → sleep 120 → smg.sh → sleep 30 → req.sh → bench-qf-new-t6t7.sh。
# launch_* / smg.sh 启动前都会自己 kill 旧进程。某一组失败只记一笔，继续下一组。

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=/root/paddlejob/inference-public/lihaode/sgl_afd/bench-qf/bench-qf-new-t6t7.sh
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/logs}
mkdir -p "$LOG_DIR"
DRIVER_LOG=$LOG_DIR/prefill_test.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

run_group() {
  local name=$1
  local launcher=$2
  shift 2
  local launch_log=$LOG_DIR/prefill_${name}.launch.log

  log "===== ${name} start $* ====="
  env "$@" bash "$launcher" >"$launch_log" 2>&1 &
  log "${name}: launcher pid=$! log=${launch_log}"
  sleep 120

  if ! bash "$SCRIPT_DIR/smg.sh" >>"$DRIVER_LOG" 2>&1; then
    log "${name}: FAIL smg.sh"
    log "===== ${name} done (fail) ====="
    return 0
  fi
  sleep 30

  if bash "$SCRIPT_DIR/req.sh" >>"$DRIVER_LOG" 2>&1; then
    log "${name}: req ok"
  else
    log "${name}: FAIL req.sh"
    log "===== ${name} done (fail) ====="
    return 0
  fi

  if bash "$BENCH" >>"$DRIVER_LOG" 2>&1; then
    log "${name}: bench ok"
  else
    log "${name}: FAIL bench"
  fi
  log "===== ${name} done ====="
}

log "prefill test start"
run_group asap_2a2f "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1" F_GPUS="2 3"
run_group baseline_ep4 "$SCRIPT_DIR/run_ep.sh" EP=4
run_group baseline_ep4_delayer "$SCRIPT_DIR/run_ep.sh" EP=4 EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"
run_group asap_2a2f_delayer "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1" F_GPUS="2 3" A_EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"

# run_group asap_2a4f "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1" F_GPUS="4 5 6 7"
# run_group asap_4a4f "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1 2 3" F_GPUS="4 5 6 7"
# run_group baseline_ep8 "$SCRIPT_DIR/run_ep.sh" EP=8
# run_group baseline_ep4 "$SCRIPT_DIR/run_ep.sh" EP=4
# run_group baseline_ep8_delayer "$SCRIPT_DIR/run_ep.sh" EP=8 EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"
# run_group baseline_ep4_delayer "$SCRIPT_DIR/run_ep.sh" EP=4 EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"
# run_group asap_2a4f_delayer "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1" F_GPUS="4 5 6 7" A_EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"
# run_group asap_4a4f_delayer "$SCRIPT_DIR/run_asap.sh" A_GPUS="0 1 2 3" F_GPUS="4 5 6 7" A_EXTRA="--prefill-delayer-max-delay-passes 50 --enable-prefill-delayer"
log "prefill test finished"
