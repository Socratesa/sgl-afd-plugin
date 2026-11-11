#!/bin/bash
# Shared by the run_*.sh scripts, nsys_wrap.sh and prefill_test.sh (sourced, not run): trays, launching, readiness, router.
# run_asap.sh and run_ep.sh are the one-tray scripts and use launch / wait_ready / start_router; run_asap_cross.sh,
# run_ep_cross.sh and run_ep_pair.sh run those on several trays with remote_run and start the router themselves.
# The caller sets SCRIPT_DIR, VENV, PLUGINS and LOG_DIR before using the functions that need them.
#
# Profiler contract (the only thing the launchers know about nsys_wrap.sh, and it is not nsys-specific): when LAUNCH_WRAP
# is set, `launch` runs the server as `$LAUNCH_WRAP HOST NAME -- ARGV...` on HOST, with LAUNCH_GPUS (the physical GPUs the
# instance owns) and LAUNCH_DEPLOY (a deployment tag such as 6a2f or ep4) in its environment. NAME is the instance (F, A0,
# A1, ..., EP, EP0, ...).

SSH_PORT=${SSH_PORT:-2222}
is_local() { [[ $1 == 127.0.0.1 || $1 == localhost || " $(hostname -I) " == *" $1 "* ]]; }
same_tray() { [[ $1 == "$2" ]] || { is_local "$1" && is_local "$2"; }; }
# on HOST CMD: run the shell command CMD on HOST, here or over ssh (the trays listen on SSH_PORT, not 22)
on() {
  if is_local "$1"; then bash -c "$2"
  else ssh -n -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$1" "$2"
  fi
}
# on_bg HOST CMD: on, in the background. Called as a plain function (not `on ... &`, whose subshell would hide the process),
# so $! afterwards is the process that runs CMD: its shell here (which execs the server), the ssh client there.
on_bg() {
  if is_local "$1"; then bash -c "$2" &
  else ssh -n -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$1" "$2" &
  fi
}

# hostfile_trays: the trays of the hostfile, this one first
hostfile_trays() {
  local h trays=
  for h in $(awk '{print $1}' "${HOSTFILE:-/root/paddlejob/workspace/hostfile}"); do
    if is_local "$h"; then trays="$h $trays"; else trays="$trays $h"; fi
  done
  echo $trays
}

# kill_servers TRAY...: the sglang launchers and their sglang:: workers on each tray. [s] keeps the pattern from matching
# the shell that runs pkill. The smg router stays: its path contains "sglang/", not "sglang.launch_server" or "sglang::".
kill_servers() {
  local t
  for t in "$@"; do on "$t" "pkill -9 -f '[s]glang(\.launch_server|::)'" || true; done
}

# launch NAME HOST CVD OWN LOG ARGV...: run ARGV on HOST in the background, CVD as its CUDA_VISIBLE_DEVICES, output in LOG.
# OWN is the list of physical GPUs the instance uses (CVD may show more, see run_asap.sh on one tray). The shell there execs
# ARGV, so the background job (local shell or ssh client) lives exactly as long as the server and its pid tells whether the
# server still runs. Every argument is %q-quoted: the server gets the argv it would get when started here.
launch() {
  local name=$1 host=$2 cvd=$3 own=$4 log=$5 wrap=
  shift 5
  [ -z "${LAUNCH_WRAP:-}" ] || wrap="$LAUNCH_WRAP $(printf '%q %q' "$host" "$name") -- "
  on_bg "$host" "export PATH=$VENV/bin:\$PATH CUDA_VISIBLE_DEVICES=$cvd SGLANG_PLUGINS=$PLUGINS \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 PYTHONDONTWRITEBYTECODE=1 LAUNCH_GPUS=$(printf %q "$own") \
LAUNCH_DEPLOY=$(printf %q "${DEPLOY:-}") && exec $wrap$(printf '%q ' "$@")> $(printf %q "$log") 2>&1"
}

# remote_run HOST LOG VAR=VALUE...: run `bash $RUN` (the run script of the caller) on HOST in the background, output in LOG.
# ssh passes no environment, so it gets the variables named in the caller's FORWARD array (those that are set), then the
# VAR=VALUEs, which win. $! is the job that lives as long as the script there (its shell, or the ssh client).
remote_run() {
  local host=$1 log=$2 v envs=
  shift 2
  for v in "${FORWARD[@]}"; do
    [ -z "${!v+x}" ] || envs+=" $v=$(printf %q "${!v}")"
  done
  for v in "$@"; do envs+=" $(printf %q "$v")"; done
  on_bg "$host" "env$envs bash $(printf %q "$RUN") > $(printf %q "$log") 2>&1"
}

# wait_ready: until healthy (the caller's function) succeeds; exits 1 as soon as an instance (PIDS, NAMES, LOGS) is gone.
wait_ready() {
  local k
  until healthy; do
    for k in "${!PIDS[@]}"; do
      kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited, see ${LOGS[k]}"; exit 1; }
    done
    sleep 5
  done
}

# start_router EXPECT_RANKS URL...: smg.sh over the engines' URLs (it replaces whatever listens on :20010 and returns once
# the router knows EXPECT_RANKS DP ranks). Needs LOG_DIR. HOST is the engines' bind address in the callers, not the router's
# (ROUTER_HOST, default all interfaces).
start_router() {
  local expect=$1
  shift
  if HOST=${ROUTER_HOST:-0.0.0.0} EXPECT_RANKS=$expect WORKER_URLS="$*" LOG_DIR="$LOG_DIR" bash "$SCRIPT_DIR/smg.sh"; then
    ROUTER_STARTED=1
    echo "router: :${ROUTER_PORT:-20010} -> $* (log $LOG_DIR/pd_router.log)"
  else
    echo "router: smg.sh failed, see $LOG_DIR/pd_router.log; start it with: WORKER_URLS=\"$*\" LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh" >&2
    return 1
  fi
}
stop_router() {
  [ "${ROUTER_STARTED:-0}" = 1 ] && [ -s "$LOG_DIR/pd_router.pid" ] || return 0
  kill -TERM "$(cat "$LOG_DIR/pd_router.pid")" 2>/dev/null || true
}

# $LOG_DIR/ready appears once the engines are healthy and the router knows every rank; whoever starts the router writes it,
# so a run with ROUTER=0 (a tray's share of a cross/pair deployment) does not (prefill_test.sh waits for it)
mark_ready() { : > "$LOG_DIR/ready"; }
