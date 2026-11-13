#!/bin/bash
# Two independent EP servers, one per tray, behind one smg router. This is not run_ep_cross.sh: that script is a
# single EP group whose ranks span both trays. Here each tray is its own EP (attention DP = EP) and the router spreads
# requests across both. Each tray is one run_ep.sh, started here or over ssh; Ctrl-C, or any engine exiting, stops all of them.
#   bash scripts/run_ep_pair.sh                              # EP4 on every hostfile tray, router :20010 here
#   NODES="ip1 ip2" EP=4 bash scripts/run_ep_pair.sh         # default: every hostfile tray
#   PER_DP_CHUNK=16384 bash scripts/run_ep_pair.sh
#   ROUTER=0 bash scripts/run_ep_pair.sh                     # engines only; prints the smg.sh line
#   MOE_BACKEND, PREFILL_MAX_RUNNING, EXTRA, LOG_DIR: as in run_ep.sh
# Engines bind their own tray IP:30000 (not 127.0.0.1, or the router on the other tray cannot reach them). The router
# starts after every engine's /health succeeds. Needs passwordless ssh to the other tray (SSH_PORT, default 2222) and MODEL
# at the same path on each. $LOG_DIR/ready appears when everything is up.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
RUN=$SCRIPT_DIR/run_ep.sh
FORWARD=(VENV MODEL MOE_BACKEND PER_DP_CHUNK PREFILL_MAX_RUNNING EXTRA LAUNCH_WRAP METRIC_MODEL)

MODEL=${MODEL:-/path/to/model}
EP=${EP:-4}
# two independent EP groups behind one router: ep4x2, not ep4. Set before NODES so a hostfile failure still leaves it set.
METRIC_MODEL=${METRIC_MODEL:-asap_dsv4_ep${EP}x2}
NODES=(${NODES:-$(hostfile_trays)})
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/ep${EP}_pair_$(date +%Y%m%d-%H%M%S)}

for t in "${NODES[@]}"; do
  on "$t" "test -d $(printf %q "$MODEL")" || { echo "cannot reach $t, or $t has no $MODEL"; exit 1; }
done
kill_servers "${NODES[@]}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/ready"

URLS=() PIDS=() NAMES=() LOGS=()
for r in "${!NODES[@]}"; do
  remote_run "${NODES[r]}" "$LOG_DIR/launch_EP$r.log" EP="$EP" HOST="${NODES[r]}" INSTANCE="EP$r" KILL=0 ROUTER=0 LOG_DIR="$LOG_DIR"
  PIDS+=($!) NAMES+=("EP$r") LOGS+=("$LOG_DIR/launch_EP$r.log") URLS+=("http://${NODES[r]}:30000")
  echo "EP$r: EP$EP on ${NODES[r]} -> $LOG_DIR/launch_EP$r.log"
done

stop_all() {
  local t
  for t in "${NODES[@]}"; do on "$t" "pkill -TERM -f '[s]glang.launch_server'" || true; done
  stop_router
}
trap 'stop_all; wait' EXIT
trap 'exit 130' INT TERM
healthy() {
  local url
  for url in "${URLS[@]}"; do curl --noproxy '*' -sf -m 5 "$url/health" > /dev/null || return 1; done
}
wait_ready
echo "ready: ${#NODES[@]} x EP$EP, moe=${MOE_BACKEND:-flashinfer_mxfp4}, ${PER_DP_CHUNK:-8192} tokens per rank"
WANT=$((${#NODES[@]} * EP))
if [ "${ROUTER:-1}" = 1 ]; then start_router "$WANT" "${URLS[@]}" || exit 1
else echo "router: WORKER_URLS=\"${URLS[*]}\" EXPECT_RANKS=$WANT LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh"; fi
mark_ready
wait -n "${PIDS[@]}" || true
for k in "${!PIDS[@]}"; do
  kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited (or the ssh to its tray dropped), see ${LOGS[k]}"; break; }
done
