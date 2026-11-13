#!/bin/bash
# EP baseline as one server whose EP ranks sit on several GB200 trays: NODES, GPUS_PER_NODE GPUs each (EP = nodes * GPUS_PER_NODE);
# the first node is this tray and serves http on 127.0.0.1:30000 (the others serve none). Each node is one run_ep.sh, started
# here or over ssh; wait for the first node's /health, then start the smg router (:20010). Ctrl-C, or any node exiting,
# stops all of them.
#   bash scripts/run_ep_cross.sh                              # EP8: GPUs 0-3 here and on the other hostfile tray
#   PER_DP_CHUNK=16384 bash scripts/run_ep_cross.sh           # 16k tokens per rank, 128k in all
#   NODES="ip1 ip2" GPUS_PER_NODE=4 bash ...                  # explicit placement, the first must be this tray
#   MOE_BACKEND=flashinfer_mxfp4 bash scripts/run_ep_cross.sh
#   MOE_BACKEND, PER_DP_CHUNK, PREFILL_MAX_RUNNING, EXTRA, LOG_DIR, ROUTER: as in run_ep.sh
# Needs the same as run_asap_cross.sh: passwordless ssh to every other tray (SSH_PORT, default 2222), MODEL at the same path
# on each (often a local disk); the venv, the plugin and LOG_DIR are on shared storage, where
# launch_EP<r>.log (what each run_ep.sh prints) lands next to the engine logs and pd_router.log.
# Ports: http 127.0.0.1:30000 | torch distributed init NODE0:9580 | nccl 9590 | router 20010.
# Like run_asap_cross.sh it first kills the sglang servers on every tray it uses. $LOG_DIR/ready appears when everything is up.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
RUN=$SCRIPT_DIR/run_ep.sh
FORWARD=(VENV MODEL MOE_BACKEND PER_DP_CHUNK PREFILL_MAX_RUNNING EXTRA LAUNCH_WRAP METRIC_MODEL)

MODEL=${MODEL:-/path/to/model}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODES=(${NODES:-$(hostfile_trays)})
N=${#NODES[@]}
EP=$((N * GPUS_PER_NODE))
is_local "${NODES[0]}" || { echo "NODES: the first node must be this tray, got ${NODES[0]}"; exit 1; }
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/baseline_ep${EP}_cross_$(date +%Y%m%d-%H%M%S)}
URL=http://127.0.0.1:30000

for t in "${NODES[@]}"; do
  on "$t" "test -d $(printf %q "$MODEL")" || { echo "cannot reach $t, or $t has no $MODEL"; exit 1; }
done
kill_servers "${NODES[@]}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/ready"

PIDS=() NAMES=() LOGS=()
for r in "${!NODES[@]}"; do
  remote_run "${NODES[r]}" "$LOG_DIR/launch_EP$r.log" EP="$EP" GPUS_PER_NODE="$GPUS_PER_NODE" NNODES="$N" NODE_RANK="$r" \
    DIST_INIT_ADDR="${NODES[0]}:9580" INSTANCE="EP$r" KILL=0 ROUTER=0 LOG_DIR="$LOG_DIR"
  PIDS+=($!) NAMES+=("EP$r") LOGS+=("$LOG_DIR/launch_EP$r.log")
  echo "node $r: ${NODES[r]} GPUs 0-$((GPUS_PER_NODE - 1)) -> $LOG_DIR/launch_EP$r.log"
done

# killing a local ssh client would not stop its server, so every tray gets SIGTERM for its launcher
stop_all() {
  local t
  for t in "${NODES[@]}"; do on "$t" "pkill -TERM -f '[s]glang.launch_server'" || true; done
  stop_router
}
trap 'stop_all; wait' EXIT
trap 'exit 130' INT TERM
healthy() { curl --noproxy '*' -sf -m 5 "$URL/health" > /dev/null; }
wait_ready
echo "ready: EP$EP over $N trays, moe=${MOE_BACKEND:-flashinfer_mxfp4}, ${PER_DP_CHUNK:-8192} tokens per rank"
if [ "${ROUTER:-1}" = 1 ]; then start_router "$EP" "$URL" || exit 1
else echo "router: WORKER_URLS=\"$URL\" EXPECT_RANKS=$EP LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh"; fi
mark_ready
wait -n "${PIDS[@]}" || true
for k in "${!PIDS[@]}"; do
  kill -0 "${PIDS[k]}" 2>/dev/null || { echo "node $k exited (or the ssh to it dropped), see ${LOGS[k]}"; break; }
done
