#!/bin/bash
# ASAP deployment over GB200 trays of one NVLink domain: F on F_HOST and one A instance (attention replicas) per entry of
# A_HOSTS. Each instance is one run_asap.sh (ROLE=ffn / ROLE=attn) on its tray, started here or over ssh; wait for every
# /health, then start the smg router (:20010) over the A instances. Ctrl-C, or any instance exiting, stops all of them.
#   bash scripts/run_asap_cross.sh                                 # 4A2F: A on GPUs 0-3 here, F on GPUs 0-1 of F_HOST
#   A_GPUS="0 1" F_GPUS="0 1 2 3" bash scripts/run_asap_cross.sh   # 2A4F
#   F_HOST=ip2 bash scripts/run_asap_cross.sh                              # default: the first host of the hostfile that is not this one
#   A_HOSTS="ip1:0,1,2,3 ip3:0,1,2,3" bash scripts/run_asap_cross.sh       # 8A2F: A instances on two trays
#   A_HOSTS="ip1:0,1,2,3 ip2:2,3" bash scripts/run_asap_cross.sh           # 6A2F: one of them next to F
#   TRANSPORT_MEM=auto bash scripts/run_asap_cross.sh              # --asap-transport-mem of every instance, default fabric
#   DBO, PER_DP_CHUNK, PREFILL_MAX_RUNNING, A_EXTRA, F_EXTRA, LOG_DIR: as in run_asap.sh
# Everything but the placement (hosts, GPUs, bootstrap address, replica numbers, transport memory) is run_asap.sh's.
#
# A_HOSTS: "HOST:GPU,GPU,..." entries, one A instance each and at most one per tray; default "127.0.0.1:<A_GPUS>",
# this tray. In the order of A_HOSTS, instance i takes the next replicas of the D in all (--asap-attn-base-replica,
# --asap-attn-dp-size). It serves http on HOST:30000 (127.0.0.1: loopback only); the script then starts the smg router
# (:20010) on all of them; ROUTER=0 only prints the smg.sh line instead, and the router is stopped with the servers.
#
# Needs (../net/README.md section 6): every tray in one NVLink domain (nvidia-smi -q, Fabric: same ClusterUUID and
# CliqueId), an IMEX channel in each container (/dev/nvidia-caps-imex-channels), ssh without a password to every
# other tray (SSH_PORT, default 2222; ~/.ssh/config can override it), and MODEL at the same path on each
# (often a local disk). The venv, the plugin and LOG_DIR are on shared storage, so all logs land here:
# launch_F.log, launch_A<i>.log (what each run_asap.sh prints) next to ffn.log, attn<i>.log, pd_router.log.
#
# Ports: A http HOST:30000 | A replica r process group 9500+r | F /health F_HOST:30100 | F process group 9590 |
# F rank f handshake F_HOST:9600+f (every A connects to it). Each instance has its own CUDA_VISIBLE_DEVICES and
# --base-gpu-id 0. It first kills the sglang servers on every tray it uses (an smg router keeps running until the next
# smg.sh replaces it). $LOG_DIR/ready appears when everything is up. Profiling: nsys_wrap.sh sets LAUNCH_WRAP, see lib.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
RUN=$SCRIPT_DIR/run_asap.sh
FORWARD=(VENV MODEL DBO PER_DP_CHUNK PREFILL_MAX_RUNNING A_EXTRA F_EXTRA LAUNCH_WRAP METRIC_MODEL)

VENV=${VENV:-/path/to/venv}
MODEL=${MODEL:-/path/to/model}
if [ -z "${F_HOST:-}" ]; then
  for h in $(hostfile_trays); do
    is_local "$h" || { F_HOST=$h; break; }
  done
fi
F_HOST=${F_HOST:?set F_HOST to the IP of the FFN tray}
# A_GPUS is a space-separated string, as in run_asap.sh ("0 1 2 3"); A_HOSTS is one HOST:GPU,GPU entry per word.
# Assign A_GPUS before joining it, or the default splits into "127.0.0.1:0" "1" "2" "3".
A_GPUS=(${A_GPUS:-0 1 2 3})
A_HOSTS=(${A_HOSTS:-127.0.0.1:$(IFS=,; echo "${A_GPUS[*]}")})
F_GPUS=(${F_GPUS:-0 1})
TRANSPORT_MEM=${TRANSPORT_MEM:-fabric}
E=${#F_GPUS[@]}
BOOTSTRAP=$F_HOST:9600

# A instance i: A_HOST[i], its GPUs A_DEV[i], its replicas [A_BASE[i], A_BASE[i] + A_N[i]) of D
A_HOST=() A_DEV=() A_BASE=() A_N=() D=0
for entry in "${A_HOSTS[@]}"; do
  [[ $entry =~ ^([^:]+):([0-9]+(,[0-9]+)*)$ ]] || { echo "A_HOSTS: not HOST:GPU,GPU,...: $entry"; exit 1; }
  host=${BASH_REMATCH[1]} devs=${BASH_REMATCH[2]}
  for other in "${A_HOST[@]}"; do
    same_tray "$host" "$other" && { echo "A_HOSTS: one A instance per tray, $host is there twice"; exit 1; }
  done
  if same_tray "$host" "$F_HOST"; then
    for g in ${devs//,/ }; do
      [[ " ${F_GPUS[*]} " == *" $g "* ]] && { echo "A on $host and F share GPU $g"; exit 1; }
    done
  fi
  IFS=, read -ra gpus <<< "$devs"
  A_HOST+=("$host") A_DEV+=("$devs") A_BASE+=("$D") A_N+=("${#gpus[@]}")
  D=$((D + ${#gpus[@]}))
done
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs/asap_${D}A${E}F_cross_$(date +%Y%m%d-%H%M%S)}
TRAYS=("$F_HOST")
for host in "${A_HOST[@]}"; do
  new=1
  for t in "${TRAYS[@]}"; do same_tray "$host" "$t" && new=0; done
  [ $new = 0 ] || TRAYS+=("$host")
done

for t in "${TRAYS[@]}"; do
  on "$t" "test -d $(printf %q "$MODEL")" || { echo "cannot reach $t, or $t has no $MODEL"; exit 1; }
done
# once, here: the run_asap.sh of an A instance next to F would otherwise kill the F that is already starting
kill_servers "${TRAYS[@]}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/ready"

# What every instance gets besides the forwarded variables: its tray, the common F address and the totals; KILL=0 because
# of the kill above, ROUTER=0 because the router is started below, over all the A instances.
COMMON=(F_GPUS="${F_GPUS[*]}" D_TOTAL="$D" BOOTSTRAP="$BOOTSTRAP" TRANSPORT_MEM="$TRANSPORT_MEM" KILL=0 ROUTER=0 LOG_DIR="$LOG_DIR")

remote_run "$F_HOST" "$LOG_DIR/launch_F.log" "${COMMON[@]}" ROLE=ffn HOST="$F_HOST"
PIDS=($!) NAMES=(F) LOGS=("$LOG_DIR/launch_F.log")
echo "F: $E ranks on $F_HOST GPUs ${F_GPUS[*]} -> $LOG_DIR/ffn.log"

URLS=()
for i in "${!A_HOST[@]}"; do
  tag=$( ((${#A_HOST[@]} > 1)) && echo "$i" || true)  # attn.log when there is one instance
  remote_run "${A_HOST[i]}" "$LOG_DIR/launch_A$tag.log" "${COMMON[@]}" ROLE=attn HOST="${A_HOST[i]}" \
    A_GPUS="${A_DEV[i]//,/ }" A_BASE="${A_BASE[i]}" A_TAG="$tag"
  PIDS+=($!) NAMES+=("A$tag") LOGS+=("$LOG_DIR/launch_A$tag.log") URLS+=("http://${A_HOST[i]}:30000")
  echo "A$tag: replicas ${A_BASE[i]}-$((A_BASE[i] + A_N[i] - 1)) of $D on ${A_HOST[i]} GPUs ${A_DEV[i]} -> $LOG_DIR/attn$tag.log"
done

# Killing a local ssh client would not stop its server, so every launcher gets SIGTERM on its tray (it stops its
# ranks itself, and the run_asap.sh that started it ends). The pattern names the role: A and F may share a tray.
stop_all() {
  local host
  for host in "${A_HOST[@]}"; do on "$host" "pkill -TERM -f '[s]glang.launch_server .*--asap-role attn'" || true; done
  on "$F_HOST" "pkill -TERM -f '[s]glang.launch_server .*--asap-role ffn'" || true
  stop_router
}
trap 'stop_all; wait' EXIT
trap 'exit 130' INT TERM
healthy() {
  local url
  curl --noproxy '*' -sf -m 5 "http://$F_HOST:30100/health" > /dev/null || return 1
  for url in "${URLS[@]}"; do curl --noproxy '*' -sf -m 5 "$url/health" > /dev/null || return 1; done
}
wait_ready
echo "ready: ${D}A + ${E}F, DBO=${DBO:-1}, transport $TRANSPORT_MEM"
if [ "${ROUTER:-1}" = 1 ]; then
  # The router's workers are exactly the A instances' addresses (they bind their tray IP, so smg.sh's default
  # 127.0.0.1:30000 reaches none of them). It knows D DP ranks once it has discovered every replica.
  start_router "$D" "${URLS[@]}" || exit 1
else
  echo "router: WORKER_URLS=\"${URLS[*]}\" EXPECT_RANKS=$D LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/smg.sh"
fi
mark_ready
wait -n "${PIDS[@]}" || true
for k in "${!PIDS[@]}"; do
  kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited (or the ssh to its tray dropped), see ${LOGS[k]}"; break; }
done
