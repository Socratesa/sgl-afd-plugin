#!/bin/bash
# ASAP deployment over GB200 trays of one NVLink domain: F (experts) on F_HOST and one A instance (attention
# replicas) per entry of A_HOSTS, started here or over ssh; wait for every /health. Ctrl-C, or any instance exiting,
# stops all of them.
#   bash scripts/run_asap_cross.sh                                 # 4A2F: A on GPUs 0-3 here, F on GPUs 0-1 of F_HOST
#   A_GPUS="0 1" F_GPUS="0 1 2 3" bash scripts/run_asap_cross.sh   # 2A4F
#   F_HOST=10.99.91.3 bash scripts/run_asap_cross.sh               # default: the first host of the hostfile that is not this one
#   A_HOSTS="10.99.91.2:0,1,2,3 10.99.91.4:0,1,2,3" bash scripts/run_asap_cross.sh   # 8A2F: A instances on two trays
#   A_HOSTS="10.99.91.2:0,1,2,3 10.99.91.3:2,3" bash scripts/run_asap_cross.sh       # 6A2F: one of them next to F
#   TRANSPORT_MEM=auto bash scripts/run_asap_cross.sh              # --asap-transport-mem of every instance, default fabric
#   DBO, PER_DP_CHUNK, PREFILL_MAX_RUNNING, A_EXTRA, F_EXTRA, LOG_DIR: as in run_asap.sh
# Everything but the placement (hosts, GPUs, bootstrap address, replica numbers, transport memory) is run_asap.sh's.
#
# A_HOSTS: "HOST:GPU,GPU,..." entries, one A instance each and at most one per tray; default "127.0.0.1:<A_GPUS>",
# this tray. In the order of A_HOSTS, instance i takes the next replicas of the D in all (--asap-attn-base-replica,
# --asap-attn-dp-size). It serves http on HOST:30000 (127.0.0.1: loopback only); the script prints the smg.sh line
# that routes to all of them.
#
# Needs (../net/README.md section 6): every tray in one NVLink domain (nvidia-smi -q, Fabric: same ClusterUUID and
# CliqueId), an IMEX channel in each container (/dev/nvidia-caps-imex-channels), ssh without a password to every
# other tray (port from ~/.ssh/config), and MODEL at the same path on each (/root/paddlejob/workspace is a local
# disk). The venv, the plugin and LOG_DIR are on the shared /root/paddlejob/inference-public, so all logs land here.
#
# Ports: A http HOST:30000 | A replica r process group 9500+r | F /health F_HOST:30100 | F process group 9590 |
# F rank f handshake F_HOST:9600+f (every A connects to it). Each instance has its own CUDA_VISIBLE_DEVICES and
# --base-gpu-id 0. Like run_asap.sh, it first kills the sglang servers, here on every tray it uses (an smg router
# keeps running).

set -euo pipefail

VENV=${VENV:-/root/paddlejob/inference-public/lihaode/sgl_afd/.venv}
MODEL=${MODEL:-/root/paddlejob/workspace/models/DeepSeek-V4-Flash-0731}
if [ -z "${F_HOST:-}" ]; then
  for h in $(awk '{print $1}' /root/paddlejob/workspace/hostfile); do
    [[ " $(hostname -I) " == *" $h "* ]] || { F_HOST=$h; break; }
  done
fi
F_HOST=${F_HOST:?set F_HOST to the IP of the FFN tray}
# A_GPUS is a space-separated string, as in run_asap.sh ("0 1 2 3"); A_HOSTS is one HOST:GPU,GPU entry per word.
# Assign A_GPUS before joining it, or the default splits into "127.0.0.1:0" "1" "2" "3".
A_GPUS=(${A_GPUS:-0 1 2 3})
A_HOSTS=(${A_HOSTS:-127.0.0.1:$(IFS=,; echo "${A_GPUS[*]}")})
F_GPUS=(${F_GPUS:-0 1})
DBO=${DBO:-1}
TRANSPORT_MEM=${TRANSPORT_MEM:-fabric}
PER_DP_CHUNK=${PER_DP_CHUNK:-16384}
PREFILL_MAX_RUNNING=${PREFILL_MAX_RUNNING:-512}
E=${#F_GPUS[@]}
BOOTSTRAP=$F_HOST:9600

is_local() { [[ $1 == 127.0.0.1 || $1 == localhost || " $(hostname -I) " == *" $1 "* ]]; }
same_tray() { [[ $1 == "$2" ]] || { is_local "$1" && is_local "$2"; }; }
# on HOST CMD: run the shell command CMD on HOST, here or over ssh
on() {
  if is_local "$1"; then bash -c "$2"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; fi
}

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
LOG_DIR=${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/asap_${D}A${E}F_cross_$(date +%Y%m%d-%H%M%S)}
TRAYS=("$F_HOST")
for host in "${A_HOST[@]}"; do
  new=1
  for t in "${TRAYS[@]}"; do same_tray "$host" "$t" && new=0; done
  [ $new = 0 ] || TRAYS+=("$host")
done

for t in "${TRAYS[@]}"; do
  on "$t" "test -d $(printf %q "$MODEL")" || { echo "cannot reach $t, or $t has no $MODEL"; exit 1; }
done
# sglang servers only (launchers and their sglang:: workers), not other tools under an sglang path such as the smg
# router; [s]: the pattern does not match the shell running pkill
for t in "${TRAYS[@]}"; do on "$t" "pkill -9 -f '[s]glang(\.launch_server|::)'" || true; done

DBO_FLAGS=()
if [ "$DBO" = 1 ]; then
  DBO_FLAGS=(--enable-two-batch-overlap --asap-max-micro-batches 2 --asap-tbo-min-tokens 12288 --tbo-token-distribution-threshold 0.48)
fi
mkdir -p "$LOG_DIR"

# launch HOST GPUS LOG ARGV...: run ARGV on HOST in the background, GPUS visible, output in LOG. The shell there
# execs it, so the background job (local shell or ssh client) lives exactly as long as the server and its pid tells
# whether the server still runs. Every argument is %q-quoted: the server gets the argv it would get when started here.
launch() {
  local host=$1 gpus=$2 log=$3
  shift 3
  on "$host" "export PATH=$VENV/bin:\$PATH CUDA_VISIBLE_DEVICES=$gpus SGLANG_PLUGINS=sgl_asap_plugin \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 PYTHONDONTWRITEBYTECODE=1 && exec $(printf '%q ' "$@")> $(printf %q "$log") 2>&1" &
}

F_VISIBLE=$(IFS=,; echo "${F_GPUS[*]}")
launch "$F_HOST" "$F_VISIBLE" "$LOG_DIR/ffn.log" \
  python -m sglang.launch_server --model-path "$MODEL" --host "$F_HOST" --port 30100 --nccl-port 9590 \
  --tp-size "$E" --ep-size "$E" --base-gpu-id 0 \
  --asap-max-dispatch-tokens "$PER_DP_CHUNK" --asap-ffn-cuda-graph-max-tokens "$PER_DP_CHUNK" \
  --mem-fraction-static 0.85 \
  --moe-runner-backend flashinfer_mxfp4 \
  --asap-role ffn --asap-attn-dp-size "$D" --asap-attn-tp-size 1 \
  --asap-bootstrap-addr "$BOOTSTRAP" --asap-transport-mem "$TRANSPORT_MEM" \
  ${NVTX_LAYER_FLAGS:-} \
  ${F_EXTRA:-}
PIDS=($!) NAMES=(F) LOGS=("$LOG_DIR/ffn.log")
echo "F: $E ranks on $F_HOST GPUs $F_VISIBLE -> $LOG_DIR/ffn.log"

URLS=()
for i in "${!A_HOST[@]}"; do
  tag=$( ((${#A_HOST[@]} > 1)) && echo "$i" || true)  # attn.log when there is one instance
  log=$LOG_DIR/attn$tag.log
  launch "${A_HOST[i]}" "${A_DEV[i]}" "$log" \
    python -m sglang.launch_server --model-path "$MODEL" --host "${A_HOST[i]}" --port 30000 \
    --dp-size "${A_N[i]}" --tp-size 1 --base-gpu-id 0 --load-balance-method total_tokens \
    --context-length 132072 --allow-auto-truncate \
    --max-running-requests "${PREFILL_MAX_RUNNING}" \
    --asap-max-dispatch-tokens "$PER_DP_CHUNK" --chunked-prefill-size "$PER_DP_CHUNK" --max-prefill-tokens "$PER_DP_CHUNK" \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --fp8-gemm-backend deep_gemm \
    "${DBO_FLAGS[@]}" \
    --extra-metric-labels "{\"model_name\": \"lhd_dsv4\"}" \
    --enable-metrics --enable-mfu-metrics --enable-metrics-for-all-schedulers --enable-cache-report \
    --enable-request-time-stats-logging --export-metrics-to-file --export-metrics-to-file-dir "$LOG_DIR/metrics_${D}a${E}f$tag" \
    --asap-role attn --asap-ffn-size "$E" --asap-attn-dp-size "$D" --asap-attn-base-replica "${A_BASE[i]}" \
    --asap-bootstrap-addr "$BOOTSTRAP" --asap-transport-mem "$TRANSPORT_MEM" \
    --asap-min-batch-tokens 12288 --asap-max-wait-ms 100 \
    ${NVTX_LAYER_FLAGS:-} \
    ${A_EXTRA:-}
  PIDS+=($!) NAMES+=("A$tag") LOGS+=("$log") URLS+=("http://${A_HOST[i]}:30000")
  echo "A$tag: replicas ${A_BASE[i]}-$((A_BASE[i] + A_N[i] - 1)) of $D on ${A_HOST[i]} GPUs ${A_DEV[i]} -> $log"
done

# Killing a local ssh client would not stop its server, so every launcher gets SIGTERM on its tray (it stops its
# ranks itself). The pattern names the role: A and F may share a tray.
stop_all() {
  for host in "${A_HOST[@]}"; do on "$host" "pkill -TERM -f '[s]glang.launch_server .*--asap-role attn'" || true; done
  on "$F_HOST" "pkill -TERM -f '[s]glang.launch_server .*--asap-role ffn'" || true
}
trap 'stop_all; wait' EXIT
trap 'exit 130' INT TERM
healthy() {
  curl --noproxy '*' -sf -m 5 "http://$F_HOST:30100/health" > /dev/null || return 1
  for url in "${URLS[@]}"; do curl --noproxy '*' -sf -m 5 "$url/health" > /dev/null || return 1; done
}
until healthy; do
  for k in "${!PIDS[@]}"; do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited, see ${LOGS[k]}"; exit 1; }
  done
  sleep 5
done
echo "ready: ${D}A + ${E}F, $PER_DP_CHUNK tokens per A replica, DBO=$DBO, transport $TRANSPORT_MEM"
echo "router: WORKER_URLS=\"${URLS[*]}\" bash $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smg.sh"
wait -n "${PIDS[@]}" || true
for k in "${!PIDS[@]}"; do
  kill -0 "${PIDS[k]}" 2>/dev/null || { echo "${NAMES[k]} exited (or the ssh to its tray dropped), see ${LOGS[k]}"; break; }
done
