#!/usr/bin/env bash
# 与线上 PD router 同一份启动参数, 只是现在不是 PD 分离。
# 非 PD 时 --prefill-policy / --decode-policy 不生效, 同一套 manual + min_group 挂在 --policy 上。
# 调用约定与 smg.sh 相同: WORKER_URLS / EXPECT_RANKS / LOG_DIR, 端口 router :20010, prometheus :9310。
# 后台启动, pid 写到 $LOG_DIR/pd_router.pid, /workers 发现 EXPECT_RANKS 个 DP rank 后返回, 约 60 s 不到则停掉并失败。
set -uo pipefail

HOST=${HOST:-0.0.0.0}
SERVE=${SERVE:-127.0.0.1}
ROUTER_PORT=${ROUTER_PORT:-20010}
ROUTER_PROMETHEUS_PORT=${ROUTER_PROMETHEUS_PORT:-9310}
# 多个 engine: WORKER_URLS="http://IP1:30000 http://IP2:30000"
WORKER_URLS=${WORKER_URLS:-http://${SERVE}:30000}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs}
mkdir -p "$LOG_DIR"

unset PYTHONPATH
ulimit -n 262144 2>/dev/null || true
# run_asap.sh 不激活 venv, 只靠 PATH 找到 python; sglang_router 装在 VENV 里, 必须用那个解释器。
PY=python
if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${VENV:-}" && -f "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
fi

# lsof 在端口空闲时不输出。先 SIGTERM, 约 10 s 后仍占用的再 SIGKILL。
listeners() { lsof -t -iTCP:"$ROUTER_PORT" -iTCP:"$ROUTER_PROMETHEUS_PORT" -sTCP:LISTEN || true; }
pids=$(listeners)
if [ -n "$pids" ]; then
  kill $pids 2> /dev/null || true
  for _ in $(seq 20); do [ -z "$(listeners)" ] && break; sleep 0.5; done
  pids=$(listeners)
  [ -z "$pids" ] || kill -9 $pids 2> /dev/null || true
fi

# shellcheck disable=SC2086
"$PY" -m sglang_router.cli launch \
  --worker-urls ${WORKER_URLS} \
  --policy manual \
  --dp-aware \
  --host "${HOST}" --port "${ROUTER_PORT}" \
  --prometheus-port "${ROUTER_PROMETHEUS_PORT}" \
  --log-level info \
  --queue-size 5000 --queue-timeout-secs 7200 --request-timeout-secs 7200 \
  --max-concurrent-requests 3000 --rate-limit-tokens-per-second 512 \
  --retry-max-retries 1 \
  --health-check-timeout-secs 20 --health-failure-threshold 5 --health-success-threshold 2 --worker-startup-check-interval 30 \
  --pool-idle-timeout-secs 30 --pool-max-idle-per-host 0 \
  --assignment-mode min_group \
  > "${LOG_DIR}/pd_router.log" 2>&1 &
ROUTER_PID=$!
echo "$ROUTER_PID" > "${LOG_DIR}/pd_router.pid"

# /workers: {"workers": [...], "total": N}, 每个 DP rank 一条。
# worker 注册是后台任务, engine 刚 healthy 时 total 先是 0, 所以等 ROUTER_WAIT_SECS 而不是固定 60 s。
total=-1
deadline=$((SECONDS + ${ROUTER_WAIT_SECS:-300}))
while [ "$SECONDS" -lt "$deadline" ]; do
  kill -0 "$ROUTER_PID" 2> /dev/null || { echo "sgl-router exited, see ${LOG_DIR}/pd_router.log" >&2; exit 1; }
  total=$(curl --noproxy '*' -sf -m 5 "http://127.0.0.1:${ROUTER_PORT}/workers" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["total"])' 2> /dev/null || echo -1)
  if [ "$total" -ge 0 ] && { [ -z "${EXPECT_RANKS:-}" ] || [ "$total" = "$EXPECT_RANKS" ]; }; then
    exit 0
  fi
  sleep 2
done
echo "sgl-router knows $total DP ranks, expected ${EXPECT_RANKS:-any}; see ${LOG_DIR}/pd_router.log" >&2
kill "$ROUTER_PID" 2> /dev/null || true
exit 1
