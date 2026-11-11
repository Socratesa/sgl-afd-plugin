#!/bin/bash
# Start the smg router on :20010 over WORKER_URLS (default: the engine of run_asap.sh / run_ep.sh on this tray), replacing
# whatever listens there. Returns once the router answers /workers, and, with EXPECT_RANKS=N, once it has discovered N DP
# ranks; fails (and stops the router) when that does not happen within ~60 s. The pid is written to $LOG_DIR/pd_router.pid.
# run_asap.sh and run_ep.sh call this after their engines are healthy; run it by hand when they were started with ROUTER=0.

set -euo pipefail

# Mix instance: prefill and decode share one process. Regular routing uses
# --policy; --prefill-policy / --decode-policy only apply with --pd-disaggregation.
#
# Same script for run_asap.sh (--dp-size D, separate replicas) and
# run_ep.sh (--attn-dp-size EP, one process). --dp-aware reads
# /server_info and expands each URL into dp_size * attn_dp_size ranks
# (discover_metadata.rs num_dp_ranks), then pins a request with
# data_parallel_rank, which the engine still accepts as routed_dp_rank.
# baseline reports dp_size=1, attn_dp_size=EP, so the product is EP.
HOST=${HOST:-0.0.0.0}
SERVE=${SERVE:-127.0.0.1}
ROUTER_PORT=${ROUTER_PORT:-20010}
# smg's prometheus listener (its own default is 29000). Keep it below 10000: ip_local_port_range here is 10000-61000, so any
# local client connection (hundreds of them to the node agent on :428) can hold a port of that range, and smg panics with
# EADDRINUSE when it takes one that is 29000.
METRICS_PORT=${METRICS_PORT:-9310}
# Several engines, e.g. the A instances of a multi-tray run_asap.sh: WORKER_URLS="http://IP1:30000 http://IP2:30000".
WORKER_URLS=${WORKER_URLS:-http://${SERVE}:30000}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/logs}
mkdir -p "$LOG_DIR"
# cargo build writes the binary next to the sglang tree, not the caller's cwd.
SMG_BIN=${SMG_BIN:-"$SCRIPT_DIR/../../sglang/sgl-model-gateway/target/release/smg"}
if [[ ! -x "$SMG_BIN" ]]; then
  echo "smg binary not found: $SMG_BIN (cargo build --release in sgl-model-gateway first)" >&2
  exit 1
fi

# lsof prints nothing when a port is free. SIGTERM first, SIGKILL for what is still bound after ~10 s.
listeners() { lsof -t -iTCP:"$ROUTER_PORT" -iTCP:"$METRICS_PORT" -sTCP:LISTEN || true; }
pids=$(listeners)
if [ -n "$pids" ]; then
  kill $pids 2> /dev/null || true
  for _ in $(seq 20); do [ -z "$(listeners)" ] && break; sleep 0.5; done
  pids=$(listeners)
  [ -z "$pids" ] || kill -9 $pids 2> /dev/null || true
fi

# cache aware 三个决策：
# 1. 判断负载是否失衡。 load 是路由器为每个 worker 统计的在途请求数：已发出、还没返回的请求。
# 失衡 = (max_load - min_load) > balance_abs_threshold   # 20
#     且  max_load > min_load * balance_rel_threshold     # 2
# 两个条件同时满足才算失衡。失衡时，所有请求都直接发给 load 最小的 worker，前缀树完全不参考。命中率再高的请求也一样。

# 2. 负载均衡时才走前缀匹配。 路由器为每个 worker 维护一棵近似 radix tree，内容是发给它的请求原文。这棵树按字符匹配，不是按 token。新请求进来时，在树里找最长前缀：
# 匹配率 > cache_threshold（0.3）：发给匹配最长的 worker，它大概率还缓存着这段 KV。
# 匹配率 ≤ 0.3：发给 load 最小的 worker。

# 3. 更新前缀树。 不管走哪条路径，都把这个请求插入被选中 worker 的树。
"$SMG_BIN" launch \
  --worker-urls ${WORKER_URLS} \
  --policy cache_aware \
  --dp-aware \
  --balance-abs-threshold 20 \
  --balance-rel-threshold 2 \
  --cache-threshold 0.3 \
  --host "${HOST}" \
  --port "${ROUTER_PORT}" \
  --prometheus-port "${METRICS_PORT}" \
  --log-level info \
  --disable-circuit-breaker \
  --health-check-interval-secs 999999 > "${LOG_DIR}/pd_router.log" 2>&1 &
ROUTER_PID=$!
echo "$ROUTER_PID" > "${LOG_DIR}/pd_router.pid"

# /workers: {"workers": [...], "total": N}, one entry per DP rank
total=-1
for _ in $(seq 30); do
  kill -0 "$ROUTER_PID" 2> /dev/null || { echo "smg exited, see ${LOG_DIR}/pd_router.log" >&2; exit 1; }
  total=$(curl --noproxy '*' -sf -m 5 "http://127.0.0.1:${ROUTER_PORT}/workers" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["total"])' 2> /dev/null || echo -1)
  if [ "$total" -ge 0 ] && { [ -z "${EXPECT_RANKS:-}" ] || [ "$total" = "$EXPECT_RANKS" ]; }; then
    exit 0
  fi
  sleep 2
done
echo "smg knows $total DP ranks, expected ${EXPECT_RANKS:-any}; see ${LOG_DIR}/pd_router.log" >&2
kill "$ROUTER_PID" 2> /dev/null || true
exit 1
