#!/bin/bash

set -euo pipefail

kill $(lsof -t -iTCP:20010 -sTCP:LISTEN) || true

sleep 20

# Mix instance: prefill and decode share one process. Regular routing uses
# --policy; --prefill-policy / --decode-policy only apply with --pd-disaggregation.
#
# Same script for run_asap.sh (--dp-size D, separate replicas) and
# run_ep.sh (--attn-dp-size EP, one process). --dp-aware reads
# /server_info and expands each URL into dp_size * attn_dp_size ranks
# (discover_metadata.rs num_dp_ranks), then pins a request with
# data_parallel_rank, which the engine still accepts as routed_dp_rank.
# baseline reports dp_size=1, attn_dp_size=EP, so the product is EP.
# run_asap.sh and run_ep.sh both bind the engine to 127.0.0.1:30000.
HOST=${HOST:-0.0.0.0}
SERVE=${SERVE:-127.0.0.1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/logs}
mkdir -p "$LOG_DIR"
# cargo build writes the binary next to the sglang tree, not the caller's cwd.
SMG_BIN=${SMG_BIN:-"$SCRIPT_DIR/../../sglang/sgl-model-gateway/target/release/smg"}
if [[ ! -x "$SMG_BIN" ]]; then
  echo "smg binary not found: $SMG_BIN (cargo build --release in sgl-model-gateway first)" >&2
  exit 1
fi
"$SMG_BIN" launch \
  --worker-urls "http://${SERVE}:30000" \
  --policy cache_aware \
  --dp-aware \
  --balance-abs-threshold 20 \
  --balance-rel-threshold 2 \
  --cache-threshold 0.3 \
  --host "${HOST}" \
  --port 20010 \
  --log-level info \
  --disable-circuit-breaker \
  --health-check-interval-secs 999999 > "${LOG_DIR}/pd_router.log" 2>&1 &
