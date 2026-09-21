# Cheatsheet. Run each block in its own terminal, node 0 first.
#
# One SGLang world across two nodes: --tp-size 4 split over --nnodes 2, so each node
# owns 2 global tp_ranks (node 0 -> 0,1 / node 1 -> 2,3). --dp-size 4 with dp attention
# gives 4 DP slots, one per rank, and --afd-role is assigned per node -- that is the
# granularity the plugin uses to decide which DP slots are FFN and must be excluded
# from request routing.
#
# Both nodes are on the same box here, hence the same --dist-init-addr and the
# --base-gpu-id offset so they do not fight over GPU 0/1. On two real machines drop
# --base-gpu-id and point --dist-init-addr at node 0.
#
# `export` matters: a bare `VAR=1` line only sets a shell variable, which
# `sglang serve` (a child process) would never see.
export SGLANG_AFD_PLUGIN_ENABLED=1

MODEL=Qwen3-30B-A3B
DIST_INIT_ADDR=127.0.0.1:3000

# node 0 -- Attention (global tp_rank 0,1 -> DP slot 0,1 -> serves requests)
sglang serve --model-path "$MODEL" --host 0.0.0.0 --port 1234 \
  --tp-size 4 --dp-size 4 --enable-dp-attention \
  --nnodes 2 --node-rank 0 --dist-init-addr "$DIST_INIT_ADDR" --base-gpu-id 0 \
  --load-balance-method round_robin \
  --afd-role attn

# node 1 -- FFN (global tp_rank 2,3 -> DP slot 2,3 -> excluded from routing)
sglang serve --model-path "$MODEL" --host 0.0.0.0 --port 1235 \
  --tp-size 4 --dp-size 4 --enable-dp-attention \
  --nnodes 2 --node-rank 1 --dist-init-addr "$DIST_INIT_ADDR" --base-gpu-id 2 \
  --load-balance-method round_robin \
  --afd-role ffn
