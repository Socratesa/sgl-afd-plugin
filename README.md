# sgl-afd-plugin

## How to install
```shell
cd path/to/sgl-afd-plugin/python

pip install -e .
```

## Example Usage
SGLang AFD Plugin is disabled by default. Set environment variable `SGLANG_AFD_PLUGIN_ENABLED=1` to enable it.
```shell
# node 0
SGLANG_AFD_PLUGIN_ENABLED=1 sglang serve --model-path /path/to/models/Qwen3-30B-A3B --host 0.0.0.0 --port 1234 --tp-size 4 --dp-size 4 --enable-dp-attention --nnodes 2 --node-rank 0 --dist-init-addr 127.0.0.1:3000 --base-gpu-id 0 --afd-role attn

# node 1
SGLANG_AFD_PLUGIN_ENABLED=1 sglang serve --model-path /path/to/models/Qwen3-30B-A3B --host 0.0.0.0 --port 1235 --tp-size 4 --dp-size 4 --enable-dp-attention --nnodes 2 --node-rank 1 --dist-init-addr 127.0.0.1:3000 --base-gpu-id 2 --afd-role ffn
```
