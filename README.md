# sgl-afd-plugin

## How to start
```shell
cd path/to/sgl-afd-plugin/python

pip install -e .
```

## Example Usage
```shell
sglang serve --model-path /path/to/models/Qwen3-30B-A3B --host 0.0.0.0 --port 1234 --tp-size 2 --dp-size 2 --enable-dp-attention --afd-role attn
```
