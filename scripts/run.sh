SGLANG_AFD_PLUGIN_ENABLED=1 

# Luanch Attention
sglang serve --model-path Qwen3-30B-A3B --host 0.0.0.0 --port 1234 --tp-size 2 --dp-size 2 --enable-dp-attention --afd-role attn

# Luanch FFN
sglang serve --model-path Qwen3-30B-A3B --host 0.0.0.0 --port 1234 --tp-size 2 --dp-size 2 --enable-dp-attention --afd-role ffn