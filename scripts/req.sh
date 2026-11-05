curl -s http://127.0.0.1:20010/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"please introduce baidu and claude","sampling_params":{"max_new_tokens":32,"temperature":0}}'