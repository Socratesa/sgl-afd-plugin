#!/bin/bash
# One short /generate through the router (default :20010); fails unless the answer carries generated text.
URL=${URL:-http://127.0.0.1:20010}
resp=$(curl --noproxy '*' -sf -m 120 "$URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"please introduce baidu and claude","sampling_params":{"max_new_tokens":32,"temperature":0}}') \
  || { echo "req: no answer, or an HTTP error, from $URL/generate" >&2; exit 1; }
echo "$resp"
grep -q '"text"' <<< "$resp" || { echo "req: no \"text\" in the answer" >&2; exit 1; }
