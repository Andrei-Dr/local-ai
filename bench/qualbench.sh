#!/bin/bash
# usage: qualbench.sh LABEL [llama-server args...]   env: MODEL (required), OFFLOAD (default 32), BUILD, QARGS (extra qual.py args)
# Starts the server on the given config, runs the fixed quality eval (bench/qual/qual.py) with telemetry, prints the
# QUALITY[...] row and appends a kind=quality record to /ai/bench/ledger.jsonl.
source /ai/bench/preflight.sh || exit 1
LABEL=$1; shift
export MODEL BUILD=${BUILD:-/ai/src/llama.cpp/build75} OFFLOAD=${OFFLOAD:-32}
export LEDGER_ARGS="$*"
GGML_OP_OFFLOAD_MIN_BATCH=$OFFLOAD $BUILD/bin/llama-server -m $MODEL -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 "$@" > /ai/bench/server_qual_$LABEL.log 2>&1 &
PID=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "$LABEL: SERVER DIED: $(grep -iE 'error|failed|out of memory' /ai/bench/server_qual_$LABEL.log | tail -1 | cut -c1-140)"; exit 1; }; sleep 2; done
python3 /ai/bench/mon.py start qual_$LABEL
/ai/.venv/bin/python /ai/bench/qual/qual.py "$LABEL" $QARGS
python3 /ai/bench/mon.py stop qual_$LABEL; mv -f /ai/bench/runs/qual_$LABEL.mon.json /ai/bench/runs/$LABEL.mon.json
echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; echo "{\"vram\": \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)\", \"rows\": null}" > /ai/bench/runs/$LABEL.client.json
kill $PID; wait $PID 2>/dev/null
python3 /ai/bench/ledger.py $LABEL --quality; exit 0
