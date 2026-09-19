#!/bin/bash
# usage: specbench.sh NGL LABEL [extra llama-server args...]   (env OFFLOAD = GGML_OP_OFFLOAD_MIN_BATCH, default 2; MODEL; BUILD)
# Every run (completed or dead) appends one record to /ai/bench/ledger.jsonl via ledger.py.
source /ai/bench/preflight.sh || exit 1
NGL=$1; LABEL=$2; shift 2
export MODEL=${MODEL:-/ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf} BUILD=${BUILD:-/ai/src/llama.cpp/build} OFFLOAD=${OFFLOAD:-2}
export LEDGER_ARGS="-ngl $NGL $*"
rm -f /ai/bench/runs/$LABEL.client.json /ai/bench/runs/$LABEL.mon.json
GGML_OP_OFFLOAD_MIN_BATCH=$OFFLOAD $BUILD/bin/llama-server -m $MODEL -ngl $NGL -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 -lv 4 "$@" > /ai/bench/server_$LABEL.log 2>&1 &
PID=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "$LABEL: SERVER DIED: $(grep -iE 'error|failed|out of memory' /ai/bench/server_$LABEL.log | tail -1 | cut -c1-140)"; python3 /ai/bench/ledger.py $LABEL; exit 1; }; sleep 2; done
python3 /ai/bench/mon.py start $LABEL; python3 /ai/bench/specclient.py "$LABEL" "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; python3 /ai/bench/mon.py stop $LABEL
kill $PID; wait $PID 2>/dev/null
python3 /ai/bench/ledger.py $LABEL; exit 0
