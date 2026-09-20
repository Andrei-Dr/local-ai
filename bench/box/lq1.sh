#!/bin/bash
# LQ1: retrieval quality AT DEPTH (longctx.py, RULER-style NIAH + variable tracking) for the KV precisions
# the 262k rung actually needs. ctx1 measures SPEED at depth; this measures whether the model still WORKS
# there. One prefill per depth per row (many questions ride the cached prefix), so a row is minutes, not
# the many-hours a naive ask-per-doc sweep would cost.
# Hypothesis: q8_0 KV == f16 within 1 SE at every depth; q4_0 KV within 1 SE up to 32k and measurably
#   worse beyond (needle pct drops first at the 0.8-1.0 position bucket).
# Kill: q4 needle pct < 90% at 64k => the 262k rung needs q8 KV + -nkvo (or fewer slots), not q4.
# One server per row, --slot-save-path so repeated questions after a restart skip the doc prefill
# (longctx.py --slot-save). A dead row never aborts the job.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
SLOTS=/ai/bench/slots; mkdir -p $SLOTS
export TIMEOUT=21600
row() { # row LABEL CTX CACHE EXTRA_KV DEPTHS
  local label=$1 ctx=$2 cache=$3 extra=$4 depths=$5
  echo "##### $label ctx=$ctx cache=$cache kv='$extra' depths=$depths"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS \
    --moe-expert-cache $cache -ub 512 -b 1024 -fa on $extra > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED: $(grep -iE 'error|failed|out of memory' server_$label.log | tail -1 | cut -c1-140)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  python3 /ai/bench/longctx.py "$label" --depths $depths --slot-save
  local rc=$?
  echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $pid; wait $pid 2>/dev/null
  [ $rc -eq 0 ] || echo "    $label: longctx exit $rc (rows kept in results/$label.longctx.jsonl; reruns resume)"
  return 0
}
D_STD=4096,16384,32768,65536
row lq1_f16    65536 24 ""                  $D_STD
row lq1_q8     65536 24 "-ctk q8_0 -ctv q8_0" $D_STD
row lq1_q4     65536 24 "-ctk q4_0 -ctv q4_0" $D_STD
row lq1_q4_131k 131072 24 "-ctk q4_0 -ctv q4_0" 131072
rm -f $SLOTS/lq1_*.slot
rmdir $SLOTS 2>/dev/null
echo LQ1_DONE
