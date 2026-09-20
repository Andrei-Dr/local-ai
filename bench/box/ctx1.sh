#!/bin/bash
# CTX1: long-context latency via slot save/restore (C1) against slotclient.py — server managed HERE, not by specbench.
# Hypothesis: restore turns minutes of prefill into seconds (prompt_n of the request ~= 0 on a working restore),
# and decode at depth stays usable (the 55 t/s class must not collapse into GDN/checkpoint replay at 32k-64k).
# Kill: restore does NOT skip the prompt (processed prompt_n ~= full prompt) on the hybrid model => C1 dead for this arch.
# Watch: RS buffer size / KV buffer size lines (the n_rs_seq tax grows with depth; 64k may simply not fit 4 GB),
# out of memory kills the row and we continue — every depth is judged on its own.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
UB=${UB:-512}                      # env-tunable upload batch; streamed bytes per prompt token fall with it
KVARGS=${KVARGS:-}                 # extra KV-cache flags appended verbatim (e.g. -ctk q8_0)
mkdir -p /ai/bench/slots
run() { # run MODE LABEL CTX REPS [extra llama-server args...]   — fresh server per row, killed at the end
  local mode=$1 label=$2 ctx=$3 reps=$4; shift 4
  echo "##### $label mode=$mode ctx=$ctx reps=$reps | $* ${KVARGS}"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -c $ctx -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path /ai/bench/slots --moe-expert-cache 24 \
    -ub $UB -b $((UB * 2)) -fa on $KVARGS "$@" > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED: $(grep -iE 'error|failed|out of memory' server_$label.log | tail -1 | cut -c1-140)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  MODE=$mode REPS=$reps python3 /ai/bench/slotclient.py "$label"
  local rc=$?
  kill $pid; wait $pid 2>/dev/null
  grep -hE "KV buffer size|RS buffer size|out of memory" server_$label.log | tail -6 | cut -c1-200 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'
  [ $rc -eq 0 ] || echo "    $label: slotclient exit $rc (row kept in runs/$label.slot.json only if it got that far)"
  return 0
}
for pair in "16384 6" "32768 13" "65536 27"; do
  set -- $pair
  run save    ctx1_c${1}_save    $1 $2
  run restore ctx1_c${1}_restore $1 $2   # restore ALWAYS after a server restart (fresh process, slot from disk)
done
run save ctx1_c32768_nkvo_save 32768 13 -nkvo           # KV-only: does the replay cost hide in the non-KV path?
run save ctx1_c32768_q8_save   32768 13 -ctk q8_0 -ctv q8_0   # half-size KV buys what at depth?
echo "--- verdict input: compare runs/ctx1_c*_restore.slot.json rows (prompt_n near 0 + wall seconds vs the save-row"
echo "    cold leg's prompt_n/wall) and decode tps at depth; depths without a restore row died — check the server logs."
echo CTX1_DONE
