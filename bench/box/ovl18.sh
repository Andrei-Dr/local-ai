#!/bin/bash
# OVL18: find the host cost of GGML_SCHED_MOE_PREFETCH=1 when NOTHING is planned. ovl17 (12:26, ABBA): unset == =0 (+0.5%, spread
# 0.53); =1 with MIN_IDS=inf (plan block runs, never plans) -7.1% decode, code prefill 56.8 -> 48.0. The only mode-dependent code is
# the plan block in split_graph. perf record (call graphs) of the server during the specbench run, U vs N -> top symbols.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
for a in U N; do
  E=""; [ $a = N ] && E="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_MIN_IDS=99999999"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 4096 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp \
    --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048 > server_ovl18_$a.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  perf record -F 999 -g -p $NP -o /ai/bench/ovl18_$a.perf -- sleep 40 > /dev/null 2>&1 &
  PP=$!
  sleep 1
  for q in 1 2; do
    curl -s localhost:8099/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."}],"temperature":0,"max_tokens":300,"chat_template_kwargs":{"enable_thinking":false}}' \
      | python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(f\"    $a: prompt {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} t/s | decode {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} t/s\")"
  done
  wait $PP
  kill $NP; wait $NP 2>/dev/null
  echo "  --- $a: top self symbols"
  perf report -i /ai/bench/ovl18_$a.perf --no-children --sort symbol --stdio 2>/dev/null | grep -E "^ +[0-9]" | head -22 | sed 's/^/    /'
  echo "  --- $a: top inclusive (children) symbols in libggml-base"
  perf report -i /ai/bench/ovl18_$a.perf --children --sort symbol --stdio --dsos libggml-base.so 2>/dev/null | grep -E "^ +[0-9]" | head -12 | sed 's/^/    /'
done
rm -f /ai/bench/ovl18_*.perf.old
echo OVL18_DONE
