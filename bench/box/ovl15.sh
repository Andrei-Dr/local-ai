#!/bin/bash
# OVL15: where does the hoisted LAYOUT cost host time? ovl14 (11:42): plan-only (mode 2) costs as much as the full overlap
# (decode -6.8% / -6.2%, short-prompt prefill 59 -> 50 t/s) -> not the second stream; hit rates slightly HIGHER with the plan;
# main compute buffer 129 -> 290 MiB (reserve at bs=128). nsys CUDA-API trace (graphs ON, the served mode) of specbench's code prompt
# + 300-token decode, off vs plan, then hand1_phases.py: per-layer host timeline (D launch -> CPU experts -> H2D -> B launch + sync
# -> D2H) -> which phase grows. t2 combo tip.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL15_REFUSED: $T2 not on combo"; exit 1; }
echo "    t2 at $(git -C $T2 rev-parse --short HEAD)"
for a in off plan; do
  l=ovl15_nsys_$a; E=""; [ $a = plan ] && E="GGML_SCHED_MOE_PREFETCH=2"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
    nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 26 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  T0=$(date +%s.%N)
  curl -s localhost:8099/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."}],"temperature":0,"max_tokens":300,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(f\"    $a: prompt {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} t/s | decode {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} t/s\")"
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY hand1_phases.py $l.sqlite --skip-s 0 2>&1 | head -24 | sed 's/^/    /'
  rm -f $l.qdstrm
done
echo OVL15_DONE
