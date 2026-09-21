#!/bin/bash
# PROF4: PROF3 redone so it can see decode. PROF3's kernel table held ONE decode step (10 x flash_attn_ext_vec, 4.69 ms per layer
# at 120k q4_0 = 47 of the 87 ms step): every later step ran inside a CUDA graph, and nsys 2022.4 does not trace kernels launched
# from a graph. It also ran the pre-FA1 build, and its report import failed (raw .qdstrm only).
# Here: GGML_CUDA_DISABLE_GRAPHS=1 (kernel durations stay true, launch gaps grow, so read GPU kernel time, not t/s), the FA1
# build, two arms GGML_CUDA_FA_VEC_GQA=0|1 on the same kept 131k slot, QdstrmImporter as the fallback, 2022.4 report names.
# Read: ms per layer of the vec kernel under each arm against the bandwidth floor (~0.4 ms: 69 MB of q4_0 K+V per layer at
# 192 GB/s) and the share of attention in GPU time per token => what a throughput-oriented quantized decode kernel can win.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; BUILD=/ai/src/llama.cpp-fa1/build75; SLOTS=/ai/bench/slots
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -x $BUILD/bin/llama-server ] || { echo "PROF4_REFUSED: FA1 build missing"; exit 1; }
[ -s $SLOTS/ctx1b_c131k.slot ] && [ -s /ai/bench/runs/ctx1b_c131k.slot.ids.json ] || { echo "PROF4_REFUSED: kept 131k slot or its ids sidecar missing"; exit 1; }
export TIMEOUT=3600 GEN=64
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
for G in 0 1; do
  T=prof4_gqa$G; echo "##### $T | GGML_CUDA_FA_VEC_GQA=$G, CUDA graphs off"
  rm -f /ai/bench/$T.qdstrm /ai/bench/$T.nsys-rep /ai/bench/$T.sqlite
  GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_VEC_GQA=$G GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$T \
    $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c 131072 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 \
    -lv 4 --slot-save-path $SLOTS --moe-expert-cache 24 -ub 128 -b 256 -fa on -ctk q4_0 -ctv q4_0 > server_$T.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || { echo "    server died: $(grep -iE 'error|out of memory' server_$T.log | tail -1 | cut -c1-140)"; echo PROF4_FAILED; exit 1; }; sleep 2; done
  MODE=extend SLOT=ctx1b_c131k.slot DROP=0 python3 /ai/bench/slotclient.py ${T}_extend
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  R=/ai/bench/$T.nsys-rep
  [ -s "$R" ] || { [ -s /ai/bench/$T.qdstrm ] && /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i /ai/bench/$T.qdstrm -o $R >/dev/null 2>&1; }
  if [ -s "$R" ]; then
    nsys stats -q --force-export=true -r gpukernsum,gpumemtimesum,cudaapisum -f csv -o /ai/bench/runs/$T "$R" >/dev/null 2>&1
    echo "--- gpu kernels: time%, total ns, instances, avg ns, name"
    cut -d, -f1,2,3,4,9- /ai/bench/runs/${T}_gpukernsum.csv | head -16 | cut -c1-200
  else echo "    no nsys report: $(tail -3 server_$T.log | cut -c1-160)"; fi
  rm -f /ai/bench/$T.sqlite /ai/bench/$T.qdstrm
done
echo PROF4_DONE
