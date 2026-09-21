#!/bin/bash
# FA5PROBE: go / no-go for a block-sparse decode attention kernel, measured on REAL attention inputs before any kernel is written.
# After FA4 the attention kernel is still the largest item at depth and it is compute bound, so the remaining lever is to walk
# fewer KV tiles. A tile (128 positions) may be skipped only if an upper bound on q.k PROVES every weight in it is < thr of the
# maximum (accuracy first: bounded error, no heuristics). Unknowns this job measures on the kept 120k slot (a summarization
# prompt = diffuse attention = the hard case), per KV-head group of 8 query heads, at decode steps 8 / 32 / 64, all 10 layers:
#   oracle   share of tiles that carry < thr for ALL 8 heads (ceiling of any scheme; a single diffuse head kills its group);
#   bounds   share a cheap per-tile summary can PROVE: Quest min/max on the stored (Hadamard-rotated) channels, centroid+radius
#            ball (rotation invariant), min/max after un-rotating with a plain Hadamard.
# GO if a provable bound skips >= 50% of tiles at 1e-8 (or 1e-6) in most layers; else the idea is dead and we stop at FA4.
# Diagnostic build: fa-dump.patch is applied, used and REMOVED again in this job (two incremental builds, ~5 min total).
source /ai/bench/preflight.sh || exit 1
W=/ai/src/llama.cpp-fa1; PB=/ai/bench; B=$W/build75/bin; C=ggml/src/ggml-cuda; SLOTS=/ai/bench/slots; DUMP=/ai/bench/fadump
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -s $PB/fa-dump.patch ] && [ -s $PB/fa-gqa-dispatch.patch ] && [ -s $SLOTS/ctx1b_c131k.slot ] && [ -s $PB/fa_sparsity.py ] || { echo "FA5PROBE_REFUSED: patch, slot or fa_sparsity.py missing"; exit 1; }
cd $W || exit 1
build() { # $1 = dump | clean
  git checkout -q -- $C/fattn.cu && patch -s $C/fattn.cu < $PB/fa-gqa-dispatch.patch || { echo FA5PROBE_PATCH_FAILED; exit 1; }
  [ "$1" = clean ] || patch -s $C/fattn.cu < $PB/fa-dump.patch || { echo FA5PROBE_PATCH_FAILED; exit 1; }
  local t0=$(date +%s)
  cmake --build build75 -j6 --target llama-server > $PB/build_fa5probe_$1.log 2>&1 || { grep -E "error" -B2 -A8 $PB/build_fa5probe_$1.log | head -40; echo "FA5PROBE_BUILD_FAILED $1"; [ "$1" = clean ] || build clean; exit 1; }
  echo "built ($1) in $(( $(date +%s) - t0 )) s"
}
build dump
mkdir -p $DUMP; rm -f $DUMP/*.bin $DUMP/*.json
cd $PB
export TIMEOUT=3600 GEN=72
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_DUMP=$DUMP GGML_CUDA_FA_DUMP_STEPS=8,32,64 GGML_CUDA_FA_VEC_KROW=1 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  $B/llama-server -m $Q -ngl 999 -ot exps=CPU -c 131072 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 \
  --slot-save-path $SLOTS -fa on -ctk q4_0 -ctv q4_0 --moe-expert-cache 24 -ub 128 -b 256 > server_fa5probe.log 2>&1 &
NP=$!
for i in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || { echo "SERVER DIED: $(grep -iE 'error|out of memory' server_fa5probe.log | tail -1 | cut -c1-140)"; cd $W && build clean; echo FA5PROBE_FAILED; exit 1; }; sleep 2; done
MODE=extend SLOT=ctx1b_c131k.slot DROP=0 python3 /ai/bench/slotclient.py fa5probe
kill $NP; wait $NP 2>/dev/null; sleep 3
echo "dumped: $(ls $DUMP/*.json 2>/dev/null | wc -l) records, $(ls $DUMP/*.k.bin 2>/dev/null | wc -l) K tensors, $(du -sh $DUMP | cut -f1) | nodes: $(ls $DUMP/*.s8.json 2>/dev/null | xargs -n1 basename | sed 's/\.s8\.json//' | tr '\n' ' ' | cut -c1-200)"
/ai/.venv/bin/python /ai/bench/fa_sparsity.py $DUMP 2>&1 | cut -c1-400
cd $W && build clean
echo FA5PROBE_DONE
