#!/bin/bash
# PG2: pre-gating restricted to MISSES. pg1 (09:50): layer L's MoE input through layer L+1's router recalls 90.9% of L+1's top-8
# with a top-16 prediction. What a prefetch can gain is only the experts NOT already cached: per decode pass (union of its tokens)
# and layer: actual misses, the share the prediction covers, and the predicted-but-uncached experts = the uploads a prefetch issues
# (useful = covered misses / uploads). Build: ov with pregate-diag2.patch (supersedes pregate-diag.patch). Serving config, MTP n=2.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
SRC=/ai/src/llama.cpp-ov; OV=$SRC/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
P1=/ai/bench/patches/pregate-diag.patch; P2=/ai/bench/patches/pregate-diag2.patch
if git -C $SRC apply -R --check $P2 2>/dev/null; then echo "    diag2 already applied"; else
  if git -C $SRC apply -R --check $P1 2>/dev/null; then git -C $SRC apply -R $P1 && echo "    reverted diag1"; fi
  git -C $SRC apply $P2 && echo "    applied diag2" || { echo "PG2_FAILED: patch"; exit 1; }; fi
cmake --build $OV --target llama-server -j6 > pg2_build.log 2>&1 || { grep error pg2_build.log | head; echo "PG2_FAILED: build"; exit 1; }
strings $OV/bin/libllama.so* | grep -q "pregate-miss" || { echo "PG2_FAILED: diag2 not in libllama"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "##### pg2 | $(date +%T)"
env LLAMA_MOE_PREGATE_DIAG=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 pg2_diag $SERVE 2>&1 | grep -E "^pg2_diag" | cut -c1-110 | sed 's/^/    /'
grep -oE "pregate(-miss)?: .*" server_pg2_diag.log | sed 's/^/    /'
grep -oE "moe-cache: steps [0-9]+ \| hit rate [0-9.]+%[^|]*\| uploads [0-9]+ \([0-9.]+ MiB, [0-9.]+ MiB/step\)" server_pg2_diag.log | tail -1 | sed 's/^/    /'
unset LD_LIBRARY_PATH
echo PG2_DONE
