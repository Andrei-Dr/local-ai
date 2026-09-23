#!/bin/bash
# PG1: pre-gating recall on Qwen3.6-35B-A3B K2. P2 (SPEC) killed TOKEN-based expert prediction (recall@30 68% vs LRU 57% on code,
# 49% vs 65% on prose); the literature's predictor is HIDDEN-STATE based: layer L+1's router applied to layer L's MoE input
# (Pre-gated MoE, HybriMoE arXiv 2504.05897, reported 90%+). If it holds here, predicted misses of layer L+1 can be uploaded over
# PCIe during layer L and computed on the idle GPU, taking bytes off the CPU/DDR4 critical path (research/design-harmony-ledger.md).
# Build: ov tree + patches/pregate-diag.patch (LLAMA_MOE_PREGATE_DIAG=1; inert when unset). Serving config, MTP n=2, decode only
# (batches <= 4). Read the last "pregate:" line: +1@8 / +1@16 / +2@16 recall, per layer band.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
SRC=/ai/src/llama.cpp-ov; OV=$SRC/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
P=/ai/bench/patches/pregate-diag.patch
if git -C $SRC apply -R --check $P 2>/dev/null; then echo "    already applied"; else git -C $SRC apply $P && echo "    applied" || { echo "PG1_FAILED: patch"; exit 1; }; fi
cmake --build $OV --target llama-server -j6 > pg1_build.log 2>&1 || { grep error pg1_build.log | head; echo "PG1_FAILED: build"; exit 1; }
strings $OV/bin/libllama.so* | grep -q LLAMA_MOE_PREGATE_DIAG || { echo "PG1_FAILED: diag not in libllama"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "##### pg1 | $(date +%T)"
env LLAMA_MOE_PREGATE_DIAG=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 pg1_diag $SERVE 2>&1 | grep -E "^pg1_diag" | cut -c1-120 | sed 's/^/    /'
grep -o "pregate: .*" server_pg1_diag.log | sed 's/^/    /'
unset LD_LIBRARY_PATH
echo PG1_DONE
