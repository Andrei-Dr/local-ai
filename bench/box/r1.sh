#!/bin/bash
# R1: cache-aware routing (--moe-expert-cache-bias B). Policy sims 2026-09-20: Belady-optimal hit rate is 75-79% vs our 60%,
# and neither usage history nor token identity reaches it => the remaining way to raise hits at fixed VRAM is to make the
# ROUTER prefer experts that are already on the device. In decode/verify batches a cached expert's selection score is
# prob x (1 + B); expert weights still use the true probs. NOT lossless: every B > 0 needs its own quality row, and temp-0
# output now depends on the cache state. Gate: >= +8% decode at GSM8K/HumanEval within 1 sigma of the B=0 row.
source /ai/bench/preflight.sh || exit 1
WANT=2582f5c
D=/ai/src/llama.cpp-mainline
cd $D || exit 1
git fetch -q /ai/bench/builds/moe-cache-$WANT.bundle moe-cache:moe-cache-mac-$WANT || { echo R1_FETCH_FAILED; exit 1; }
git checkout -q -B moe-cache moe-cache-mac-$WANT && [ "$(git rev-parse --short=7 HEAD)" = "$WANT" ] || { echo R1_CHECKOUT_FAILED; exit 1; }
cmake --build build75 -j6 --target llama-server llama-bench llama-quantize llama-imatrix > /ai/bench/build_r1.log 2>&1 || { grep -E "error" -A4 /ai/bench/build_r1.log | head -30; echo R1_BUILD_FAILED; exit 1; }
echo "built $(git rev-parse --short=12 HEAD) dirty=[$(git status --porcelain | head -3)]"
cd /ai/bench
export BUILD=$D/build75 EDIT=1 GEN=300
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf; G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
QA="-ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
GA="-ot exps=CPU --moe-expert-cache 15 -ub 128 -b 256 -md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 2"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "cache-aware|moe-cache: steps|statistics +draft|out of memory" server_$l.log | sort -u | tail -4 | cut -c1-220 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
for b in 0 0.25 0.5 1.0 2.0; do t=$(echo $b | tr -d .); run $Q r1_q36_b$t $QA --moe-expert-cache-bias $b; done
for b in 0 0.5 1.0; do t=$(echo $b | tr -d .); run $G2 r1_g4q2k_b$t $GA --moe-expert-cache-bias $b; done
# quality per bias (math + code; MMLU-Pro only for the value that survives), cache config without MTP like the reference rows
export QARGS="--sets gsm8k,humaneval"
for b in 0.25 0.5 1.0; do t=$(echo $b | tr -d .); MODEL=$Q ./qualbench.sh q36_iq2m_cache48_r1b$t -ngl 999 -ot "exps=CPU" --moe-expert-cache 48 --moe-expert-cache-bias $b; done
echo R1_DONE
