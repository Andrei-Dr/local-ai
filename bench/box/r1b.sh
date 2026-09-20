#!/bin/bash
# R1B: clean apples-to-apples for the --moe-expert-cache-bias keep/kill. The first r1 run had two gaps that make the
# gate ("+8% decode within 1 sigma of the B=0 row") uncomputable: (a) speed swept at --moe-expert-cache 30 but quality
# at 48 -> different hit rate, not comparable; (b) NO b0 quality row -> no B=0 baseline + sigma to test against.
# This run fixes both: Qwen3.6 only (the decision model), ONE cache value (48, the quality footing), bias 0..2.0,
# BOTH speed (specbench) and quality (qualbench gsm8k+humaneval) at that same cache 48, INCLUDING b0 for both.
# Result = for each B: decode t/s AND GSM8K/HumanEval pct, all at cache 48, with a real B=0 baseline. Gate unchanged:
# keep a bias only if decode >= +8% over b0 AND quality stays within 1 sigma of the b0 quality row.
source /ai/bench/preflight.sh || exit 1
WANT=2582f5c
D=/ai/src/llama.cpp-mainline
cd $D || exit 1
git fetch -q /ai/bench/builds/moe-cache-$WANT.bundle moe-cache:moe-cache-mac-$WANT || { echo R1B_FETCH_FAILED; exit 1; }
git checkout -q -B moe-cache moe-cache-mac-$WANT && [ "$(git rev-parse --short=7 HEAD)" = "$WANT" ] || { echo R1B_CHECKOUT_FAILED; exit 1; }
cmake --build build75 -j6 --target llama-server llama-bench llama-quantize llama-imatrix > /ai/bench/build_r1b.log 2>&1 || { grep -E "error" -A4 /ai/bench/build_r1b.log | head -30; echo R1B_BUILD_FAILED; exit 1; }
echo "built $(git rev-parse --short=12 HEAD)"
cd /ai/bench
export BUILD=$D/build75 EDIT=1 GEN=300
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
# speed footing = cache 48 (same as quality), MTP on like the reference decode rows
QA="-ot exps=CPU --moe-expert-cache 48 -ub 128 -b 256 -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "cache-aware|moe-cache: steps|statistics +draft|out of memory" server_$l.log | sort -u | tail -4 | cut -c1-220 | sed -E "s/^[0-9.]+ +[A-Z] +/    /"; }
# SPEED at cache 48, incl b0 baseline
for b in 0 0.25 0.5 1.0 2.0; do t=$(echo $b | tr -d .); run $Q r1b_q36_c48_b$t $QA --moe-expert-cache-bias $b; done
# QUALITY at cache 48, incl b0 baseline (this is the missing sigma reference)
export QARGS="--sets gsm8k,humaneval"
for b in 0 0.25 0.5 1.0 2.0; do t=$(echo $b | tr -d .); MODEL=$Q ./qualbench.sh q36_iq2m_c48_r1b_b$t -ngl 999 -ot "exps=CPU" --moe-expert-cache 48 --moe-expert-cache-bias $b; done
echo R1B_DONE
