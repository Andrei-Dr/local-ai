#!/bin/bash
# P4c (prompt warm-up of the expert cache) + N2-lite (n_rs_seq covers short n-gram drafts) on the mainline tree.
# The tree is moved to the EXACT Mac commits through a git bundle (research/patches/bundles/), so ledger commit hashes
# match src/llama.cpp-mainline on the Mac. A/B = same binary, one flag: --moe-expert-cache-warm 0 vs 32, run ABAB for noise.
# Prompts: code, reason, edit (EDIT=1), long (LONG=1, W4). The first prompt of a run is the cold case, the rest are topic shifts.
source /ai/bench/preflight.sh || exit 1
WANT=498696c
D=/ai/src/llama.cpp-mainline
cd $D || exit 1
git fetch -q /ai/bench/builds/moe-cache-$WANT.bundle moe-cache:moe-cache-mac || { echo P4C1_FETCH_FAILED; exit 1; }
echo "old-vs-new base tree diff (must be empty): [$(git diff --stat cfb1ecdc7 0af8ea3 | tail -1)]"
git checkout -q -B moe-cache moe-cache-mac && [ "$(git rev-parse --short=7 HEAD)" = "$WANT" ] || { echo P4C1_CHECKOUT_FAILED; exit 1; }
cmake --build build75 -j6 --target llama-server llama-bench > /ai/bench/build_p4c1.log 2>&1 || { grep -E "error" -A4 /ai/bench/build_p4c1.log | head -30; echo P4C1_BUILD_FAILED; exit 1; }
echo "built $(git rev-parse --short=12 HEAD) dirty=[$(git status --porcelain | head -3)]"
cd /ai/bench
export BUILD=$D/build75 EDIT=1 LONG=1 GEN=300
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
GH="-md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 2"
QN="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 2"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "n_rs_seq +=|RS buffer size|MoE expert cache (enabled|prompt)|moe-cache: (steps|prompt)|draft acceptance|statistics +(ngram|draft)" server_$l.log | sort -u | tail -12 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
for i in a b; do
  run $Q  p4c_q36_warm0_$i    -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $QH --moe-expert-cache-warm 0
  run $Q  p4c_q36_warm32_$i   -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $QH --moe-expert-cache-warm 32
  run $G2 p4c_g4q2k_warm0_$i  -ot exps=CPU --moe-expert-cache 15 -ub 128 -b 256 $GH --moe-expert-cache-warm 0
  run $G2 p4c_g4q2k_warm32_$i -ot exps=CPU --moe-expert-cache 15 -ub 128 -b 256 $GH --moe-expert-cache-warm 32
done
# N2-lite: n-gram cap 3 with MTP at 2 => n_rs_seq 3 (+62.8 MiB => one slot less), vs the N1 winner (cap 2)
run $Q n2_q36_c30_ngmod2 -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $QN --spec-ngram-mod-n-max 2
run $Q n2_q36_c29_ngmod3 -ot exps=CPU --moe-expert-cache 29 -ub 128 -b 256 $QN --spec-ngram-mod-n-max 3
echo "--- temp-0 text identity, warm 0 vs warm 32 (a divergence is a greedy tie-break from different cache contents, not an error)"
for m in q36 g4q2k; do python3 /ai/bench/textdiff.py runs/p4c_${m}_warm0_a.client.json runs/p4c_${m}_warm32_a.client.json | sed "s/^/    $m /"; done
echo P4C1_DONE
