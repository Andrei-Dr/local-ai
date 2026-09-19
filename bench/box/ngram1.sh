#!/bin/bash
# N-gram drafting stacked on the MTP drafter. --spec-type takes a comma list; common_speculative_draft() tries the
# implementations in order and the first one that returns a draft wins, so "ngram-mod,draft-mtp" = n-gram first,
# MTP fallback. EDIT=1 adds the copy-heavy prompt; the code/reason rows stay comparable with the moe6 rows.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
export EDIT=1 GEN=300
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
GH="-md $M/mtp-gemma-4-26B-A4B-it.gguf"
QC="-ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256"
GC="-ot exps=CPU --moe-expert-cache 15 -ub 128 -b 256"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps|draft acceptance|statistics +(ngram|draft)" server_$l.log | tail -5 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
# baselines with the edit prompt
run $Q  ng_q36_mtp2          $QC $QH --spec-type draft-mtp --spec-draft-n-max 2
# hybrid, draft kept inside the 4-token cache window
run $Q  ng_q36_ngmod3_mtp2   $QC $QH --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 3
# hybrid, long n-gram drafts (verify batch leaves the cache window: does a 16-token copy still pay?)
run $Q  ng_q36_ngmod16_mtp2  $QC $QH --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16
run $Q  ng_q36_ngmod16_m16   $QC $QH --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16 --spec-ngram-mod-n-match 16
# n-gram alone (no head in VRAM)
run $Q  ng_q36_ngmod16       $QC --spec-type ngram-mod --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16
# Gemma Q2_K_P pair
run $G2 ng_g4q2k_mtp2        $GC $GH --spec-type draft-mtp --spec-draft-n-max 2
run $G2 ng_g4q2k_ngmod16_mtp2 $GC $GH --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16
echo NGRAM1_DONE
