#!/bin/bash
# HQ1: the HARD quality sets, thinking ON — the first measurement that can see a reasoning collapse. Scouts (2026-09-20) report
# ~2-bit post-training quants holding knowledge sets while losing 20-30 pts on AIME / LiveCodeBench; our gate (GSM8K 96-100%,
# HumanEval, MMLU-Pro, thinking off) is saturated and cannot see that. Sets (published items verbatim, bench/qual/data_hard):
# aime (AIME 2024 + 2025, 60), math_l5 (MATH-500 level 5, numeric golds, 40), humaneval_plus (EvalPlus tests, the same 41 tasks).
# usage: hq1.sh iq2m | k2      (one arm per queue job, ~12-20 h each at ~30 t/s: long chains of thought are the point)
# Read: absolute AIME % against the model card's full-precision score = is the served ~2.5-bit file broken on hard reasoning?
# (if yes: RAM upgrade + 4-bit experts outranks every speed lever); and IQ2_M vs K2 paired per question (bench/qual/paired.py).
# Also `truncated` + `looping`: a chain cut at the cap scores wrong. Run 1 (aime cap 24.5k, -c 32768) lost both of its first two
# items that way, all 24.5k tokens inside the reasoning block; run 2 lost a prealgebra item at 16k with a repetition score of
# 0.00. Both runs decoded GREEDILY, which is off the model card. qual.py now samples as the card says (seeded per item), caps
# are 32k (math_l5) / 41k (aime; the card allows 81k, F16 KV at -c 49152 does not), and each row keeps the reasoning tail + a
# repetition score, so a cut chain reads as "still working" or "degenerate loop".
# Set order: math_l5 and humaneval_plus first (short chains, a usable paired read within hours), aime last (up to ~25 min/item).
# Config: mainline, F16 KV (the KV precision is NOT under test here), -c 49152 (KV 960 MiB, ~3.1 of 3.6 GiB used), cache 24,
# -ub 128, no MTP (F16 KV + head = OOM).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
case "$1" in
  iq2m) F=$M/$N-IQ2_M.gguf; L=q36_iq2m_hard ;;
  k2)   F=$M/$N-K2-expQ2K-downQ3K.gguf; L=q36_k2_hard ;;
  *) echo "usage: hq1.sh iq2m|k2"; exit 2 ;;
esac
[ -f "$F" ] || { echo "HQ1_REFUSED: $F missing"; exit 1; }
[ -s /ai/bench/qual/data_hard/aime.jsonl ] && grep -q '"aime"' /ai/bench/qual/qual.py || { echo "HQ1_REFUSED: deploy qual.py + data_hard first"; exit 1; }
export BUILD=/ai/src/llama.cpp-mainline/build75 QARGS="--data data_hard --sets math_l5,humaneval_plus,aime --think"
MODEL=$F ./qualbench.sh $L -ngl 999 -ot "exps=CPU" --moe-expert-cache 24 -ub 128 -b 256 -c 49152
echo "HQ1_DONE $1"
