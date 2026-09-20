#!/bin/bash
# KQ1H: the K2 expert file at the h2qual sample size. kq1's quality row is n=50/41/70 (SE 3-5 pts): it says "not visibly worse",
# it cannot rule out a 2-3 pt loss. h2qual measured the current default (IQ2_M) at n=200/280: GSM8K 96.0 +- 1.4, MMLU-Pro 71.4 +- 2.7.
# Same sets, same caps, same label scheme => one directly comparable row. Gate (KQ1): K2 within 1 sigma of those on both sets.
# Build = mainline (the default tree as of 2026-09-20, mainqual in band of the fork); the expert cache is lossless, so the slot
# count (26, as in kq1) does not enter the quality number.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
export BUILD=/ai/src/llama.cpp-mainline/build75 MMLU_CAP=2048 QARGS="--data data_h2 --sets gsm8k,mmlu_pro"
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive; K2=$M/$N-K2-expQ2K-downQ3K.gguf
[ -f "$K2" ] || { echo "KQ1H_REFUSED: $K2 missing"; exit 1; }
MODEL=$K2 ./qualbench.sh q36_k2_cache26 -ngl 999 -ot "exps=CPU" --moe-expert-cache 26
echo KQ1H_DONE
