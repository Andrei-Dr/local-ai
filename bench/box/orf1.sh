#!/bin/bash
# ORF1: over-refusal compliance (published benign sets only: OR-Bench-Hard-1K + XSTest safe, 100 + 100, bench/qual/data_h2/overrefusal.jsonl)
# for the served "Uncensored" files, with the STOCK distill as the contrast row that validates the metric (a stock model should
# refuse a visible share). pct = compliance (higher = fewer over-refusals). Same labels as the quality rows => one ledger record more per label.
cd /ai/bench
export QARGS="--data data_h2 --sets overrefusal"
M=/ai/models; MC=/ai/src/llama.cpp-moecache/build75
MODEL=$M/Qwen3.8-35B-A3B-Distill-IQ2_M.gguf BUILD=$MC ./qualbench.sh distill_iq2m -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MC ./qualbench.sh q36_iq2m_cache48 -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf BUILD=$MC ./qualbench.sh g4_q2kp_cache19 -ngl 999 -ot "exps=CPU" --moe-expert-cache 19 -ub 128 -b 256
echo ORF1_DONE
