#!/bin/bash
# H2 larger-n quality pass for the default-file decision (S1): GSM8K-200 + MMLU-Pro-280 (cap 2048) on the four candidates.
# data_h2 is a nested superset of data (first 50 / 70 items identical, verified), and the labels are the SAME as the
# n=50/70 rows, so qual.py reuses every finished answer and only runs the 150 + 210 new items per model. Same build and
# flags as the rows being extended (fork build75). HumanEval stays at 41. Fastest candidate first: it is the one the
# picker excludes today on GSM8K 96 vs 100 (1.4 sigma at n=50).
cd /ai/bench
export MMLU_CAP=2048 QARGS="--data data_h2 --sets gsm8k,mmlu_pro"
M=/ai/models; MC=/ai/src/llama.cpp-moecache/build75
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf BUILD=$MC ./qualbench.sh g4_q2kp_cache19 -ngl 999 -ot "exps=CPU" --moe-expert-cache 19 -ub 128 -b 256
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf BUILD=$MC ./qualbench.sh g4_q3km_cache15 -ngl 999 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
MODEL=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MC ./qualbench.sh q36_iq2m_cache48 -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf  BUILD=$MC ./qualbench.sh g4_iq3m_cache16 -ngl 999 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
echo H2QUAL_DONE
