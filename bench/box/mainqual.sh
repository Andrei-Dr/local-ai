#!/bin/bash
# G2 confirmation: quality on the MAINLINE build for the two A/B configs, to prove the +5-7% speedup does not
# move scores (the Gemma-code temp-0 divergence is a greedy tie-break from cross-tree numerics, not a regression).
# Same labels+_main suffix; mmlu at cap 2048 so these are clean and directly comparable to the fork quality rows.
cd /ai/bench
export MMLU_CAP=2048
MM=/ai/src/llama.cpp-mainline/build75
M=/ai/models
MODEL=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MM ./qualbench.sh q36_iq2m_cache48_main -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf BUILD=$MM ./qualbench.sh g4_q3km_cache15_main -ngl 999 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
echo MAINQUAL_DONE
