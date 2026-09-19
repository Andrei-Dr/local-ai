#!/bin/bash
# Re-run MMLU-Pro at cap 2048 for every candidate config that had answers cut off at 1024 (a cut-off scores
# wrong => the pct was a FLOOR). Uses the SAME labels as qual3.sh, so qual.py's resume logic reuses every
# finished answer and only re-runs the rows whose finish=="length" and tokens<2048. mmlu_pro only; no Bonsai (dropped).
cd /ai/bench
export MMLU_CAP=2048 QARGS="--sets mmlu_pro"
M=/ai/models; MC=/ai/src/llama.cpp-moecache/build75
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf BUILD=$MC ./qualbench.sh g4_q2kp_cache19 -ngl 999 -ot "exps=CPU" --moe-expert-cache 19 -ub 128 -b 256
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf BUILD=$MC ./qualbench.sh g4_q3km_cache15 -ngl 999 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf  BUILD=$MC ./qualbench.sh g4_iq3m_cache16 -ngl 999 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
MODEL=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MC ./qualbench.sh q36_iq2m_cache48 -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
echo MMLU2K_DONE
