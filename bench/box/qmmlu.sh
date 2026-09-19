#!/bin/bash
# Clean Qwen3.6-35B-A3B MMLU-Pro at cap 2048: the cache-48 run scored 62.9 with 19/70 answers cut off at 1024 (thinking
# off => Qwen rambles), so that was a FLOOR, not a score. New label q36_iq2m_mmlu2k, fresh (no stale rows), mmlu_pro only.
cd /ai/bench
export MMLU_CAP=2048
MC=/ai/src/llama.cpp-moecache/build75
export QARGS="--sets mmlu_pro"
MODEL=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MC ./qualbench.sh q36_iq2m_mmlu2k -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
echo QMMLU_DONE
