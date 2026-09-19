#!/bin/bash
# Quality pass: each MoE file with the cache off (reference numerics) and on its fastest cached config
# (doubles as the correctness check of the expert cache: GPU-served experts must not move the scores).
cd /ai/bench
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
MC=/ai/src/llama.cpp-moecache/build75
MODEL=$Q ./qualbench.sh q36_iq2m -ngl 999 -ot "exps=CPU"
MODEL=$Q BUILD=$MC ./qualbench.sh q36_iq2m_cache48 -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$G BUILD=$MC ./qualbench.sh g4_iq3m_cache16 -ngl 999 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
MODEL=$G ./qualbench.sh g4_iq3m -ngl 999 -ot "exps=CPU"
echo QUAL2_DONE
