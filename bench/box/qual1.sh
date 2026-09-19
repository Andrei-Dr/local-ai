#!/bin/bash
# Quality pass, one row per model file, each on its fastest measured config. Resumable (qual.py skips finished ids).
cd /ai/bench
until grep -q PROF1_DONE /ai/bench/prof1.log; do sleep 10; done
MODEL=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf ./qualbench.sh q36_iq2m -ngl 999 --n-cpu-moe 32 -ub 128 -b 256
MODEL=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf ./qualbench.sh g4_iq3m -ngl 999 --n-cpu-moe 26 -ub 128 -b 256
MODEL=/ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf OFFLOAD=2 ./qualbench.sh bonsai27_pq2 -ngl 99 \
  -ot "blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU" -ub 128 -b 256 --spec-type draft-mtp --spec-draft-n-max 2
echo QUAL1_DONE
