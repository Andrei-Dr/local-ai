#!/bin/bash
# Stock Qwen3.8-27B dense IQ3_M vs Bonsai-2-27B PQ2_0 (same base model, different quant/abliteration).
# Apples-to-apples: same harness, speed + quality. Dense => no expert cache; test the placements that worked for Bonsai.
cd /ai/bench
Q38=/ai/models/Qwen3.8-27B-i1-IQ3_M.gguf
FFN='blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU'
# speed: no-spec baseline (ngl fallback), then FFN-on-CPU streamed like Bonsai's best
MODEL=$Q38 OFFLOAD=32 ./specbench.sh 24 q38_ngl24 || \
MODEL=$Q38 OFFLOAD=32 ./specbench.sh 20 q38_ngl20 || \
MODEL=$Q38 OFFLOAD=32 ./specbench.sh 16 q38_ngl16
MODEL=$Q38 OFFLOAD=2 ./specbench.sh 99 q38_ffncpu_ub128 -ot "$FFN" -ub 128 -b 256
echo Q38_SPEED_DONE
# quality on whichever FFN-on-CPU config loaded (best dense config); dense IQ3_M has no MTP head
MODEL=$Q38 OFFLOAD=2 BUILD=/ai/src/llama.cpp/build75 ./qualbench.sh q38_iq3m -ngl 99 -ot "$FFN" -ub 128 -b 256
echo Q38_DONE
