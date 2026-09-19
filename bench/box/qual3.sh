#!/bin/bash
# Quality pass 3: new Gemma quants on their fastest cached config (also re-validates P3/P4 numerics), re-run of the
# answers that were cut off under the old token caps (same labels => only those rows run), then Bonsai (~3 h).
cd /ai/bench
M=/ai/models
MC=/ai/src/llama.cpp-moecache/build75
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf BUILD=$MC ./qualbench.sh g4_q3km_cache15 -ngl 999 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf BUILD=$MC ./qualbench.sh g4_q2kp_cache19 -ngl 999 -ot "exps=CPU" --moe-expert-cache 19 -ub 128 -b 256
MODEL=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf  BUILD=$MC ./qualbench.sh g4_iq3m_cache16 -ngl 999 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
MODEL=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf BUILD=$MC ./qualbench.sh q36_iq2m_cache48 -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
MODEL=$M/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf OFFLOAD=2 ./qualbench.sh bonsai27_pq2 -ngl 99 \
  -ot "blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU" -ub 128 -b 256 --spec-type draft-mtp --spec-draft-n-max 2
echo QUAL3_DONE
