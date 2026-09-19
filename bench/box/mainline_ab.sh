#!/bin/bash
# Fork-vs-mainline A/B: same cache configs on both trees, to measure what the 431 mainline commits buy.
# Requires /ai/src/llama.cpp-mainline built at build75 (build_mainline.sh). Rows land in the ledger with each tree's commit.
cd /ai/bench
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G3=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf
run() { local b=$1 l=$2; shift 2; echo "##### $l | BUILD=$b | $*"; MODEL=$MODEL BUILD=$b OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1 | grep -E "tok \||telemetry|SERVER DIED"; }
for pair in "fork /ai/src/llama.cpp-moecache/build75" "main /ai/src/llama.cpp-mainline/build75"; do
  set -- $pair; tag=$1; b=$2
  MODEL=$Q  run $b ab_q36_c48_$tag   -ot "exps=CPU" --moe-expert-cache 48
  MODEL=$G3 run $b ab_g4q3k_c15_$tag -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
done
echo MAINLINE_AB_DONE
