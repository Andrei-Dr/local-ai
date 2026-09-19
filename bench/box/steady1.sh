#!/bin/bash
# Steady-state decode for the headline configs: 800-token generations, so the cold-cache start stops dominating the number
# (the 46.3 / 48.2 headline rows are 200-token runs).
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
export GEN=800
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps" server_$l.log | tail -2 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
run $Q  st_q36_c30_mtp2   -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
run $G2 st_g4q2k_c15_mtp2 -ot exps=CPU --moe-expert-cache 15 -ub 128 -b 256 -md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 2
echo STEADY1_DONE
