#!/bin/bash
# Expert cache P2b (free slots fill ungated; gate guards evictions only) + first ledger-recorded runs.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps" server_$l.log | tail -2 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
run $G p2b_g4_c15        -ot "exps=CPU" --moe-expert-cache 15
run $G p2b_g4_c15_lv4    -ot "exps=CPU" --moe-expert-cache 15 -lv 4
run $G p2b_g4_c16_ub128  -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256 -lv 4
run $Q p2b_q36_c48       -ot "exps=CPU" --moe-expert-cache 48 -lv 4
run $Q p2b_q36_c48_a2    -ot "exps=CPU" --moe-expert-cache 48 --moe-expert-cache-admit 2 -lv 4
run $Q p2b_q36_c50_ub128 -ot "exps=CPU" --moe-expert-cache 50 -ub 128 -b 256 -lv 4
echo MOE4_DONE
