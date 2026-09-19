#!/bin/bash
# Expert cache P2 (fused gate_up + GELU + scales, gated admission): Gemma4 first, then Qwen3.6.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache|moe-cache:" server_$l.log | tail -2 | cut -c1-220 | sed 's/^.* I /    /'; }
echo "########## Gemma4 ##########"
run $G p2_g4_off       -ot "exps=CPU"
run $G p2_g4_c8        -ot "exps=CPU" --moe-expert-cache 8
run $G p2_g4_c12       -ot "exps=CPU" --moe-expert-cache 12
run $G p2_g4_c15       -ot "exps=CPU" --moe-expert-cache 15
run $G p2_g4_c15_a1    -ot "exps=CPU" --moe-expert-cache 15 --moe-expert-cache-admit 1
run $G p2_g4_c15_a2    -ot "exps=CPU" --moe-expert-cache 15 --moe-expert-cache-admit 2
run $G p2_g4_c16_ub128 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
echo "########## Qwen3.6 ##########"
run $Q p2_q36_c32      -ot "exps=CPU" --moe-expert-cache 32
run $Q p2_q36_c48      -ot "exps=CPU" --moe-expert-cache 48
run $Q p2_q36_c48_a2   -ot "exps=CPU" --moe-expert-cache 48 --moe-expert-cache-admit 2
run $Q p2_q36_c50_ub128 -ot "exps=CPU" --moe-expert-cache 50 -ub 128 -b 256
echo MOE3_DONE
