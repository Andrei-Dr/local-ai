#!/bin/bash
# Expert cache P1: mainline #27861 as-is on Qwen3.6 (separate gate/up, SILU). Same harness as moe1.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
run() { local l=$1; shift; echo "##### $l | $*"; MODEL=$Q OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "moe.cache|MoE expert cache" server_$l.log | tail -3 | cut -c1-200 | sed 's/^/    /'; }
run p1_q36_off    -ot "exps=CPU"
run p1_q36_c16    -ot "exps=CPU" --moe-expert-cache 16
run p1_q36_c32    -ot "exps=CPU" --moe-expert-cache 32
run p1_q36_c48    -ot "exps=CPU" --moe-expert-cache 48
run p1_q36_c32_i1 -ot "exps=CPU" --moe-expert-cache 32 --moe-expert-cache-inserts 1
echo MOE2_DONE
