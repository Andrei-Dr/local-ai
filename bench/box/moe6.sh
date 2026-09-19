#!/bin/bash
# MTP on top of the cache, re-fitted to VRAM (36 slots + Qwen head OOMed at n=2), and MTP on the K-quant Gemma files.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G3=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf
G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
QD="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp"
GD="-md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps" server_$l.log | tail -2 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
run $Q  p4_q36_c30_mtp2  -ot "exps=CPU" --moe-expert-cache 30 -ub 128 -b 256 $QD --spec-draft-n-max 2
run $Q  p4_q36_c28_mtp2  -ot "exps=CPU" --moe-expert-cache 28 -ub 128 -b 256 $QD --spec-draft-n-max 2
run $Q  p4_q36_c28_mtp3  -ot "exps=CPU" --moe-expert-cache 28 -ub 128 -b 256 $QD --spec-draft-n-max 3
run $G3 p4_g4q3k_c11_mtp1 -ot "exps=CPU" --moe-expert-cache 11 -ub 128 -b 256 $GD --spec-draft-n-max 1
run $G3 p4_g4q3k_c11_mtp2 -ot "exps=CPU" --moe-expert-cache 11 -ub 128 -b 256 $GD --spec-draft-n-max 2
run $G2 p4_g4q2k_c15_mtp1 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256 $GD --spec-draft-n-max 1
run $G2 p4_g4q2k_c15_mtp2 -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256 $GD --spec-draft-n-max 2
echo MOE6_DONE
