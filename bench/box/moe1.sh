#!/bin/bash
# Queue step 2: MoE baselines on build75. One measurement at a time.
cd /ai/bench
export BUILD=/ai/src/llama.cpp/build75
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
GD=/ai/models/mtp-gemma-4-26B-A4B-it.gguf
run() { # run MODEL OFFLOAD LABEL args... ; returns 1 if no generation completed
  local m=$1 o=$2 l=$3; shift 3
  echo "##### $l | OFFLOAD=$o | $*"
  local out; out=$(MODEL=$m OFFLOAD=$o ./specbench.sh 999 "$l" "$@" 2>&1); echo "$out"
  grep -E "CUDA0 (model|compute)|CUDA_Host|CPU.*model buffer|KV buffer|RS buffer" server_$l.log | cut -c1-160 | sed 's/^/    /'
  grep -q "decode" <<<"$out"
}
echo "########## Qwen3.6-35B-A3B IQ2_M ##########"
run $Q 32 q36_expscpu -ot "exps=CPU"
for n in 36 34 32 30 28; do run $Q 32 q36_ncmoe$n --n-cpu-moe $n || break; done
for n in 34 32 30 28 26; do run $Q 32 q36_ub128_ncmoe$n --n-cpu-moe $n -ub 128 -b 256 || break; done
echo "########## Gemma4-26B-A4B IQ3_M ##########"
run $G 32 g4_expscpu -ot "exps=CPU"
for n in 28 27 26 25; do run $G 32 g4_ncmoe$n --n-cpu-moe $n || break; done
for n in 27 26 25 24; do run $G 32 g4_ub128_ncmoe$n --n-cpu-moe $n -ub 128 -b 256 || break; done
echo "########## Gemma4 + MTP drafter ##########"
for o in 32 2; do for d in 1 2 3; do
  run $G $o g4_mtp_n${d}_o$o -ot "exps=CPU" -md $GD --spec-type draft-mtp --spec-draft-n-max $d
done; done
echo MOE1_DONE
