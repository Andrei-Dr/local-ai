#!/bin/bash
# Qwen3.8-35B-A3B-Distill IQ2_M: A3B distill of Qwen3.8-27B knowledge onto the Qwen3.6-A3B body (qwen35moe, has MTP head).
# Direct challenger to our Qwen3.6-A3B daily driver. Same cache + MTP stack, same harness -> speed AND quality.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
D=/ai/models/Qwen3.8-35B-A3B-Distill-IQ2_M.gguf
HEAD="-md $D --spec-type draft-mtp"   # MTP head is in-file; drafter = the model itself (mtp_num_hidden_layers=1)
run() { local l=$1; shift; echo "##### $l | $*"; MODEL=$D OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps|will use checkpoints|draft" server_$l.log | tail -3 | cut -c1-200 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
run distill_off        -ot "exps=CPU"
run distill_c48        -ot "exps=CPU" --moe-expert-cache 48
run distill_c30_mtp2   -ot "exps=CPU" --moe-expert-cache 30 -ub 128 -b 256 $HEAD --spec-draft-n-max 2
echo DISTILL_SPEED_DONE
# quality on the best config (matches how we scored Qwen3.6)
MODEL=$D ./qualbench.sh distill_iq2m -ngl 999 -ot "exps=CPU" --moe-expert-cache 48
echo DISTILL_DONE
