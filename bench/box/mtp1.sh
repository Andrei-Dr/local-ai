#!/bin/bash
# MTP1 (V1 + V1b without C++): graft the standalone MTP head (blk.40 of the same base model) into the Qwen3.6 IQ2_M file and run
# MTP against the TARGET model (--spec-type draft-mtp, no -md). Today the separate head costs 727.7 MiB of VRAM (second
# output.weight 272.8 + its layer's 256 experts 432 + small) and 272.8 MiB of host RAM (second token_embd). In-model: one
# output.weight, one token_embd, and blk.40's experts follow -ot exps=CPU and get their own cache slots (41 cached layers).
# Expected: ~700 MiB of VRAM back => ~16 more slots => hit rate 53% -> ~62%. Speculation stays lossless either way.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
IQ=$M/$N-IQ2_M.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; OUT=$M/$N-IQ2_M-MTP.gguf
if [ ! -s $OUT ]; then
  free=$(df -BG --output=avail /ai | tail -1 | tr -dc 0-9); [ "$free" -ge 20 ] || { echo "MTP1_DISK_REFUSED: ${free}G free"; exit 1; }
  /ai/.venv/bin/python /ai/bench/gguf_graft_mtp.py $IQ $HEAD $OUT.part || { echo MTP1_GRAFT_FAILED; rm -f $OUT.part; exit 1; }
  mv $OUT.part $OUT; ls -la $OUT | awk '{print "grafted file:", $5, "bytes"}'
fi
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=300
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "n_layer_nextn|MoE expert cache enabled|moe-cache: steps|creating MTP|statistics +draft|out of memory|CUDA0 model buffer" server_$l.log | sort -u | tail -8 | cut -c1-220 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
SP="--spec-type draft-mtp --spec-draft-n-max 2"
# reference on the same build: separate head (-md), 30 slots
run $IQ  mtp1_q36_md_c30       -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 -md $HEAD $SP
# in-model head, same slots: VRAM delta + does it run at all
run $OUT mtp1_q36_inmodel_c30  -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $SP
# spend the VRAM on slots
run $OUT mtp1_q36_inmodel_c42  -ot exps=CPU --moe-expert-cache 42 -ub 128 -b 256 $SP
run $OUT mtp1_q36_inmodel_c46  -ot exps=CPU --moe-expert-cache 46 -ub 128 -b 256 $SP
echo "--- temp-0 text identity, separate head vs in-model head (lossless speculation: divergences can only be numeric tie-breaks)"
python3 /ai/bench/textdiff.py runs/mtp1_q36_md_c30.client.json runs/mtp1_q36_inmodel_c30.client.json | sed 's/^/    /'
echo MTP1_DONE
