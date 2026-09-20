#!/bin/bash
# KQ1GRAFT: the K-quant expert file WITHOUT the -md MTP head (mtp1 finding applied to kq1's model).
# Hypothesis: grafting blk.40 into the K2 file returns the standalone head's ~727 MiB of VRAM; at 26 slots in-model
# must be within noise of -md, and 38 slots (the freed VRAM spent) must beat -md by the hit-rate curve (+2-4%/6 slots
# near saturation) => K2 + in-model head is the Qwen config to adopt.
# Kill: in-model_c26 within noise of -md_c26 AND c38 <+1% => the VRAM gift bought nothing on this quant, stay -md.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; OUT=$M/$N-K2-expQ2K-downQ3K-MTP.gguf
[ -s $K2 ] || { echo "KQ1GRAFT_SKIPPED: no $K2 yet (kq1.sh produces it)"; exit 0; }
if [ ! -s $OUT ]; then
  free=$(df -BG --output=avail /ai | tail -1 | tr -dc 0-9); [ "$free" -ge 20 ] || { echo "KQ1GRAFT_DISK_REFUSED: ${free}G free"; exit 1; }
  /ai/.venv/bin/python /ai/bench/gguf_graft_mtp.py $K2 $HEAD $OUT.part || { echo KQ1GRAFT_GRAFT_FAILED; rm -f $OUT.part; exit 1; }
  mv $OUT.part $OUT; ls -la $OUT | awk '{print "grafted file:", $5, "bytes"}'
fi
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=300
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "n_layer_nextn|MoE expert cache enabled|moe-cache: steps|creating MTP|statistics +draft|out of memory|CUDA0 model buffer" server_$l.log | sort -u | tail -8 | cut -c1-220 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
SP="--spec-type draft-mtp --spec-draft-n-max 2"
run $K2  kq1g_k2_md_c26       -ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD $SP
run $OUT kq1g_k2_inmodel_c26  -ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 $SP
run $OUT kq1g_k2_inmodel_c38  -ot exps=CPU --moe-expert-cache 38 -ub 128 -b 256 $SP
echo "--- temp-0 text identity, separate head vs in-model head at 26 slots (lossless speculation: divergences can only be numeric tie-breaks)"
python3 /ai/bench/textdiff.py runs/kq1g_k2_md_c26.client.json runs/kq1g_k2_inmodel_c26.client.json | sed 's/^/    /'
echo KQ1GRAFT_DONE
