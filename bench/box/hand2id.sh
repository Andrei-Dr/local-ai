#!/bin/bash
# HAND2ID: the identity gate hand2 could not decide. hand2 measured +5.4% mean decode (every prompt above base by more than the
# base spread) but base diverged from ITSELF on 2 of 4 prompts: the MoE expert cache publishes uploads when the PCIe copy
# happens to finish, so which experts hit (GPU kernel) or miss (CPU kernel) depends on timing, and near-ties flip. Both trees
# now carry LLAMA_MOE_CACHE_SYNC=1 (uploads scheduled at step N publish at step N+1: hit/miss = f(token history)).
#   det = moe-cache + sync commit (the served code + an env switch that is OFF unless set);  ov = det + 0007 + 0008 + 0009.
# Read: det_a vs det_b must be IDENTICAL (else the floor is still noisy -> UNDECIDED); then det vs ov IDENTICAL on every prompt
# => the three levers are bit-identical => with hand2's +5.4% they are GO. Any det vs ov divergence with a clean floor => bisect.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
DET=/ai/src/llama.cpp-det/build75; OV=/ai/src/llama.cpp-ov/build75; PY=/ai/.venv/bin/python
for b in $DET $OV; do strings $b/bin/libllama.so.0 | grep -q LLAMA_MOE_CACHE_SYNC || { echo "HAND2ID_REFUSED: $b has no sync mode"; exit 1; }; done
ARGS="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 LLAMA_MOE_CACHE_SYNC=1
arm() { # label build   (both builds' binaries carry a RUNPATH to the served tree: LD_LIBRARY_PATH picks the right libs)
  local l=$1; export BUILD=$2 LD_LIBRARY_PATH=$2/bin
  echo "##### $l | $(ldd $BUILD/bin/llama-server | grep -oE "(libllama|libggml-cuda|libggml-base)\.so\.0 => [^ ]+" | tr "\n" " ")"
  ./specbench.sh 999 $l $ARGS 2>&1
  grep -hE "DETERMINISTIC publish|moe-cache: steps|out of memory" server_$l.log | tail -2 | cut -c1-200 | sed 's/^/    /'
}
arm hand2id_det_a $DET
arm hand2id_ov_a  $OV
arm hand2id_det_b $DET
arm hand2id_ov_b  $OV
unset LD_LIBRARY_PATH
echo "--- IDENTITY (deterministic cache publish, temp 0, byte-exact)"
echo "  floor det_a vs det_b:"; $PY textdiff.py runs/hand2id_det_a.client.json runs/hand2id_det_b.client.json | sed 's/^/    /'; floor=${PIPESTATUS[0]}
echo "  det_a vs ov_a:";        $PY textdiff.py runs/hand2id_det_a.client.json runs/hand2id_ov_a.client.json  | sed 's/^/    /'; id1=${PIPESTATUS[0]}
echo "  det_b vs ov_b:";        $PY textdiff.py runs/hand2id_det_b.client.json runs/hand2id_ov_b.client.json  | sed 's/^/    /'; id2=${PIPESTATUS[0]}
if [ $floor -ne 0 ]; then v=UNDECIDED; elif [ $id1 -eq 0 ] && [ $id2 -eq 0 ]; then v=IDENTICAL; else v=DIVERGES; fi
echo "HAND2ID identity $v"
echo HAND2ID_DONE
