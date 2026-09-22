#!/bin/bash
# MMQDP1: GGML_CUDA_MMQ_NO_MMA (MMQ = the dp4a kernels + tile configs on this tensor-core-less Turing) inside the prefill-mode
# build. arch1 measured the Pascal-path build at 3x the prefill (pp512 336 vs 113) with -8% decode — but that build also swapped
# MMVQ and FlashAttention; this switch touches MMQ only (batches > 8), so decode must stay put.
# Build: llama.cpp-ov (branch prefill-mux, 5179d52) with the switch set by a LOCAL define in mmq.cuh (experiment: rebuilds only
# the 25 objects that see mmq.cuh; the CMake option GGML_CUDA_MMQ_NO_MMA is the shipping form).
# Gates: 0 unit (test-backend-ops MUL_MAT + MUL_MAT_ID, CUDA vs CPU) | 1 long-prompt prefill at ubp 2048 / 4096 vs pmux3's MMA
# numbers (152.3 / 167.1 t/s, same prompt / config) | 2 decode 4-prompt mean vs the served build (base spread).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
SERVED=/ai/src/llama.cpp-mainline/build75; OV=/ai/src/llama.cpp-ov/build75
head -1 /ai/src/llama.cpp-ov/ggml/src/ggml-cuda/mmq.cuh | grep -q "define GGML_CUDA_MMQ_NO_MMA" || { echo "MMQDP1_REFUSED: ov tree is not the dp4a variant"; exit 1; }
echo "--- 0. UNIT (test-backend-ops, CUDA0 vs CPU)"
for op in MUL_MAT_ID MUL_MAT; do
  LD_LIBRARY_PATH=$OV/bin $OV/bin/test-backend-ops -o $op -b CUDA0 > mmqdp1_unit_$op.log 2>&1; rc=$?
  echo "    $op: $(grep -E "tests passed" mmqdp1_unit_$op.log | tail -1) rc=$rc $(grep -c FAIL mmqdp1_unit_$op.log) FAIL lines"
  [ $rc -eq 0 ] || { grep FAIL mmqdp1_unit_$op.log | head -5; echo "MMQDP1_FAILED: unit $op"; exit 1; }
done
echo "--- 1. LONG-PROMPT PREFILL (9,279 tok; -c 12288, cache 22, MTP head) — MMA reference from pmux3: ubp2048 152.3, ubp4096 167.1 t/s"
BASEARGS="-ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
for U in 2048 4096; do
  export LD_LIBRARY_PATH=$OV/bin
  GGML_OP_OFFLOAD_MIN_BATCH=32 $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja --parallel 1 \
    --port 8099 --cache-ram 0 $BASEARGS -ub 128 -b 4096 -ubp $U > server_mmqdp1_ubp$U.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  python3 longpf.py mmqdp1_ubp$U 40000 2>&1 | sed 's/^/    /'
  kill $NP; wait $NP 2>/dev/null
  grep -hE "decode mode: restored|could not re-allocate|out of memory" server_mmqdp1_ubp$U.log | tail -1 | cut -c1-140 | sed 's/^/    /'
done
unset LD_LIBRARY_PATH
echo "--- 2. DECODE (specbench, served build vs dp4a build; prompts <= 377 tok so prefill mode barely engages)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
ARGS="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128"
arm() { local l=$1 b=$2; shift 2; if [ $b = ov ]; then export BUILD=$OV LD_LIBRARY_PATH=$OV/bin; else export BUILD=$SERVED; unset LD_LIBRARY_PATH; fi
  echo "##### $l | $b"; ./specbench.sh 999 $l $ARGS "$@" 2>&1; }
arm mmqdp1_base_a served -b 256
arm mmqdp1_dp4a_a ov     -b 2048 -ubp 2048
arm mmqdp1_base_b served -b 256
arm mmqdp1_dp4a_b ov     -b 2048 -ubp 2048
unset LD_LIBRARY_PATH
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
ba, bb, da, db = (L(x) for x in ("mmqdp1_base_a", "mmqdp1_base_b", "mmqdp1_dp4a_a", "mmqdp1_dp4a_b"))
for k in ba:
    b, d = (ba[k]["decode_tps"] + bb[k]["decode_tps"]) / 2, (da[k]["decode_tps"] + db[k]["decode_tps"]) / 2
    pb, pd = (ba[k]["prefill_tps"] + bb[k]["prefill_tps"]) / 2, (da[k]["prefill_tps"] + db[k]["prefill_tps"]) / 2
    print(f"  {k:6s} decode base {b:6.2f} dp4a {d:6.2f} ({100 * (d / b - 1):+5.1f}%, base spread {abs(ba[k]['decode_tps'] - bb[k]['decode_tps']):.2f}) | prefill {pb:6.1f} -> {pd:6.1f} t/s")
PY
echo MMQDP1_DONE
