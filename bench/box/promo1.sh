#!/bin/bash
# PROMO1: build the promotion candidate and prove it before anything points at it.
# Branch tu116-served (research/patches/mainline-series 0001-0019) in a NEW worktree /ai/src/llama.cpp-v2, build75 with the
# served build's flags + -DGGML_CUDA_MMQ_NO_MMA=ON (the shipping form of the ov tree's local #define). build75 of the old served
# tree stays untouched: race1 is pinned to it and hq1 / lq1 / opt2 keep it so their rows stay comparable.
# Gates behind this (all passed on the ov build): pmux5 KLD (prefill mode, dp4a, FA tile non-inferior vs Q6), mmqdp1 / tu1 / tu2
# speed. 0018 (no copy sync) is OPT-IN (tu2: -2.0% +- noise); 0019 serves with GGML_CUDA_FA_TILE_MIN_BATCH=32.
# Steps: 0 worktree + build + flag checks | 1 unit (MUL_MAT_ID, MUL_MAT, FLASH_ATTN_EXT at TILE_MIN_BATCH=32) | 2 identity: v2 vs
# the ov build under LLAMA_MOE_CACHE_SYNC=1, serving config -> must be IDENTICAL (same code, CMake option vs #define) | 3 headline:
# old served config vs v2 serving config, 2 interleaved rounds + the 9,279-token prompt on each.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
REPO=/ai/src/llama.cpp-mainline; V2=/ai/src/llama.cpp-v2; BR=tu116-served; SHA=1c54372acfd8f38eba1fe5145572174104ed9510
OLD=$REPO/build75; NEW=$V2/build75; OV=/ai/src/llama.cpp-ov/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
echo "--- 0. WORKTREE + BUILD | $(date +%T)"
[ -d $V2 ] || git -C $REPO worktree add -q $V2 $BR || { echo "PROMO1_FAILED: worktree"; exit 1; }
[ "$(git -C $V2 rev-parse HEAD)" = "$SHA" ] || { echo "PROMO1_REFUSED: $V2 is at $(git -C $V2 rev-parse --short HEAD), expected ${SHA:0:9}"; exit 1; }
[ -z "$(git -C $V2 status --porcelain --untracked-files=no)" ] || { echo "PROMO1_REFUSED: $V2 has local changes"; exit 1; }
cmake -S $V2 -B $NEW -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_MMQ_NO_MMA=ON > promo1_cmake.log 2>&1 || { tail -20 promo1_cmake.log; echo "PROMO1_FAILED: cmake"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-perplexity llama-cli llama-bench test-backend-ops > promo1_build.log 2>&1 \
  || { grep -E "error" promo1_build.log | head -10; echo "PROMO1_FAILED: build"; exit 1; }
grep -q "^GGML_CUDA_MMQ_NO_MMA:BOOL=ON" $NEW/CMakeCache.txt || { echo "PROMO1_FAILED: MMQ_NO_MMA not ON"; exit 1; }
for s in GGML_CUDA_FA_TILE_MIN_BATCH GGML_CUDA_FA_NO_MMA GGML_CUDA_MMQ_MOE_EXPERT_COLS; do
  strings $NEW/bin/libggml-cuda.so* | grep -q $s || { echo "PROMO1_FAILED: libggml-cuda lacks $s"; exit 1; }; done
for s in GGML_SCHED_NO_COPY_SYNC GGML_SCHED_MOE_READBACK; do
  strings $NEW/bin/libggml-base.so* | grep -q $s || { echo "PROMO1_FAILED: libggml-base lacks $s"; exit 1; }; done
$NEW/bin/llama-server --help 2>/dev/null | grep -q -- "--ubatch-prefill" || { echo "PROMO1_FAILED: no --ubatch-prefill"; exit 1; }
readelf -d $NEW/bin/llama-server | grep -q "$NEW/bin" || { echo "PROMO1_FAILED: RUNPATH does not point at $NEW/bin"; exit 1; }
echo "    built $(git -C $V2 rev-parse --short HEAD) with MMQ_NO_MMA=ON | $(date +%T)"
echo "--- 1. UNIT (test-backend-ops CUDA0 vs CPU) | $(date +%T)"
for op in MUL_MAT_ID MUL_MAT FLASH_ATTN_EXT; do
  GGML_CUDA_FA_TILE_MIN_BATCH=32 $NEW/bin/test-backend-ops -o $op -b CUDA0 > promo1_unit_$op.log 2>&1; rc=$?
  echo "    $op: $(grep -E "tests passed" promo1_unit_$op.log | tail -1) rc=$rc"
  [ $rc -eq 0 ] || { grep FAIL promo1_unit_$op.log | head -5; echo "PROMO1_FAILED: unit $op"; exit 1; }
done
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
BASE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128"
SERVE="$BASE -b 2048 -ubp 2048"
arm() { # label build ld envs args...
  local l=$1 b=$2 ld=$3 envs=$4; shift 4; echo "##### $l | build $b | env: ${envs:-none}"
  if [ -n "$ld" ]; then env $envs BUILD=$b LD_LIBRARY_PATH=$ld ./specbench.sh 999 $l "$@" 2>&1; else env -u LD_LIBRARY_PATH $envs BUILD=$b ./specbench.sh 999 $l "$@" 2>&1; fi
}
echo "--- 2. IDENTITY v2 vs ov (LLAMA_MOE_CACHE_SYNC=1, serving config) | $(date +%T)"
arm promo1_id_ov  $OV  $OV/bin "LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32" $SERVE
arm promo1_id_new $NEW ""      "LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32" $SERVE
$PY textdiff.py runs/promo1_id_ov.client.json runs/promo1_id_new.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PROMO1_FAILED: v2 is not identical to the ov build it replaces"; exit 1; }
echo "--- 3. HEADLINE: old served config vs v2 serving config (2 interleaved rounds) | $(date +%T)"
for r in a b; do
  arm promo1_old_$r $OLD "" "" $BASE -b 256
  arm promo1_new_$r $NEW "" "GGML_CUDA_FA_TILE_MIN_BATCH=32" $SERVE
done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
o = [L("promo1_old_a"), L("promo1_old_b")]; n = [L("promo1_new_a"), L("promo1_new_b")]
avg = lambda a, k, f: sum(x[k][f] for x in a) / 2
for k in o[0]:
    print(f"  {k:6s} decode {avg(o, k, 'decode_tps'):6.2f} -> {avg(n, k, 'decode_tps'):6.2f} ({100 * (avg(n, k, 'decode_tps') / avg(o, k, 'decode_tps') - 1):+5.1f}%)"
          f" | prefill {avg(o, k, 'prefill_tps'):6.1f} -> {avg(n, k, 'prefill_tps'):6.1f} t/s ({avg(n, k, 'prefill_tps') / avg(o, k, 'prefill_tps'):4.2f}x)")
PY
for b in old new; do
  if [ $b = old ]; then BIN=$OLD/bin; ENVS=""; EX="-b 256"; else BIN=$NEW/bin; ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32"; EX="-b 4096 -ubp 2048"; fi
  env -u LD_LIBRARY_PATH $ENVS GGML_OP_OFFLOAD_MIN_BATCH=32 $BIN/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none \
    --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp \
    --spec-draft-n-max 2 -ub 128 $EX > server_promo1_pf_$b.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$b] "; python3 longpf.py promo1_pf_$b 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done
echo PROMO1_DONE
