#!/bin/bash
# TU1: three switches for this tensor-core-less Turing (branch tu116-kernels on prefill-mux; research/patches/tu116/0011-0013),
# A/B'd by env var inside ONE build (llama.cpp-ov, dp4a MMQ variant):
#   GGML_CUDA_FA_NO_MMA=1           FlashAttention never selects the tensor-core MMA kernel. fattn.cu picks MMA on any cc >= 7.5
#                                   (turing_mma_available) — TU116 reports 7.5 and has no tensor cores; tile/vector are its native kernels.
#   GGML_CUDA_MMQ_MOE_EXPERT_COLS=1 MoE MMQ tile width sized against tokens per expert instead of the ubatch. Mainline does this on
#                                   RDNA3/4 only (mmq.cu ncols_opt); on NVIDIA an expert with ~4 tokens (ub 128, 8 of 256) gets a
#                                   128-column tile. At ub 4096 (~128 per expert) both pick the same tile: this is an ub 128-2048 lever.
#   GGML_SCHED_MOE_READBACK=1       the OLD behavior. New default: when a ubatch routes >= 8 tokens per expert on average (ub >= 256
#                                   here) the sched uploads the whole expert tensor with no router readback and no host sync. Exact by
#                                   construction (unused experts are never read).
#   GGML_SCHED_SYNC_BEFORE_COPY=1   the OLD behavior (0014). With one GPU the sched has no events, so before EVERY input copy into a
#                                   split it ran ggml_backend_synchronize(GPU): host blocks until the device drains, then enqueues the
#                                   copy + split while the device idles — once per layer in decode (CPU expert outputs -> GPU). New
#                                   default: a host-buffer copy enqueued on the GPU's own stream is already ordered, no host wait. Exact.
# (0014 is in research/patches/tu116 too.)
# Gates: 0 build + unit (test-backend-ops FLASH_ATTN_EXT under FA_NO_MMA, MUL_MAT_ID under EXPERT_COLS; CUDA vs CPU)
#        1 long prefill (9,279 tok, cache 22, MTP head): ubp 4096 readback | full upload | full upload + FA tile; ub 128 without
#          prefill mode: expert cols off | on
#        2 decode (MTP verify = 3-token FA batches): FA MMA vs FA tile, interleaved; must stay within the base spread
#        3 identity of the full upload under LLAMA_MOE_CACHE_SYNC=1 (readback vs full upload, prompts engage prefill mode)
# Accuracy of FA tile / expert cols (float order changes): the KLD arms in pmux5.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
SRC=/ai/src/llama.cpp-ov; OV=$SRC/build75
head -1 $SRC/ggml/src/ggml-cuda/mmq.cuh | grep -q "define GGML_CUDA_MMQ_NO_MMA" || { echo "TU1_REFUSED: ov tree is not the dp4a variant"; exit 1; }
echo "--- 0. BUILD (apply research/patches/tu116 to the ov tree; idempotent) | $(date +%T)"
for p in /ai/bench/patches/tu116/*.patch; do
  if git -C $SRC apply -R --check "$p" 2>/dev/null; then echo "    already applied: $(basename $p)"
  else git -C $SRC apply "$p" && echo "    applied: $(basename $p)" || { echo "TU1_FAILED: patch $(basename $p) does not apply"; exit 1; }; fi
done
cmake --build $OV --target llama-server llama-perplexity test-backend-ops -j6 > tu1_build.log 2>&1 || { tail -20 tu1_build.log; echo "TU1_FAILED: build"; exit 1; }
strings $OV/bin/libggml-cuda.so* | grep -q GGML_CUDA_FA_NO_MMA || { echo "TU1_FAILED: libggml-cuda lacks the FA switch"; exit 1; }
strings $OV/bin/libggml-base.so* | grep -q GGML_SCHED_SYNC_BEFORE_COPY || { echo "TU1_FAILED: libggml-base lacks 0014"; exit 1; }
strings $OV/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_READBACK || { echo "TU1_FAILED: libggml-base lacks the full-upload path"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin
unit() { # label env op
  env $2 $OV/bin/test-backend-ops -o $3 -b CUDA0 > tu1_unit_$1.log 2>&1; local rc=$?
  echo "    $1 ($2 $3): $(grep -E "tests passed" tu1_unit_$1.log | tail -1) rc=$rc"
  [ $rc -eq 0 ] || { grep FAIL tu1_unit_$1.log | head -5; echo "TU1_FAILED: unit $1"; exit 1; }
}
unit fa_tile  GGML_CUDA_FA_NO_MMA=1           FLASH_ATTN_EXT
unit expcols  GGML_CUDA_MMQ_MOE_EXPERT_COLS=1 MUL_MAT_ID
echo "--- 1. LONG-PROMPT PREFILL (9,279 tok; -c 12288, cache 22, MTP head) | $(date +%T)"
BASEARGS="-ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
pf() { # label "ENV=.." extra-args...
  local l=$1 envs=$2; shift 2
  env $envs GGML_OP_OFFLOAD_MIN_BATCH=32 $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 $BASEARGS "$@" > server_$l.log 2>&1 &
  local NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [${envs:-default}] "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
  grep -hE "could not re-allocate|out of memory|failed to allocate" server_$l.log | tail -1 | cut -c1-140 | sed 's/^/    /'
}
pf tu1_ubp4096_old   "GGML_SCHED_MOE_READBACK=1 GGML_SCHED_SYNC_BEFORE_COPY=1" -ub 128 -b 4096 -ubp 4096
pf tu1_ubp4096_rb    "GGML_SCHED_MOE_READBACK=1"                     -ub 128 -b 4096 -ubp 4096
pf tu1_ubp4096_full  ""                                              -ub 128 -b 4096 -ubp 4096
pf tu1_ubp4096_fa    "GGML_CUDA_FA_NO_MMA=1"                         -ub 128 -b 4096 -ubp 4096
pf tu1_ub128         ""                                              -ub 128 -b 4096
pf tu1_ub128_expcols "GGML_CUDA_MMQ_MOE_EXPERT_COLS=1"               -ub 128 -b 4096
pf tu1_ubp4096_all   "GGML_CUDA_FA_NO_MMA=1 GGML_CUDA_MMQ_MOE_EXPERT_COLS=1" -ub 128 -b 4096 -ubp 4096
echo "--- 2. DECODE (specbench, ov build, prefill mode on; FA MMA vs FA tile interleaved) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
ARGS="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; shift 2; echo "##### $l | env: ${envs:-none}"; env $envs ./specbench.sh 999 $l $ARGS "$@" 2>&1; }
arm tu1_mma_a  ""
arm tu1_tile_a "GGML_CUDA_FA_NO_MMA=1"
arm tu1_sync_a "GGML_SCHED_SYNC_BEFORE_COPY=1"
arm tu1_mma_b  ""
arm tu1_tile_b "GGML_CUDA_FA_NO_MMA=1"
arm tu1_sync_b "GGML_SCHED_SYNC_BEFORE_COPY=1"
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
ma, mb, ta, tb, sa, sb = (L(x) for x in ("tu1_mma_a", "tu1_mma_b", "tu1_tile_a", "tu1_tile_b", "tu1_sync_a", "tu1_sync_b"))
bm = bt = bs = 0
for k in ma:
    m, t = (ma[k]["decode_tps"] + mb[k]["decode_tps"]) / 2, (ta[k]["decode_tps"] + tb[k]["decode_tps"]) / 2
    sy = (sa[k]["decode_tps"] + sb[k]["decode_tps"]) / 2
    bm += m; bt += t; bs += sy
    print(f"  {k:6s} decode OLD sync-before-copy {sy:6.2f} -> new {m:6.2f} ({100 * (m / sy - 1):+5.1f}%, sync spread {abs(sa[k]['decode_tps'] - sb[k]['decode_tps']):.2f})")
    print(f"  {k:6s} decode FA mma {m:6.2f} tile {t:6.2f} ({100 * (t / m - 1):+5.1f}%, mma spread {abs(ma[k]['decode_tps'] - mb[k]['decode_tps']):.2f})"
          f" | prefill {(ma[k]['prefill_tps'] + mb[k]['prefill_tps']) / 2:6.1f} -> {(ta[k]['prefill_tps'] + tb[k]['prefill_tps']) / 2:6.1f} t/s")
print(f"  MEAN   decode sync-before-copy {bs / len(ma):6.2f} -> no host sync {bm / len(ma):6.2f} ({100 * (bm / bs - 1):+5.1f}%)")
print(f"  MEAN   decode FA mma {bm / len(ma):6.2f} tile {bt / len(ma):6.2f} ({100 * (bt / bm - 1):+5.1f}%)")
PY
echo "--- 3. IDENTITY of 0013 + 0014 (LLAMA_MOE_CACHE_SYNC=1; old sched path vs new default) | $(date +%T)"
LLAMA_MOE_CACHE_SYNC=1 arm tu1_id_rb   "GGML_SCHED_MOE_READBACK=1 GGML_SCHED_SYNC_BEFORE_COPY=1"
LLAMA_MOE_CACHE_SYNC=1 arm tu1_id_full ""
$PY textdiff.py runs/tu1_id_rb.client.json runs/tu1_id_full.client.json | sed 's/^/    /'
unset LD_LIBRARY_PATH
echo TU1_DONE
