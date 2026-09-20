#!/bin/bash
# FA1: build + verify + measure the GQA-aware quantized decode attention kernel (bench/box/patches/fa1-vec-gqa.patch).
# Why: decode falls 28.6 t/s @27k -> 12.4 @120k. With a quantized KV cache and one query token our card takes the CUDA `vec`
# flash-attention kernel, which launches one block per QUERY head; Qwen3.6 has 16 query heads on 2 K/V heads, so every K/V row is
# walked and dequantized 8x per step. The patch adds a head-group column dimension (ncols2 = 8): one walk over a K/V head serves
# its 8 query heads. Same math per head => lossless. GGML_CUDA_FA_VEC_GQA=0 selects the old path in the SAME binary (clean A/B).
# Separate worktree + build dir: the benchmark tree (/ai/src/llama.cpp-mainline @ 2582f5c, build75) is not touched.
# Gates: (1) test-backend-ops FLASH_ATTN_EXT on CUDA0 passes; (2) temp-0 reply from the kept 131k slot identical old vs new;
# (3) decode t/s at 120k depth new > old. Kill: any test failure or a text mismatch => the patch is wrong, do not use the binary.
source /ai/bench/preflight.sh || exit 1
SRC=/ai/src/llama.cpp-mainline; W=/ai/src/llama.cpp-fa1; P=/ai/bench/fa1-vec-gqa.patch
[ -s $P ] || { echo "FA1_REFUSED: $P missing"; exit 1; }
if [ ! -d $W ]; then git -C $SRC worktree add -f $W 2582f5c > /dev/null 2>&1 || { echo FA1_WORKTREE_FAILED; exit 1; }; fi
cd $W || exit 1
git checkout -q -- ggml/src/ggml-cuda/fattn-vec.cuh
patch -s ggml/src/ggml-cuda/fattn-vec.cuh < $P || { echo FA1_PATCH_FAILED; exit 1; }
echo "patched: $(git diff --stat | tail -1)"
if [ ! -f build75/CMakeCache.txt ]; then
  EXTRA=$(grep -E "^(GGML_CUDA_[A-Z_]+|LLAMA_CURL|GGML_NATIVE|GGML_OPENMP):BOOL=" $SRC/build75/CMakeCache.txt | sed -E 's/^([A-Z_]+):BOOL=(.*)$/-D\1=\2/' | tr '\n' ' ')
  cmake -B build75 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=ON $EXTRA > /ai/bench/build_fa1.log 2>&1 || { tail -5 /ai/bench/build_fa1.log; echo FA1_CONFIGURE_FAILED; exit 1; }
fi
t0=$(date +%s)
cmake --build build75 -j6 --target llama-server test-backend-ops >> /ai/bench/build_fa1.log 2>&1 || { grep -E "error" -B2 -A6 /ai/bench/build_fa1.log | head -60; echo FA1_BUILD_FAILED; exit 1; }
echo "built in $(( $(date +%s) - t0 )) s"
echo "########## 1. test-backend-ops FLASH_ATTN_EXT (CUDA0 vs CPU) ##########"
grep -nE "nr23|for \(int nr|hsk.*256|hs : \{" tests/test-backend-ops.cpp | head -6 | cut -c1-160
build75/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > /ai/bench/fa1_ops.log 2>&1; rc=$?
echo "    exit $rc | OK $(grep -c 'OK' /ai/bench/fa1_ops.log) | FAIL $(grep -cE 'FAIL|ERR' /ai/bench/fa1_ops.log) | q4_0/q8_0 hs256 cases: $(grep -E 'hsk=256|hs=256' /ai/bench/fa1_ops.log | grep -cE 'q4_0|q8_0')"
grep -E "FAIL|ERR" /ai/bench/fa1_ops.log | head -8 | cut -c1-200; tail -2 /ai/bench/fa1_ops.log | cut -c1-160
[ $rc -eq 0 ] || { echo FA1_TEST_FAILED; exit 1; }
echo "########## 2+3. old vs new kernel, same binary, from the kept deep slots ##########"
cd /ai/bench; M=/ai/models; SLOTS=/ai/bench/slots; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
export TIMEOUT=3600 GEN=128
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
ab() { # ab TAG CTX CACHE
  local tag=$1 ctx=$2 cache=$3 g
  [ -s $SLOTS/ctx1b_$tag.slot ] && [ -s /ai/bench/runs/ctx1b_$tag.slot.ids.json ] || { echo "##### fa1_$tag SKIPPED: no kept slot"; return 0; }
  for g in 0 1; do
    echo "##### fa1_${tag}_gqa$g ctx=$ctx cache=$cache GGML_CUDA_FA_VEC_GQA=$g"
    GGML_CUDA_FA_VEC_GQA=$g GGML_OP_OFFLOAD_MIN_BATCH=32 $W/build75/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 --load-mode none --jinja \
      --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS --moe-expert-cache $cache -ub 128 -b 256 -fa on -ctk q4_0 -ctv q4_0 > server_fa1_${tag}_$g.log 2>&1 &
    local pid=$! i
    for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
    MODE=extend SLOT=ctx1b_$tag.slot DROP=0 python3 /ai/bench/slotclient.py fa1_${tag}_gqa$g
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
  done
  python3 - "$tag" <<'PY'
import json, sys
t = sys.argv[1]; r = [json.load(open(f"/ai/bench/runs/fa1_{t}_gqa{g}.slot.json"))["rows"][0] for g in (0, 1)]
same = r[0].get("reply") == r[1].get("reply")
print(f"    {t}: decode old {r[0]['decode_tps']:.2f} t/s -> new {r[1]['decode_tps']:.2f} t/s ({(r[1]['decode_tps']/r[0]['decode_tps']-1)*100:+.1f}%) | reply identical: {same}")
if not same: print("    OLD:", repr(r[0].get("reply"))); print("    NEW:", repr(r[1].get("reply")))
PY
}
ab c131k 131072 24
ab c262k 262144 8
echo FA1_DONE
