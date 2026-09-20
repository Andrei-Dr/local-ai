#!/bin/bash
# FA1/FA2: GQA-aware quantized flash attention for few query tokens (bench/box/patches/fa-gqa-{vec,dispatch}.patch).
# Why: decode falls 28.6 t/s @27k -> 12.4 @120k. On cc 7.5 with a quantized KV cache: 1 query token => the CUDA `vec` kernel, one
# block per QUERY head, so with Qwen3.6's 16 query heads on 2 K/V heads every K/V row is walked + dequantized 8x per step; 2-4
# query tokens (MTP verify) => MMA_F16, which dequantizes the whole used KV cache to F16 on every step. The patch gives the vec
# kernel a head-group column dimension (ncols2 = 8: one walk over a K/V head serves its 8 query heads) and routes batches of up
# to 4 tokens to it. Same math per head => lossless. GGML_CUDA_FA_VEC_GQA = 0 old kernels | 1 decode only | 2 (default) + verify
# batches; GGML_CUDA_FA_VEC_GQA_NCOLS1=1 = one token per block for the multi-token case. One binary => clean A/B.
# usage: fa1.sh build   (CPU only: worktree, patches, unit-test cases, compile; safe next to a GPU job, run it right away)
#        fa1.sh verify  (GPU: unit test, old-vs-new text + decode t/s from the kept deep slots, llama-bench at depth, MTP check)
# Separate worktree + build dir: the benchmark tree (/ai/src/llama.cpp-mainline @ 2582f5c, build75) is never touched.
SRC=/ai/src/llama.cpp-mainline; W=/ai/src/llama.cpp-fa1; PB=/ai/bench
case "${1:-verify}" in
build)
  [ -s $PB/fa-gqa-vec.patch ] && [ -s $PB/fa-gqa-dispatch.patch ] || { echo "FA1_REFUSED: patches missing in $PB"; exit 1; }
  [ -d $W ] || git -C $SRC worktree add -f $W 2582f5c > /dev/null 2>&1 || { echo FA1_WORKTREE_FAILED; exit 1; }
  cd $W || exit 1
  git checkout -q -- ggml/src/ggml-cuda/fattn-vec.cuh ggml/src/ggml-cuda/fattn.cu tests/test-backend-ops.cpp
  patch -s ggml/src/ggml-cuda/fattn-vec.cuh < $PB/fa-gqa-vec.patch && patch -s ggml/src/ggml-cuda/fattn.cu < $PB/fa-gqa-dispatch.patch || { echo FA1_PATCH_FAILED; exit 1; }
  python3 - <<'PY' || { echo FA1_TESTPATCH_FAILED; exit 1; }
p = "tests/test-backend-ops.cpp"; s = open(p).read()
a = "    for (int hsk : { 40, 64, 72, 80, 96, 128, 192, 256, 320, 512, 576 }) {\n"
assert s.count(a) == 1
add = '''    // FA-GQA: grouped-query attention (ratio 8 and 16) on a quantized KV cache with 1-5 query tokens, head size 256:
    // the vec kernel's head-group columns (upstream covers quantized KV only for head sizes 64/72 and ratio 8 only for 192).
    for (int kv : { 512, 2048 }) {
        for (int nb : { 1, 2, 3, 4, 5 }) {
            for (bool sinks : { false, true }) {
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {8, 1}, kv, nb, true, sinks, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {8, 1}, kv, nb, true, sinks, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {8, 1}, kv, nb, true, sinks, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1));
            }
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 1, {16, 1}, kv, nb, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 3, {8, 2}, kv, nb, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
        }
    }
'''
open(p, "w").write(s.replace(a, add + a))
PY
  echo "patched: $(git diff --stat | tail -1)"
  if [ ! -f build75/CMakeCache.txt ]; then
    EXTRA=$(grep -E "^(GGML_CUDA_[A-Z_]+|LLAMA_CURL|GGML_NATIVE|GGML_OPENMP):BOOL=" $SRC/build75/CMakeCache.txt | sed -E 's/^([A-Z_]+):BOOL=(.*)$/-D\1=\2/' | tr '\n' ' ')
    cmake -B build75 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=ON $EXTRA > $PB/build_fa1.log 2>&1 || { tail -5 $PB/build_fa1.log; echo FA1_CONFIGURE_FAILED; exit 1; }
  fi
  t0=$(date +%s)
  nice -n 5 cmake --build build75 -j${J:-4} --target llama-server llama-bench test-backend-ops >> $PB/build_fa1.log 2>&1 || { grep -E "error" -B2 -A8 $PB/build_fa1.log | head -80; echo FA1_BUILD_FAILED; exit 1; }
  echo "built in $(( $(date +%s) - t0 )) s"; echo FA1_BUILD_DONE ;;
verify)
  source /ai/bench/preflight.sh || exit 1
  B=$W/build75/bin
  [ -x $B/test-backend-ops ] && [ -x $B/llama-server ] && [ $B/llama-server -nt $PB/fa-gqa-vec.patch ] || { echo "FA1_REFUSED: run 'fa1.sh build' first (binaries missing or older than the patch)"; exit 1; }
  echo "########## 1. test-backend-ops FLASH_ATTN_EXT (CUDA0 vs CPU), new GQA cases included ##########"
  for g in 2 0; do   # 2 = new kernels (the gate); 0 = old kernels on the same cases (sanity of the added cases themselves)
    GGML_CUDA_FA_VEC_GQA=$g $B/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > $PB/fa1_ops_g$g.log 2>&1; rc=$?
    echo "    GQA=$g: exit $rc | OK $(grep -c 'OK' $PB/fa1_ops_g$g.log) | FAIL $(grep -cE 'FAIL|ERR' $PB/fa1_ops_g$g.log) | hs256 quantized cases run: $(grep -E 'hsk=256' $PB/fa1_ops_g$g.log | grep -cE 'q4_0|q8_0')"
    grep -E "FAIL|ERR" $PB/fa1_ops_g$g.log | head -6 | cut -c1-220
    [ $g = 2 ] && [ $rc -ne 0 ] && { tail -3 $PB/fa1_ops_g$g.log | cut -c1-200; echo FA1_TEST_FAILED; exit 1; }
  done
  cd /ai/bench; M=/ai/models; SLOTS=/ai/bench/slots; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive; Q=$M/$N-IQ2_M.gguf
  echo "########## 2. decode from the kept deep slots: old vs new kernel (same math per head => the text must be identical) ##########"
  export TIMEOUT=3600 GEN=128
  export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  ab() { # ab TAG CTX CACHE
    local tag=$1 ctx=$2 cache=$3 g
    [ -s $SLOTS/ctx1b_$tag.slot ] && [ -s /ai/bench/runs/ctx1b_$tag.slot.ids.json ] || { echo "##### fa1_$tag SKIPPED: no kept slot"; return 0; }
    for g in 0 2; do
      echo "##### fa1_${tag}_gqa$g ctx=$ctx cache=$cache GGML_CUDA_FA_VEC_GQA=$g"
      GGML_CUDA_FA_VEC_GQA=$g GGML_OP_OFFLOAD_MIN_BATCH=32 $B/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 --load-mode none --jinja \
        --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS --moe-expert-cache $cache -ub 128 -b 256 -fa on -ctk q4_0 -ctv q4_0 > server_fa1_${tag}_$g.log 2>&1 &
      local pid=$! i
      for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
      MODE=extend SLOT=ctx1b_$tag.slot DROP=0 python3 /ai/bench/slotclient.py fa1_${tag}_gqa$g
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
    done
    python3 - "$tag" <<'PY'
import json, sys
t = sys.argv[1]
try:
    r = [json.load(open(f"/ai/bench/runs/fa1_{t}_gqa{g}.slot.json"))["rows"][0] for g in (0, 2)]
    same = r[0].get("reply") == r[1].get("reply")
    print(f"    {t}: decode old {r[0]['decode_tps']:.2f} -> new {r[1]['decode_tps']:.2f} t/s ({(r[1]['decode_tps']/r[0]['decode_tps']-1)*100:+.1f}%) | reply identical: {same}")
    if not same: print("    OLD:", repr(r[0].get("reply"))); print("    NEW:", repr(r[1].get("reply")))
except Exception as e: print(f"    {t}: no comparison ({e})")
PY
  }
  ab c131k 131072 24
  ab c262k 262144 8
  echo "########## 3. llama-bench at depth 32768, q4_0 KV: pp3 = a 3-token verify batch, tg32 = decode ##########"
  LB="$B/llama-bench -m $Q -ngl 999 -ot exps=CPU -fa 1 -ctk q4_0 -ctv q4_0 -t 6 -ub 2048 -b 2048 -d 32768 -r 1 --load-mode none"
  row() { echo "--- $1"; shift; env GGML_OP_OFFLOAD_MIN_BATCH=32 "$@" 2>&1 | grep -E "pp3|tg32|error|failed" | cut -c1-200; }
  row "old kernels (GQA=0)"                 GGML_CUDA_FA_VEC_GQA=0 $LB -p 3 -n 32
  row "new kernels (GQA=2, 2 tokens/block)" GGML_CUDA_FA_VEC_GQA=2 $LB -p 3 -n 32
  row "new kernels (GQA=2, 1 token/block)"  GGML_CUDA_FA_VEC_GQA=2 GGML_CUDA_FA_VEC_GQA_NCOLS1=1 $LB -p 3 -n 0
  echo "########## 4. real MTP verify path (in-model head, q4_0 KV, temp 0): old vs new ##########"
  QM=$M/$N-IQ2_M-MTP.gguf
  if [ -e "$QM" ]; then
    for g in 0 2; do
      GGML_CUDA_FA_VEC_GQA=$g GGML_OP_OFFLOAD_MIN_BATCH=32 $B/llama-server -m $QM -ngl 999 -ot exps=CPU -c 8192 -t 6 --load-mode none --jinja --parallel 1 \
        --port 8099 --cache-ram 0 -lv 4 --moe-expert-cache 24 -ub 128 -b 256 -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 > server_fa1_mtp_$g.log 2>&1 &
      pid=$!; for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
      python3 - $g <<'PY'
import json, sys, urllib.request
doc = open("/ai/bench/w4_doc.txt", encoding="utf-8").read()
body = {"messages": [{"role": "user", "content": doc + "\n\nList the five most important facts in the text above."}], "temperature": 0, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False}}
d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
t = d.get("timings", {}); txt = d["choices"][0]["message"]["content"]
json.dump({"text": txt}, open(f"/ai/bench/runs/fa1_mtp_g{sys.argv[1]}.json", "w"))
print(f"    GQA={sys.argv[1]}: decode {t.get('predicted_per_second', 0):.2f} t/s | draft n {t.get('draft_n')} accepted {t.get('draft_n_accepted')} | {txt[:90]!r}")
PY
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
    done
    python3 - <<'PY'
import json
a, b = (json.load(open(f"/ai/bench/runs/fa1_mtp_g{g}.json"))["text"] for g in (0, 2))
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"    MTP text identical: {a == b} (common prefix {n} of {len(a)} chars; MMA-on-F16 vs vec-on-q4 are different arithmetic, a late greedy tie flip is not a bug, garbage is)")
PY
  else echo "    SKIPPED: $QM missing"; fi
  echo FA1_DONE ;;
*) echo "usage: fa1.sh build|verify"; exit 2 ;;
esac
