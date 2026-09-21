#!/bin/bash
# LQ2: is the FA4 row-owning K walk (GGML_CUDA_FA_VEC_KROW=1, +17-22% decode at depth) NON-INFERIOR where it matters — retrieval at
# depth? fa4 showed its logprob drift equals that of the accepted upstream->FA1 change, and that text identity means nothing at
# 120k (any float reordering moves a 2.5-bit model's logprobs by ~0.13). So the proof has to be task-level: the lq1 instrument
# (RULER-style needles + variable tracking, one prefill per depth), q4_0 KV, FA build, KROW 0 vs 1. Prefill never touches the vec
# kernel, so the two arms differ ONLY in the decode kernel. Kill for KROW: needle pct or vt mean worse than the KROW=0 arm by more
# than 1 item in 10 at any depth. Pass => KROW becomes the code default at the next build.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
grep -q "reuse-guard" /ai/bench/qual/longctx.py || { echo "LQ2_REFUSED: deploy bench/qual/longctx.py (with --reuse-guard) first"; exit 1; }
M=/ai/models; BUILD=/ai/src/llama.cpp-fa1/build75
[ -x $BUILD/bin/llama-server ] || { echo "LQ2_REFUSED: FA build missing"; exit 1; }
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
export TIMEOUT=21600
row() { # row LABEL CTX UB KROW DEPTHS [kv args...]
  local label=$1 ctx=$2 ub=$3 norot=$4 depths=$5; shift 5
  echo "##### $label ctx=$ctx ub=$ub krow=$norot depths=$depths | $*"
  GGML_CUDA_FA_VEC_GQA=1 GGML_CUDA_FA_VEC_KROW=$norot GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 --moe-expert-cache 0 -ub $ub -b $ub -fa on "$@" \
    > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED: $(grep -iE 'error|failed|out of memory' server_$label.log | tail -1 | cut -c1-140)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  python3 /ai/bench/qual/longctx.py "$label" --depths $depths
  local rc=$?
  echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $pid; wait $pid 2>/dev/null
  grep -hE "attn_rot_k|KV buffer size|forcing full prompt|out of memory" server_$label.log | sort -u | tail -5 | cut -c1-170 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'
  [ $rc -eq 0 ] || echo "    $label: longctx exit $rc (rows kept in qual/results/$label.longctx.jsonl; a rerun resumes)"
  return 0
}
D131=4096,16384,32768,65536,131072
row lq2_q4_fa1  131072 512 0 $D131 -ctk q4_0 -ctv q4_0
row lq2_q4_krow 131072 512 1 $D131 -ctk q4_0 -ctv q4_0
echo "--- verdict input: qual/results/lq2_*.longctx.summary.json (needle pct + vt mean per depth, by position bucket, prefix re-use)"
echo LQ2_DONE
