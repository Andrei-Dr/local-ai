#!/bin/bash
# LQ1: retrieval quality AT DEPTH (bench/qual/longctx.py: RULER-style multi-key needles + variable tracking, one prefill per depth,
# the questions ride the cached document prefix) for the KV precisions the long-context config can use. ctx1/ctx1b measure SPEED
# at depth; this measures whether the model still WORKS there. Accuracy-first rule (SPEC 8.x): q4_0 KV does not ship on a guess.
# Five arms. The mainline attn-rot (Hadamard rotation of Q/K/V before caching, PR #21038) is ON by default in our build, so every
# quantized arm already includes it; the last arm turns it off to price what the rotation buys on THIS model at depth:
#   f16 (reference, <= 64k: 131k F16 KV = 2.6 GiB does not fit) | q8_0 | q8_0 K + q5_1 V (K is the fragile side; community KLD
#   data puts this next to q8/q8) | q4_0 | q4_0 with LLAMA_ATTN_ROT_DISABLE=1
# Hypothesis: q8_0 and q8/q5_1 == f16 at every shared depth; q4_0 == f16 to 32k, first misses at 64k-131k in the late-position
# bucket; q4_0 without rotation visibly worse than q4_0. Kill for q4_0: needle pct < f16/q8 by more than 1 question in 10 at any
# depth => the 131k/262k config is q8_0 (or q8/q5_1) with fewer cache slots.
# Expert cache = 0: the cache is lossless (does not enter accuracy), prefill never uses it, and the VRAM goes to KV + ubatch.
# On the GDN hybrid the document prefix is re-used through the server's context checkpoints (~1 ubatch before the prompt end);
# longctx's reuse guard stops a deep depth whose 2nd request re-prefilled instead of burning 40 min per question.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
grep -q "reuse-guard" /ai/bench/qual/longctx.py || { echo "LQ1_REFUSED: deploy bench/qual/longctx.py (with --reuse-guard) first"; exit 1; }
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
export TIMEOUT=21600
row() { # row LABEL CTX UB ROT_DISABLE DEPTHS [kv args...]
  local label=$1 ctx=$2 ub=$3 norot=$4 depths=$5; shift 5
  echo "##### $label ctx=$ctx ub=$ub rot_disable=$norot depths=$depths | $*"
  LLAMA_ATTN_ROT_DISABLE=$norot GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 \
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
D64=4096,16384,32768,65536; D131=$D64,131072
row lq1_f16     65536  1024 0 $D64
row lq1_q8      131072 512  0 $D131 -ctk q8_0 -ctv q8_0
row lq1_q8q51   131072 512  0 $D131 -ctk q8_0 -ctv q5_1
row lq1_q4      131072 512  0 $D131 -ctk q4_0 -ctv q4_0
row lq1_q4norot 131072 512  1 $D131 -ctk q4_0 -ctv q4_0
echo "--- verdict input: qual/results/lq1_*.longctx.summary.json (needle pct + vt mean per depth, by position bucket, prefix re-use)"
echo LQ1_DONE
