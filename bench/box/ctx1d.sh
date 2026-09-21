#!/bin/bash
# CTX1D: (A) the 262k DECODE number, from the kept slot; (B) the Pascal-path build as the PREFILL server.
# (A) ctx1b / fa1 lost every 262k decode row to a runtime OOM: model ~1430 + KV 1440 (q4_0) + RS 63 + expert cache 8 = 349 +
#     compute buffer 698 MiB EVEN AT ub 128. ~512 MiB of that buffer is the reserve for dequantizing the whole used KV to F16,
#     which the multi-token attention path (MMA_F16 / tile) needs for quantized KV. The GQA vec path (FA2, mode 2) handles <= 4
#     query tokens WITHOUT that copy, so a decode server with -ub 4 -b 4 should reserve almost nothing => 262k fits WITH a real
#     expert cache. FA2 lost on speed at pp3 (-8.5%) but may be what makes 262k decode fit at all. Suffix prefill at ub 4 is
#     slow (~30 t/s) — fine for a user turn, wrong for documents: those go through the prefill server (two-phase).
#     Rows: a) cache 0, ub 128, mode 1 = does plain 262k decode fit at all; b) mode 2 + ub 4 + cache 16; c) same, cache 20;
#     d) b under the Pascal-path build (tg at depth was +6% there). Read: compute buffer MiB, vram, decode t/s at ~240k, reply.
# (B) arch1: llama-bench pp512 = 336 t/s on the 61-virtual;80-virtual + FORCE_MMQ build vs 113 on arch 75 (tg64 -8%, pp3 -24%):
#     the arch-75 MMQ tiles are sized for tensor cores a GTX 16xx does not have. Verify on the REAL prefill server (cache 0):
#     32k presave at ub 2048 under both builds, plus ub 4096 on the Pascal build. Hypothesis: >= 2x prefill => the two-phase
#     design uses TWO BINARIES: Pascal build prefills, arch-75 + FA1 decodes. Slot must restore across builds (row e).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
W=/ai/src/llama.cpp-fa1; M=/ai/models; SLOTS=/ai/bench/slots
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -x $W/build75/bin/llama-server ] && [ -x $W/build6180/bin/llama-server ] || { echo "CTX1D_REFUSED: fa1 + arch1 builds needed"; exit 1; }
export TIMEOUT=21600 GEN=64
export EXT=$'<|im_end|>\n<|im_start|>user\nNow list the three most important open risks from the text above, one line each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
Q4="-ctk q4_0 -ctv q4_0"
run() { # run BUILD GQA MODE LABEL SLOTNAME CTX REPS [server args]
  local b=$1 gqa=$2 mode=$3 label=$4 slot=$5 ctx=$6 reps=$7; shift 7
  echo "##### $label build=$b gqa=$gqa mode=$mode ctx=$ctx | $*"
  GGML_CUDA_FA_VEC_GQA=$gqa GGML_OP_OFFLOAD_MIN_BATCH=32 $W/$b/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS -fa on "$@" > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED at load: $(grep -iE 'error|out of memory' server_$label.log | tail -1 | cut -c1-140)"; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  MODE=$mode REPS=$reps SLOT=$slot PRE_GEN=64 DROP=0 python3 /ai/bench/slotclient.py "$label"; local rc=$?
  [ "$mode" = extend ] && python3 -c "import json;print('    reply:', repr(json.load(open('/ai/bench/runs/$label.slot.json'))['rows'][0].get('reply')))" 2>/dev/null
  echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  grep -hE "compute buffer size|KV buffer size|MoE expert cache enabled|out of memory" server_$label.log | sort -u | tail -5 | cut -c1-170 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'
  [ $rc -eq 0 ] || echo "    $label: slotclient exit $rc"
  return 0
}
echo "########## A. 262k decode from the kept slot ##########"
if [ -s $SLOTS/ctx1b_c262k.slot ] && [ -s /ai/bench/runs/ctx1b_c262k.slot.ids.json ]; then
  S=ctx1b_c262k.slot
  run build75   1 extend ctx1d_c262k_c0_ub128      $S 262144 116 $Q4 --moe-expert-cache 0  -ub 128 -b 256
  run build75   2 extend ctx1d_c262k_c16_ub4       $S 262144 116 $Q4 --moe-expert-cache 16 -ub 4 -b 4
  run build75   2 extend ctx1d_c262k_c20_ub4       $S 262144 116 $Q4 --moe-expert-cache 20 -ub 4 -b 4
  run build6180 2 extend ctx1d_c262k_c16_ub4_pascal $S 262144 116 $Q4 --moe-expert-cache 16 -ub 4 -b 4
else echo "    SKIPPED: kept 262k slot missing"; fi
echo "########## B. prefill server: arch 75 vs the Pascal-path build (32k, cache 0) ##########"
run build75   1 presave ctx1d_c32k_pf_b75_ub2048    ctx1d_b75.slot 32768 13 --moe-expert-cache 0 -ub 2048 -b 2048
run build6180 1 presave ctx1d_c32k_pf_b6180_ub2048  ctx1d_p.slot   32768 13 --moe-expert-cache 0 -ub 2048 -b 2048
run build6180 1 presave ctx1d_c32k_pf_b6180_ub4096  ctx1d_p4.slot  32768 13 --moe-expert-cache 0 -ub 4096 -b 4096
echo "--- e) slot written by the Pascal build, restored by the arch-75 decode server"
run build75   1 extend  ctx1d_c32k_xbuild_extend    ctx1d_p.slot   32768 13 --moe-expert-cache 24 -ub 128 -b 256
rm -f $SLOTS/ctx1d_b75.slot $SLOTS/ctx1d_p.slot $SLOTS/ctx1d_p4.slot
echo CTX1D_DONE
