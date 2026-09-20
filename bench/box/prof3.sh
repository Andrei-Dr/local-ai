#!/bin/bash
# PROF3: what does a decode step cost AT DEPTH, by CUDA kernel? Decode falls 28.6 t/s @27k -> 12.4 @120k (q4_0 KV): ~54 ms of
# attention per token where the card's bandwidth bound is ~4 ms (0.7 GB of q4 KV at 192 GB/s). Reading fattn.cu: batch-1 decode
# with quantized KV uses the `vec` kernel, which has no GQA path, so each of the 16 query heads re-walks and re-dequantizes the
# K/V of its KV head (gqa 8 => 8x redundant); MTP verify batches (3 tokens) + quantized KV go to mma_f16, which dequantizes the
# used KV to F16 every step; and turing_mma_available() cannot tell a GTX 16xx (no tensor cores) from an RTX 20xx.
# This job restores the KEPT 131k slot (ctx1b) under the decode config and profiles 128 generated tokens with nsys.
# Read: share of flash_attn_ext_vec* (or mma/dequantize) in GPU time per token vs everything else => size of the prize for a
# GQA-aware quantized decode kernel. Second pass: same with the in-tree MTP verify path is NOT possible (the -md head is blind
# after a restore), so this is the no-MTP number.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75; SLOTS=/ai/bench/slots
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -s $SLOTS/ctx1b_c131k.slot ] && [ -s /ai/bench/runs/ctx1b_c131k.slot.ids.json ] || { echo "PROF3_REFUSED: kept 131k slot or its ids sidecar missing"; exit 1; }
export TIMEOUT=3600 GEN=128
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
rm -f /ai/bench/prof3_nsys.*
GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/prof3_nsys \
  $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c 131072 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 \
  -lv 4 --slot-save-path $SLOTS --moe-expert-cache 24 -ub 128 -b 256 -fa on -ctk q4_0 -ctv q4_0 > server_prof3.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || { echo "    server died: $(grep -iE 'error|out of memory' server_prof3.log | tail -1 | cut -c1-140)"; echo PROF3_FAILED; exit 1; }; sleep 2; done
MODE=extend SLOT=ctx1b_c131k.slot DROP=0 python3 /ai/bench/slotclient.py prof3_c131k_extend
pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
R=/ai/bench/prof3_nsys.nsys-rep
if [ -s "$R" ]; then
  for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum; do
    echo "--- $rep"; nsys stats --report $rep --format table "$R" 2>/dev/null | grep -vE "^$|Processing|Generating|SQLite" | head -18 | cut -c1-220
  done
else echo "    no nsys report: $(tail -3 server_prof3.log | cut -c1-160)"; fi
rm -f /ai/bench/prof3_nsys.sqlite
echo PROF3_DONE
