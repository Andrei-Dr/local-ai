#!/bin/bash
# DAVE6: can Dave's q38fn (Qwen4ExpForConditionalGeneration, native 262,144 positions) serve 1M tokens with YaRN, and does it
# still find a fact at depth? Andrei (2026-09-24): "go for 1M, let's check it out". Dave's live q38fn server (R9700s) is NOT
# touched: this runs a separate vLLM on OUR MI210 (renderD128) with the MI210 image, the same weights read-only.
# Why it can work: only 12 of 48 layers are full attention (2 KV heads x 256, fp8 KV = 12 KiB / token -> 12 GiB at 1M);
# the 36 linear-attention layers keep a fixed state. The limit is the RoPE range: YaRN factor 4 over 262,144.
# vLLM 0.28.1rc0+mi210.7 builds YaRN on the interleaved mrope (rotary_embedding/__init__.py:243-269, mrope.py:254-271).
# The model is 118 GB and the MI210 has 64 GB: experts beyond VRAM go to host RAM via --cpu-offload-gb (slow prefill,
# slower decode; this measures IF it works and HOW FAST it reads, not a serving config).
#
# PRE-REGISTERED (2026-09-24, before the run):
#   R0 START: the server comes up with --max-model-len 1010000 and the YaRN override (log shows rope_type yarn or the
#      scaled max length); a refusal / crash = FAIL, the finding is the error text.
#   R1 NEEDLE: a 6-digit code planted in a code haystack (llama.cpp sources) at depth 10% / 50% / 90% of the prompt, asked
#      for at the end, greedy. Sizes ~200k, ~500k, ~950k tokens. PASS per (size, depth) = the exact code in the answer.
#      YaRN WORKS here if every 200k and 500k cell passes; 950k is reported (the first place 4x scaling can fray).
#   R2 SHORT-CONTEXT SANITY (static YaRN scales every request): 3 short arithmetic / fact prompts, greedy, answers printed
#      for Andrei / Dave to judge; no rule (no same-hardware native-RoPE baseline in this run).
#   Measured: time to first token per size (prefill t/s), decode t/s of the 32-token answer, peak VRAM.
set -u
W=/mnt/llm-storage/localai; A=renderD128; PORT=8099; OUT=$W/bench/results/dave6; mkdir -p $OUT
VIMG=local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm
M=/models/q38fn-heretic2-mxfp4-fp8
log() { echo "$*" | tee -a $OUT/dave6.log; }
[ "$(cat /sys/class/drm/$A/device/mem_info_vram_used)" -lt 2000000000 ] || { log "DAVE6_FAILED: $A not idle"; exit 1; }
docker rm -f localai-1m >/dev/null 2>&1
OVR='{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true},"max_position_embeddings":1048576}}'
log "--- R0 START $(date +%T)"
docker run -d --name localai-1m --network host --ipc host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A \
  --group-add 44 --group-add 991 --security-opt seccomp=unconfined -e HSA_NO_SCRATCH_RECLAIM=1 \
  -v $W:/w -v $W/vcache:/cache -v /mnt/llm-storage:/models:ro --entrypoint /usr/local/bin/mi210-entrypoint $VIMG \
  serve $M --served-model-name m --port $PORT --host 127.0.0.1 --hf-overrides "$OVR" --max-model-len 1010000 \
  --kv-cache-dtype fp8 --cpu-offload-gb 66 --gpu-memory-utilization 0.93 --max-num-seqs 1 --max-num-batched-tokens 32768 \
  --enable-chunked-prefill --reasoning-parser qwen3 --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 >/dev/null
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break
  docker ps -q -f name=localai-1m | grep -q . || break; sleep 10; done
docker logs localai-1m > $OUT/server.log 2>&1
if ! curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then
  log "R0 FAIL: server not up"; grep -iE "error|exception|raise" $OUT/server.log | tail -8 | tee -a $OUT/dave6.log
  docker rm -f localai-1m >/dev/null; log DAVE6_DONE; exit 0; fi
log "R0 PASS: up at $(date +%T) | $(grep -ioE "max_model_len[^,]*|rope[^,]*yarn[^,]*" $OUT/server.log | head -3 | tr '\n' ' ')"
docker run --rm --network host -v $W:/w -v /mnt/llm-storage/localai/combo:/src:ro --entrypoint python3 $VIMG /w/bench/dave6_needle.py \
  --port $PORT --src /src --out /w/bench/results/dave6 2>&1 | grep -vE "^(INFO|WARNING)|Triton" | tee -a $OUT/dave6.log
docker logs localai-1m > $OUT/server.log 2>&1
docker rm -f localai-1m >/dev/null
log DAVE6_DONE
