#!/bin/bash
# CTX1C: does MTP survive a slot restore when the head is IN the model? ctx1b showed the standalone -md head drafting blind after
# a restore (it is a separate model with its own, empty KV): 32k decode 17.2 t/s with it vs 28.6 without, acceptance 36% vs ~85%.
# The in-model head (V1c graft: blk.40 inside the main GGUF, --spec-type draft-mtp without -md) shares the target's state, which
# the slot file carries. Hypothesis: after a restore the in-model head keeps its normal acceptance (>= 0.75) and decode at 27k
# depth beats the no-MTP control => the long-context DECODE server runs MTP, and the in-model file becomes the long-context file.
# Kill: acceptance < 0.6 or decode <= control => long-context decode runs without speculation.
# Rows (32k, F16 KV): presave under the prefill config, then extend twice from the same slot: MTP on, MTP off (control).
# The spec flags are ON in the prefill server too, so whatever state the head needs is in the slot.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M-MTP.gguf
[ -e "$Q" ] || { echo "CTX1C_REFUSED: $Q missing (mtp1's grafted file; cold-storage symlink?)"; exit 1; }
SP="--spec-type draft-mtp --spec-draft-n-max 2"
export TIMEOUT=21600
export EXT=$'<|im_end|>\n<|im_start|>user\nNow list the three most important open risks from the text above, one line each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
SLOTS=/ai/bench/slots; mkdir -p $SLOTS
run() { # run MODE LABEL SLOTNAME CTX REPS PRE_GEN DROP [llama-server args, last one wins]  -- fresh server per row
  local mode=$1 label=$2 slot=$3 ctx=$4 reps=$5 pregen=$6 drop=$7; shift 7
  echo "##### $label mode=$mode ctx=$ctx reps=$reps pre_gen=$pregen drop=$drop | $*"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS --moe-expert-cache 24 -ub 512 -b 1024 -fa on "$@" \
    > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED: $(grep -iE 'error|failed|out of memory' server_$label.log | tail -1 | cut -c1-140)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  MODE=$mode REPS=$reps SLOT=$slot PRE_GEN=$pregen DROP=$drop python3 /ai/bench/slotclient.py "$label"
  local rc=$?
  [ "$mode" = extend ] && python3 -c "import json;print('    reply:', repr(json.load(open('/ai/bench/runs/$label.slot.json'))['rows'][0].get('reply')))" 2>/dev/null
  echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $pid; wait $pid 2>/dev/null
  grep -hE "forcing full prompt|KV buffer size|RS buffer size|compute buffer size|MoE expert cache enabled|out of memory|statistics +draft" server_$label.log | sort -u | tail -8 | cut -c1-200 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'
  [ $rc -eq 0 ] || echo "    $label: slotclient exit $rc"
  return 0
}
run presave ctx1c_c32k_pf_presave   ctx1c_c32k.slot 32768 13 64 0 --moe-expert-cache 0 -ub 2048 -b 2048 $SP
run extend  ctx1c_c32k_mtp_extend   ctx1c_c32k.slot 32768 13 64 0 --moe-expert-cache 16 -ub 128 -b 256 $SP
run extend  ctx1c_c32k_nomtp_extend ctx1c_c32k.slot 32768 13 64 0 --moe-expert-cache 16 -ub 128 -b 256
rm -f $SLOTS/ctx1c_c32k.slot
echo CTX1C_DONE
