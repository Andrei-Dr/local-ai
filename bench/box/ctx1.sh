#!/bin/bash
# CTX1: long-context on the 4 GB card, as a STAGGERED DEPTH LADDER: 16k -> 32k -> 64k -> 131k -> 262k. Every decode number in
# the ledger before this job was taken at -c 4096 with <= ~1.9k tokens in the KV; nothing is known about depth. Qwen3.6 is
# natively 262144-trained (n_ctx_train 262144, freq_base 10M, no rope stretch), so depth is a VRAM/time problem, not a quality one.
# Per rung, a save row (cold prefill + decode 64 + slot save to NVMe) then, after a server RESTART, a restore row (slot restore +
# the same request). Read per rung: prefill t/s, decode t/s at depth, KV/RS buffer sizes, slot bytes + save/restore ms.
# Hypothesis: restore turns minutes-hours of prefill into seconds (prompt_n of the request ~= 0 on a working restore), and decode
# at depth stays usable. Kill: restore does NOT skip the prompt on the hybrid (GDN) model => C1 dead for this arch.
# VRAM is the budget, so the ladder trades KV precision and cache slots for depth (KV = 20.5 KB/tok F16):
#   16k/32k/64k  F16 KV on GPU (0.3/0.6/1.3 GiB), cache 24      -- 64k may not fit; a dead row is a result
#   131k         q4_0 KV on GPU (~0.7 GiB), cache 24
#   262k         q4_0 KV on GPU (~1.4 GiB), cache 12              -- the target rung
# plus 32k variants that price the alternatives: -nkvo (KV in RAM, CPU attention), q8_0 KV, q4_0 KV (the deep rungs' precision,
# so its decode cost is known at a depth where F16 also ran), and MTP n=2 at depth (speculation + n_rs_seq tax with a real KV).
# Rungs run shallow -> deep so cheap answers land first; a failed row never aborts the job. Deep restore rows are gated on the
# 16k restore having actually skipped the prompt -- otherwise a broken restore would re-prefill 262k for hours to say "no" again.
# Time: 131k prefill ~0.5-1 h, 262k ~1-3 h (attention is n^2). TIMEOUT below covers one request. Queue this LAST.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
grep -q 'environ.get("SLOT"' /ai/bench/slotclient.py && grep -q 'environ.get("TIMEOUT"' /ai/bench/slotclient.py \
  || { echo "CTX1_REFUSED: /ai/bench/slotclient.py lacks SLOT/TIMEOUT env (deploy the brief-13 version first)"; exit 1; }
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
UB=${UB:-512}                      # lat1: -ub 512 = 2.4x long prefill for 6 cache slots
KVARGS=${KVARGS:-}                 # extra KV-cache flags appended verbatim to every row
Q4="-ctk q4_0 -ctv q4_0"
export TIMEOUT=21600               # one request may be a 262k prefill
SLOTS=/ai/bench/slots; mkdir -p $SLOTS
run() { # run MODE LABEL SLOTNAME CTX REPS [extra llama-server args, last one wins]  -- fresh server per row, killed at the end
  local mode=$1 label=$2 slot=$3 ctx=$4 reps=$5; shift 5
  echo "##### $label mode=$mode ctx=$ctx reps=$reps | $* ${KVARGS}"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $Q -ngl 999 -ot exps=CPU -c $ctx -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -lv 4 --slot-save-path $SLOTS --moe-expert-cache 24 \
    -ub $UB -b $((UB * 2)) -fa on $KVARGS "$@" > server_$label.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "    $label: SERVER DIED: $(grep -iE 'error|failed|out of memory' server_$label.log | tail -1 | cut -c1-140)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 0; }; sleep 2; done
  MODE=$mode REPS=$reps SLOT=$slot python3 /ai/bench/slotclient.py "$label"
  local rc=$?
  echo "    vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $pid; wait $pid 2>/dev/null
  grep -hE "KV buffer size|RS buffer size|MoE expert cache enabled|out of memory" server_$label.log | tail -6 | cut -c1-200 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'
  [ $rc -eq 0 ] || echo "    $label: slotclient exit $rc (row kept in runs/$label.slot.json only if it got that far)"
  return 0
}
rung() { # rung TAG CTX REPS [args...] : save, restart, restore, then drop the slot file (262k q4 ~= 1.4 GiB; disk is tight)
  local tag=$1 ctx=$2 reps=$3; shift 3
  run save ctx1_${tag}_save ctx1_$tag.slot $ctx $reps "$@"
  if [ "$RESTORE_OK" != 0 ]; then run restore ctx1_${tag}_restore ctx1_$tag.slot $ctx $reps "$@"
  else echo "##### ctx1_${tag}_restore SKIPPED: the 16k restore did not skip the prompt"; fi
  rm -f $SLOTS/ctx1_$tag.slot
}
restore_works() { python3 - <<'PY'
import json, sys
try:
    s = json.load(open("/ai/bench/runs/ctx1_c16k_save.slot.json"))["rows"][0]
    r = json.load(open("/ai/bench/runs/ctx1_c16k_restore.slot.json"))["rows"][0]
    sys.exit(0 if (r.get("prompt_n") or 0) < 0.1 * (s.get("prompt_n") or 1) else 1)
except Exception:
    sys.exit(1)
PY
}
# REPS = copies of w4_doc.txt (~1.9k tok each) => ~70-90% of the window; the row's prompt_n is the real depth.
RESTORE_OK=1
rung c16k 16384 6
restore_works && echo "    restore gate: 16k restore skipped the prompt => deep restores ON" || { RESTORE_OK=0; echo "    restore gate: 16k restore did NOT skip the prompt => deep restores OFF"; }
rung c32k 32768 13
run save ctx1_c32k_nkvo_save ctx1_tmp.slot 32768 13 -nkvo
run save ctx1_c32k_q8_save   ctx1_tmp.slot 32768 13 -ctk q8_0 -ctv q8_0
run save ctx1_c32k_q4_save   ctx1_tmp.slot 32768 13 $Q4
run save ctx1_c32k_mtp_save  ctx1_tmp.slot 32768 13 --moe-expert-cache 16 $QH
rm -f $SLOTS/ctx1_tmp.slot
rung c64k  65536  27
rung c131k 131072 58  $Q4
rung c262k 262144 116 $Q4 --moe-expert-cache 12
echo "--- verdict input: runs/ctx1_*.slot.json. Per rung: save row = cold prefill t/s + decode t/s at depth; restore row ="
echo "    prompt_n near 0 + wall seconds. A rung without rows died: read the SERVER DIED / out of memory line above it."
rmdir $SLOTS 2>/dev/null
echo CTX1_DONE
