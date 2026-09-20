#!/bin/bash
# CTX1B: the restore test ctx1 should have run. ctx1's restore rows re-sent the SAME prompt against a slot saved AFTER 64 generated
# tokens; on a GDN hybrid that needs a recurrent-state rollback, so the server logged "forcing full prompt re-processing" and
# recomputed everything (restore prompt_n == save prompt_n). Slot I/O itself was fine (307 MiB in 0.16-0.6 s, 723 MiB at 131k in
# 0.4 s). Supported path = restore, then a prompt whose TOKEN IDS extend the saved ones: slotclient presave/extend (brief 17).
# Hypothesis: extend rows show prompt_n ~= ext_n (tens of tokens) and a coherent reply => hours of deep prefill become < 1 s, and
# TWO-PHASE works: PREFILL server (cache 0, big ubatch: 32k measured 75.7/107.6/135.4 t/s at ub 512/1024/2048) -> slot -> DECODE
# server (expert cache, -ub 128 => small compute buffer). That split is also what makes 262k fit: ctx1's 262k rung died on a
# 926 MiB COMPUTE buffer (ub 512 + cache 12), not on the 1.4 GiB q4 KV.
# Kill: no (PRE_GEN, DROP) combo passes at 16k (prompt_n < 10% of presave AND non-empty reply) => slot restore is dead on this arch.
# 16k matrix: PRE_GEN 64 = state saved after generation (the production/ctxproxy path), PRE_GEN 1 = near-pure prompt state;
# DROP 1 = resend without the last saved id (covers "last sampled token is in the token list but not in the KV").
# The 131k and 262k slot files are KEPT (0.7 + 1.4 GiB): they are 0.5-3 h of prefill that lq1 / decode tuning can reuse for free.
# Same build as ctx1 (moe-cache 2582f5c, left checked out by r1b). A dead row never aborts the job.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
grep -q 'presave' /ai/bench/slotclient.py && [ "$(grep -c '"parse_special": True' /ai/bench/slotclient.py)" -ge 2 ] \
  || { echo "CTX1B_REFUSED: /ai/bench/slotclient.py lacks presave/extend with parse_special EXT (deploy the brief-18 version first)"; exit 1; }
# Run 1 (2026-09-20): all four 16k combos restored + hit the cache (prompt_n 15, cache_n 12460/12461, 0.5 s vs 147 s) but replied
# with 1 token = end-of-turn: EXT was raw text inside the assistant turn. EXT is now a chat-formatted user turn (needs brief 18).
export EXT=$'<|im_end|>\n<|im_start|>user\nNow list the three most important open risks from the text above, one line each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
M=/ai/models; BUILD=/ai/src/llama.cpp-mainline/build75
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
Q4="-ctk q4_0 -ctv q4_0"
DEC="--moe-expert-cache 24 -ub 128 -b 256"      # decode server: the config every -c 4096 ledger row used
export TIMEOUT=21600
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
pick() { python3 - <<'PY'
import json
def row(l):
    try: return json.load(open(f"/ai/bench/runs/{l}.slot.json"))["rows"][0]
    except Exception: return None
for g in (64, 1):
    s = row(f"ctx1b_c16k_g{g}_presave")
    for d in (0, 1):
        x = row(f"ctx1b_c16k_g{g}_d{d}_extend")
        if s and x and (x.get("prompt_n") or 10**9) < 0.1 * (s.get("prompt_n") or 1) and (x.get("reply") or "").strip():
            print(g, d); raise SystemExit(0)
raise SystemExit(1)
PY
}
# 1. the mechanism, 16k, same config both sides (REPS as in ctx1 so prompt_n is comparable: 12397)
for g in 64 1; do
  run presave ctx1b_c16k_g${g}_presave ctx1b_c16k_g$g.slot 16384 6 $g 0
  for d in 0 1; do run extend ctx1b_c16k_g${g}_d${d}_extend ctx1b_c16k_g$g.slot 16384 6 $g $d; done
  rm -f $SLOTS/ctx1b_c16k_g$g.slot
done
if GD=$(pick); then set -- $GD; G=$1; D=$2; echo "    extend gate: PASS with pre_gen=$G drop=$D => deep rungs ON"
else echo "    extend gate: NO combo skipped the prompt with a coherent reply => slot restore is dead on this arch, deep rungs OFF"; echo CTX1B_DONE; exit 0; fi
# 2. two-phase at 32k: prefill config -> decode config, then the same slot under decode + MTP (does the draft head survive a restore?)
run presave ctx1b_c32k_pf_presave  ctx1b_c32k.slot 32768 13 $G 0  --moe-expert-cache 0 -ub 2048 -b 2048
run extend  ctx1b_c32k_dec_extend  ctx1b_c32k.slot 32768 13 $G $D $DEC
run extend  ctx1b_c32k_mtp_extend  ctx1b_c32k.slot 32768 13 $G $D --moe-expert-cache 8 -ub 128 -b 256 $QH
rm -f $SLOTS/ctx1b_c32k.slot
# 3. 131k two-phase (q4 KV ~0.7 GiB; ub 1024 keeps the prefill compute buffer ~0.9 GiB). Slot KEPT.
run presave ctx1b_c131k_pf_presave ctx1b_c131k.slot 131072 58 $G 0  $Q4 --moe-expert-cache 0 -ub 1024 -b 2048
run extend  ctx1b_c131k_dec_extend ctx1b_c131k.slot 131072 58 $G $D $Q4 $DEC
# 4. 262k two-phase, the target (q4 KV ~1.4 GiB; cache 0 frees what the 926 MiB compute buffer needed). Slot KEPT. ~2-3 h prefill.
run presave ctx1b_c262k_pf_presave ctx1b_c262k.slot 262144 116 $G 0  $Q4 --moe-expert-cache 0 -ub 512 -b 1024
run extend  ctx1b_c262k_dec_extend ctx1b_c262k.slot 262144 116 $G $D $Q4 --moe-expert-cache 8 -ub 128 -b 256
echo "--- verdict input: runs/ctx1b_*.slot.json. extend rows: prompt_n ~= ext_n + a coherent reply = restore works; decode_tps ="
echo "    decode at depth under the decode config. Kept slots: $(ls -la $SLOTS/ctx1b_c131k.slot $SLOTS/ctx1b_c262k.slot 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
echo CTX1B_DONE
