#!/bin/bash
# PMUX1: MoE prefill mode (--ubatch-prefill, llama.cpp-ov branch prefill-mux = served build + 1 commit). A batch larger than
# -ub releases the expert-cache slots (~1.1 GB at 26 slots), re-reserves the scheduler for the prefill ubatch in that VRAM,
# and the first decode step shrinks it back and restores the slots, re-ranked by the prompt's routing.
# Facts: the long prompt (2,181 tok) spends 45 s of its 51 s wall in prefill at ub 128 (48.5 t/s); opt1: ub 256 at cache 26 =
# +50% prefill at no decode cost; lat1: ub 512 = 2.4x prefill but cost ~6 slots + 3-5% decode when both lived at once.
# Arms: served ub128 (base), served ub256 (the free config win), prefill mode at ubp 512 / 1024 / 2048 (-b 2048 so the prompt
# arrives in one batch), base again. Per arm: long prefill t/s, decode t/s (after the switch back), peak VRAM (1 Hz), the
# release / restore lines. Then identity under LLAMA_MOE_CACHE_SYNC=1 (served ub128 vs prefill mode at the best fitting ubp),
# and the multi-turn probe (turns.py, -c 8192) on base and prefill mode.
# Read: a prefill-mode arm is GO when it (1) never OOMs, (2) beats ub256 on long prefill, (3) decode within the base spread.
# Identity: prefill at another ubatch size moves ubatch boundaries (delta-net chunks, attention tiles), so byte-identical texts
# are NOT guaranteed the way they were for 0007-0009; a divergence is judged by KLD next, not waved through.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
SERVED=/ai/src/llama.cpp-mainline/build75; OV=/ai/src/llama.cpp-ov/build75
LD_LIBRARY_PATH=$OV/bin $OV/bin/llama-server --help 2>/dev/null | grep -q -- "--ubatch-prefill" || { echo "PMUX1_REFUSED: ov build has no --ubatch-prefill"; exit 1; }
BASEARGS="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
vram_peak() { local m=0 v; while kill -0 $1 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$m" ] && m=$v; echo $m > /tmp/pmux_vram.$2; sleep 1; done; }
arm() { # label build(served|ov) extra-args...
  local l=$1 b=$2; shift 2
  if [ $b = ov ]; then export BUILD=$OV LD_LIBRARY_PATH=$OV/bin; else export BUILD=$SERVED; unset LD_LIBRARY_PATH; fi
  echo "##### $l | $b | $*"
  ( sleep 5; vram_peak $$ $l ) & local VP=$!
  ./specbench.sh 999 $l $BASEARGS "$@" 2>&1
  kill $VP 2>/dev/null; wait $VP 2>/dev/null
  echo "    peak VRAM $(cat /tmp/pmux_vram.$l 2>/dev/null) MiB"
  grep -hE "prefill mode: released|decode mode: restored|could not re-allocate|n_ubatch_prefill|compute buffer size|out of memory|failed to allocate" server_$l.log \
    | sort | uniq -c | sort -rn | head -8 | cut -c1-170 | sed 's/^/    /'
}
arm pmux1_base_a    served -ub 128 -b 256
arm pmux1_ub256     served -ub 256 -b 512
arm pmux1_ubp512    ov     -ub 128 -b 2048 -ubp 512
arm pmux1_ubp1024   ov     -ub 128 -b 2048 -ubp 1024
arm pmux1_ubp2048   ov     -ub 128 -b 2048 -ubp 2048
arm pmux1_base_b    served -ub 128 -b 256
unset LD_LIBRARY_PATH
echo "--- READ (long prompt prefill, decode 4-prompt mean vs base mean +- base spread)"
BEST=$($PY - 2>&1 1>/tmp/pmux_best <<'PY' | cat >&2; cat /tmp/pmux_best
import json, os, sys
P = lambda *a, **k: print(*a, file=sys.stderr, **k)
def L(l):
    p = f"/ai/bench/runs/{l}.client.json"
    return {r["prompt"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else None
ba, bb = L("pmux1_base_a"), L("pmux1_base_b")
kinds = list(ba)
bm = {k: (ba[k]["decode_tps"] + bb[k]["decode_tps"]) / 2 for k in kinds}
bs = sum(abs(ba[k]["decode_tps"] - bb[k]["decode_tps"]) for k in kinds) / len(kinds)
bmean = sum(bm.values()) / len(kinds)
bpf = (ba["long"]["prefill_tps"] + bb["long"]["prefill_tps"]) / 2
u = L("pmux1_ub256"); upf = u["long"]["prefill_tps"] if u else 0
P(f"  base      long prefill {bpf:6.1f} t/s | decode mean {bmean:6.2f} (spread {bs:.2f})")
best, bestpf = "none", 0
for arm in ("pmux1_ub256", "pmux1_ubp512", "pmux1_ubp1024", "pmux1_ubp2048"):
    a = L(arm)
    if not a:
        P(f"  {arm:13s} NO RESULT (server died: see its lines above)"); continue
    m = sum(a[k]["decode_tps"] for k in kinds) / len(kinds)
    pf = a["long"]["prefill_tps"]
    wall = a["long"]["prompt_tokens"] / pf + a["long"]["tokens"] / a["long"]["decode_tps"]
    wall0 = ba["long"]["prompt_tokens"] / bpf + ba["long"]["tokens"] / bm["long"]
    go = arm != "pmux1_ub256" and pf > upf and abs(m - bmean) <= max(bs, 0.02 * bmean)
    P(f"  {arm:13s} long prefill {pf:6.1f} t/s ({pf / bpf:4.2f}x) | long wall {wall:5.1f} s vs {wall0:5.1f} s | decode mean {m:6.2f} ({100 * (m / bmean - 1):+5.1f}%) | {'GO' if go else '-'}")
    if go and pf > bestpf:
        best, bestpf = arm, pf
print(best.replace("pmux1_ubp", ""))
PY
)
echo "  best fitting prefill-mode ubatch: ${BEST:-none}"
if [ -n "$BEST" ] && [ "$BEST" != none ]; then
  echo "--- IDENTITY under LLAMA_MOE_CACHE_SYNC=1 (served ub128 vs prefill mode ubp $BEST)"
  LLAMA_MOE_CACHE_SYNC=1 arm pmux1_id_base served -ub 128 -b 256
  LLAMA_MOE_CACHE_SYNC=1 arm pmux1_id_pmux ov     -ub 128 -b 2048 -ubp $BEST
  unset LD_LIBRARY_PATH
  $PY textdiff.py runs/pmux1_id_base.client.json runs/pmux1_id_pmux.client.json | sed 's/^/    /'
  echo "--- MULTI-TURN (turns.py, -c 8192, thinking on)"
  for b in served ov; do
    if [ $b = ov ]; then BLD=$OV; export LD_LIBRARY_PATH=$OV/bin; EXTRA="-ub 128 -b 2048 -ubp $BEST"; else BLD=$SERVED; unset LD_LIBRARY_PATH; EXTRA="-ub 128 -b 256"; fi
    GGML_OP_OFFLOAD_MIN_BATCH=32 $BLD/bin/llama-server -m $K2 -ngl 999 -fa on -c 8192 -t 6 --load-mode none --jinja --parallel 1 \
      --port 8099 --cache-ram 0 $BASEARGS $EXTRA > server_pmux1_turns_$b.log 2>&1 &
    NP=$!
    for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
    python3 turns.py pmux1_turns_$b 2>&1 | sed "s/^/    [$b] /"
    grep -hiE "checkpoint|forcing full prompt|erased invalidated|prefill mode|decode mode" server_pmux1_turns_$b.log | tail -6 | cut -c1-170 | sed "s/^/    [$b] /"
    kill $NP; wait $NP 2>/dev/null
  done
  unset LD_LIBRARY_PATH
fi
echo PMUX1_DONE
