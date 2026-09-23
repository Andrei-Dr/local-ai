#!/bin/bash
# OVL16: can the overlap keep its prefill win without the decode cost? ovl15 (11:57): with the plan (mode 2) EVERY layer-pass is
# ~5% slower in BOTH phases (CPU experts 441.6 -> 460.1 us, device 408.8 -> 431.1, period 914.6 -> 962.8), same alloc/sync counts,
# and cache 22 still -6.1% (not VRAM pressure); the startup reserve graph (bs=128), short prompts and the long prompt are planned.
# Gate: GGML_SCHED_MOE_PREFETCH_MIN_IDS=2048 plans only batches >= 256 tokens (prefill-mode ubatches), leaving the reserve graph,
# short prompts and decode unplanned. t2 combo tip.
# PRE-REGISTERED: gated overlap PASSES if (a) specbench mean decode is within off's spread of off, and (b) the 9,279-token prefill
# (ubp 2048) beats off by more than off's spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL16_REFUSED: $T2 not on combo"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH_MIN_IDS || { echo "OVL16_REFUSED: build lacks MIN_IDS"; exit 1; }
GATE="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_MIN_IDS=2048"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for a in off gated; do
  l=ovl16_${a}_$r; E=""; [ $a = gated ] && E="$GATE"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
  echo "    $(grep -m1 -oE 'CUDA0 compute buffer size = +[0-9.]+ MiB' server_$l.log)"
done; done
for r in a b; do for a in off gated; do
  l=ovl16_pf_${a}_$r; E=""; [ $a = gated ] && E="$GATE"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"ovl16_{a}_{r}") for r in "ab"] for a in ("off", "gated")}
kinds = list(S["off"][0]); m = lambda a, k, f="decode_tps": sum(x[k][f] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode off {m('off', k):6.2f} gated {m('gated', k):6.2f} ({100 * (m('gated', k) / m('off', k) - 1):+5.1f}%) | prefill off {m('off', k, 'prefill_tps'):6.1f} gated {m('gated', k, 'prefill_tps'):6.1f}")
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in S}
sp = sum(abs(S["off"][0][k]["decode_tps"] - S["off"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
P = lambda a, f: [json.load(open(f"/ai/bench/runs/ovl16_pf_{a}_{r}.longpf.json"))[f] for r in "ab"]
po, pg = P("off", "prefill_tps"), P("gated", "prefill_tps"); do, dg = P("off", "decode_tps"), P("gated", "decode_tps")
psp = abs(po[0] - po[1])
a_ok = abs(M["gated"] - M["off"]) <= sp or M["gated"] >= M["off"]
b_ok = sum(pg) / 2 - sum(po) / 2 > psp
print(f"  specbench MEAN off {M['off']:.2f} (spread {sp:.2f}) gated {M['gated']:.2f} ({100 * (M['gated'] / M['off'] - 1):+.1f}%) -> (a) {'OK' if a_ok else 'FAIL'}")
print(f"  9.3k prefill off {po} gated {pg} -> {100 * (sum(pg) / sum(po) - 1):+.1f}% -> (b) {'OK' if b_ok else 'FAIL'} | decode after off {do} gated {dg}")
print(f"  -> {'PASS' if a_ok and b_ok else 'FAIL'}")
PY
echo OVL16_DONE
