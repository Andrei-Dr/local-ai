#!/bin/bash
# OVL15: where does the hoisted LAYOUT cost host time? ovl14 (11:42): plan-only (mode 2) costs as much as the full overlap
# (decode -6.8% / -6.2%, short-prompt prefill 59 -> 50 t/s) -> not the second stream; hit rates slightly HIGHER with the plan;
# main compute buffer 129 -> 290 MiB (reserve at bs=128). nsys CUDA-API trace (graphs ON, the served mode) of specbench's code prompt
# + 300-token decode, off vs plan, then hand1_phases.py: per-layer host timeline (D launch -> CPU experts -> H2D -> B launch + sync
# -> D2H) -> which phase grows. t2 combo tip.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL15_REFUSED: $T2 not on combo"; exit 1; }
echo "    t2 at $(git -C $T2 rev-parse --short HEAD)"
for a in off plan; do
  l=ovl15_nsys_$a; E=""; [ $a = plan ] && E="GGML_SCHED_MOE_PREFETCH=2"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
    nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 26 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  T0=$(date +%s.%N)
  curl -s localhost:8099/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."}],"temperature":0,"max_tokens":300,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(f\"    $a: prompt {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} t/s | decode {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} t/s\")"
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY hand1_phases.py $l.sqlite --skip-s 0 2>&1 | head -24 | sed 's/^/    /'
  $PY - $l.sqlite <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
q = """select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
       where s.value like '%alloc%' or s.value like '%Free%' or s.value like 'cuMem%' or s.value like '%Synchronize%' group by s.value order by 3 desc"""
for name, n, t in c.execute(q):
    print(f"    api {name[:40]:40s} {n:8d} calls {t / 1e6:9.1f} ms")
PY
  rm -f $l.qdstrm
done
echo "--- VRAM-pressure test: specbench off vs plan at cache 22 (2 rounds) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
for r in a b; do for a in off plan; do
  l=ovl15_c22_${a}_$r; E=""; [ $a = plan ] && E="GGML_SCHED_MOE_PREFETCH=2"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l" | cut -c1-100 | sed 's/^/    /'
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"ovl15_c22_{a}_{r}") for r in "ab"] for a in ("off", "plan")}
kinds = list(S["off"][0]); m = lambda a, k, f="decode_tps": sum(x[k][f] for x in S[a]) / 2
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in S}
sp = sum(abs(S["off"][0][k]["decode_tps"] - S["off"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  cache 22: MEAN decode off {M['off']:.2f} (spread {sp:.2f}) | plan {M['plan']:.2f} ({100 * (M['plan'] / M['off'] - 1):+.1f}%)  [cache 26: ovl14 plan -6.8%]")
print("  prefill " + " | ".join(f"{k} off {m('off', k, 'prefill_tps'):.1f} plan {m('plan', k, 'prefill_tps'):.1f}" for k in kinds))
PY
echo OVL15_DONE
