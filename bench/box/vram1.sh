#!/bin/bash
# VRAM1 (lead F): reclaim the MTP draft's token embedding from VRAM and spend it on expert-cache slots. The draft file keeps
# token_embd (Q4_0, 508M weights, ~286 MB) on CUDA0 (draft buffer 727.68 MiB) but reads ONE row per draft step; the main model's
# token_embd is already on the CPU (CUDA_Host 11328 = experts + its IQ3_S embedding). For draft-mtp the draft inherits the MAIN
# -ot list (-otd is ignored: ovl4), so "-ot exps=CPU,token_embd=CPU" moves only the draft's embedding. +286 MB = ~250 slot-layers =
# +6 slots per layer (26 -> 32). Exact (same math; only the cache contents / hit pattern change, the cache's accepted class).
# Arms (STABLE binary, serving config, 2 interleaved rounds): base (cache 26) | te (token_embd=CPU, cache 26: cost of the extra
# host split per draft step) | te32 (token_embd=CPU, cache 32) | te30. Plus 9.3k: base (cache 22) vs te (cache 28).
# PRE-REGISTERED: te32 (or te30) WINS if its 4-prompt mean decode beats base by more than base's spread; VRAM must stay < 4000 MiB.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$V2
REST="-md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for a in base te te30 te32; do
  l=vram1_${a}_$r
  case $a in base) X="-ot exps=CPU --moe-expert-cache 26";; te) X="-ot exps=CPU,token_embd=CPU --moe-expert-cache 26";;
    te30) X="-ot exps=CPU,token_embd=CPU --moe-expert-cache 30";; te32) X="-ot exps=CPU,token_embd=CPU --moe-expert-cache 32";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $X | $(date +%T)"
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $X $REST 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-120 | sed 's/^/    /'
  echo "    $(grep -oE 'CUDA0 model buffer size = +[0-9.]+ MiB' server_$l.log | tr '\n' ' ') $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(grep -hE 'out of memory|failed to allocate' server_$l.log | head -1 | cut -c1-80)"
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = ("base", "te", "te30", "te32")
R = {a: [L(f"vram1_{a}_{r}") for r in "ab"] for a in A}
kinds = list(R["base"][0])
m = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 2
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"{a} {m(a, k):6.2f} ({100 * (m(a, k) / m('base', k) - 1):+5.1f}%)" for a in A))
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in A}
sp = sum(abs(R["base"][0][k]["decode_tps"] - R["base"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
vr = {a: max(x[k].get("vram_mib", 0) or 0 for x in R[a] for k in kinds) for a in A}
print(f"  MEAN base {M['base']:.2f} (spread {sp:.2f}) | " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['base'] - 1):+.1f}%)" for a in A[1:]))
w = [a for a in ("te30", "te32") if M[a] - M["base"] > sp]
print(f"  -> {'WIN: ' + ', '.join(w) if w else 'NOT PROVEN'}")
PY
echo "--- 9,279-token prompt: base (cache 22) vs te (cache 28) | $(date +%T)"
for r in a b; do for a in base te; do
  l=vram1_pf_${a}_$r; X="-ot exps=CPU --moe-expert-cache 22"; [ $a = te ] && X="-ot exps=CPU,token_embd=CPU --moe-expert-cache 28"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 $X -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(grep -hE 'out of memory|failed to allocate|could not re-allocate' server_$l.log | head -1 | cut -c1-80)"
  kill $NP; wait $NP 2>/dev/null
done; done
echo VRAM1_DONE
