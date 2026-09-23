#!/bin/bash
# KVQ1 (lead G): q8_0 KV cache — two faces: VRAM (f16 KV = 20 KiB/token x 12288 = ~240 MB at -c 12288; q8_0 halves it -> ~+3 expert
# slots/layer) and FA bandwidth (the KV scan reads half the bytes; att1: FA 2.98 ms/token at 9.3k). LOSSY for the target -> KLD
# gate first. STABLE binary.
# PRE-REGISTERED: (1) KLD: q8 mean KLD within 1.0% (relative) of f16 and same-top within 0.5 points (llama-perplexity -c 2048, 6
# chunks, vs pfkld_q6.kld); (2) SPEED (only meaningful if 1 passes): 9.3k decode q8 + cache 25 beats f16 + cache 22 by more than
# f16's spread, 2 rounds; specbench (-c 4096) q8 cache 27 vs f16 cache 26 not worse than spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -s pfkld_q6.kld ] || { echo "KVQ1_REFUSED: no pfkld_q6.kld"; exit 1; }
echo "--- 1. KLD | $(date +%T)"
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on -b 2048 -ub 512"
for a in f16 q8; do
  X=""; [ $a = q8 ] && X="-ctk q8_0 -ctv q8_0"
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-perplexity -m $K2 $COMMON $X \
    --kl-divergence-base pfkld_q6.kld --kl-divergence > kvq1_kld_$a.log 2>&1 || { tail -3 kvq1_kld_$a.log; echo "KVQ1_FAILED: kld $a"; exit 1; }
  echo "  $a: $(grep -E 'Mean +KLD|Same top p' kvq1_kld_$a.log | tr -s ' ' | tr '\n' ' ')"
done
PASS=$($PY - <<'PY'
import re
g = lambda a: (lambda t: (float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)), float(re.search(r"Same top p: +([\d.]+)", t).group(1))))(open(f"/ai/bench/kvq1_kld_{a}.log").read())
f, q = g("f16"), g("q8")
rel = 100 * (q[0] / f[0] - 1)
ok = (abs(rel) <= 1.0 or q[0] <= f[0]) and q[1] >= f[1] - 0.5
print(f"KLD f16 {f[0]:.6f} top {f[1]:.3f} | q8 {q[0]:.6f} top {q[1]:.3f} -> {rel:+.2f}% -> {'PASS' if ok else 'FAIL'}")
PY
)
echo "  $PASS"
case "$PASS" in *PASS*) ;; *) echo KVQ1_DONE; exit 0;; esac
echo "--- 2. SPEED 9.3k: f16 cache 22 vs q8 cache 25 | $(date +%T)"
for r in a b; do for a in f16 q8; do
  l=kvq1_pf_${a}_$r; X="--moe-expert-cache 22"; [ $a = q8 ] && X="-ctk q8_0 -ctv q8_0 --moe-expert-cache 25"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU $X -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(grep -hE 'out of memory|failed to allocate|could not re-allocate' server_$l.log | head -1 | cut -c1-80)"
  kill $NP; wait $NP 2>/dev/null
done; done
echo "--- 3. SPEED specbench: f16 cache 26 vs q8 cache 27 | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$V2
REST="-ot exps=CPU -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for a in f16 q8; do
  l=kvq1_${a}_$r; X="--moe-expert-cache 26"; [ $a = q8 ] && X="-ctk q8_0 -ctv q8_0 --moe-expert-cache 27"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $REST $X 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-100 | sed 's/^/    /'
done; done
$PY - <<'PY'
import json
P = lambda a: [json.load(open(f"/ai/bench/runs/kvq1_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab"]
f, q = P("f16"), P("q8"); sp = abs(f[0] - f[1])
print(f"  9.3k decode f16 {f} | q8 {q} -> {100 * (sum(q) / sum(f) - 1):+.1f}% (spread {sp:.2f}) -> {'WIN' if sum(q) / 2 - sum(f) / 2 > sp else 'NOT PROVEN'}")
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"kvq1_{a}_{r}") for r in "ab"] for a in ("f16", "q8")}
kinds = list(S["f16"][0]); m = lambda a, k: sum(x[k]["decode_tps"] for x in S[a]) / 2
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in S}
ssp = sum(abs(S["f16"][0][k]["decode_tps"] - S["f16"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  specbench MEAN f16 {M['f16']:.2f} (spread {ssp:.2f}) | q8 {M['q8']:.2f} ({100 * (M['q8'] / M['f16'] - 1):+.1f}%)")
PY
echo KVQ1_DONE
