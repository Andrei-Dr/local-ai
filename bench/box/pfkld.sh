#!/bin/bash
# PFKLD: decode-path quality gate for pre-gated prefetch (pregate-prefetch f43cd33, t2). Prefetch changes only WHICH experts are
# cache hits (device vs host rounding), so the rule is non-inferiority vs the same build without prefetch, measured on the decode
# path: llama-perplexity with -b 4 -ub 4 (every ubatch <= 4 tokens -> the expert-cache chain and the prefetch run on every step),
# KLD vs Q6_K_P truth logits, 6 chunks of wikitext at -c 2048. Arms: nocache (--moe-expert-cache 0, all experts on the CPU),
# cache (26 slots, no prefetch), pf3 (26 slots, LLAMA_MOE_PREFETCH=3, TOPK=8).
# PRE-REGISTERED: pf3 PASSES if its mean KLD is within 0.5% (relative) of `cache` AND its same-top is within 0.5 points.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; V2=/ai/src/llama.cpp-v2/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; Q6=$M/$N-Q6_K_P.gguf; PY=/ai/.venv/bin/python
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "f43cd33" ] || { echo "PFKLD_REFUSED: $T2 not at f43cd33"; exit 1; }
cmake --build $NEW -j6 --target llama-perplexity > pfkld_build.log 2>&1 || { grep error pfkld_build.log | head; echo "PFKLD_FAILED: build"; exit 1; }
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on"
stats() { grep -E "Mean +KLD|99\.0% +KLD|Same top p|Mean PPL\(Q\)" "$1" | cut -c1-110 | sed 's/^/    /'; }
echo "##### Q6_K_P reference logits (c 2048, ub 512) | $(date +%T)"
[ -s pfkld_q6.kld ] || env -u LD_LIBRARY_PATH GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-perplexity -m $Q6 $COMMON -b 2048 -ub 512 --kl-divergence-base pfkld_q6.kld > pfkld_q6.log 2>&1 \
  || { tail -3 pfkld_q6.log; echo "PFKLD_FAILED: base"; exit 1; }
arm() { local l=$1 envs=$2; shift 2; echo "##### $l | env: ${envs:-none} | $* | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-perplexity -m $K2 $COMMON -b 4 -ub 4 "$@" \
    --kl-divergence-base pfkld_q6.kld --kl-divergence > pfkld_$l.log 2>&1 || { tail -3 pfkld_$l.log; echo "PFKLD_FAILED: $l"; exit 1; }
  stats pfkld_$l.log; grep -oE "hit rate [0-9.]+%|prefetch [0-9]+ \([0-9.]+ MiB/step\)" pfkld_$l.log | tail -2 | tr '\n' ' ' | sed 's/^/    /'; echo; }
arm nocache ""                                          --moe-expert-cache 0
arm cache   ""                                          --moe-expert-cache 26
arm pf3     "LLAMA_MOE_PREFETCH=3 LLAMA_MOE_PREFETCH_TOPK=8" --moe-expert-cache 26
$PY - <<'PY'
import re
def get(l):
    t = open(f"/ai/bench/pfkld_{l}.log").read()
    k = float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)); s = float(re.search(r"Same top p: +([\d.]+)", t).group(1))
    return k, s
c, p, n = get("cache"), get("pf3"), get("nocache")
rel = 100 * (p[0] / c[0] - 1)
ok = abs(rel) <= 0.5 or p[0] <= c[0]
ok = ok and (p[1] >= c[1] - 0.5)
print(f"  nocache KLD {n[0]:.6f} top {n[1]:.3f} | cache {c[0]:.6f} top {c[1]:.3f} | pf3 {p[0]:.6f} top {p[1]:.3f} -> pf3 vs cache {rel:+.2f}% KLD, {p[1] - c[1]:+.3f} top -> {'PASS' if ok else 'FAIL'}")
PY
echo PFKLD_DONE
