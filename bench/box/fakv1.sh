#!/bin/bash
# FAKV1 (lead D, the surviving part): FlashAttention MMA only below a KV length (GGML_CUDA_FA_MMA_MAX_KV, fa-kv 3ea44a4 on combo).
# att1: at 9.3k KV the tile kernel is 1.9x faster per launch (630 -> 334 us) and decode +3.2%; pf4: w4+NO_MMA +3.0% at 9.3k; on
# short prompts NO_MMA alone is neutral (pf4 nm -0.2%, spread 1.09) while tu1 once measured -3.9%. A KV threshold keeps MMA where
# the context is short and the tile kernel where the KV scan dominates. Arms (combo build, prefetch off):
#   specbench (short/2.2k prompts): MMA (served) | MAX_KV 4096 | NO_MMA            — 2 interleaved rounds
#   9,279-token prompt + 128-token decode: MMA | MAX_KV 2048 | MAX_KV 4096 | MAX_KV 8192 — 2 rounds
# PRE-REGISTERED: a threshold WINS if the 9.3k decode mean beats MMA by more than MMA's spread AND the specbench 4-prompt mean is not
# below MMA by more than MMA's spread. Numerics: the tile kernel is KLD non-inferior (pmux5 0.199801 vs MMA 0.200676).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
SHA=$(git -C /ai/src/llama.cpp-mainline rev-parse --short=7 combo 2>/dev/null)
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "FAKV1_REFUSED: $T2 has local changes"; exit 1; }
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "FAKV1_REFUSED: $T2 not on combo"; exit 1; }
git -C $T2 merge -q --ff-only $SHA || { echo "FAKV1_REFUSED: cannot fast-forward to $SHA"; exit 1; }
cmake --build $NEW -j6 --target llama-server > fakv1_build.log 2>&1 || { grep error fakv1_build.log | head; echo "FAKV1_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-cuda.so* | grep -q GGML_CUDA_FA_MMA_MAX_KV || { echo "FAKV1_FAILED: build lacks fa-kv"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "--- 1. SPECBENCH | $(date +%T)"
for r in a b; do for a in mma kv4096 nomma; do
  l=fakv1_${a}_$r; case $a in mma) E="";; kv4096) E="GGML_CUDA_FA_MMA_MAX_KV=4096";; nomma) E="GGML_CUDA_FA_NO_MMA=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-served} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-100 | sed 's/^/    /'
done; done
echo "--- 2. 9,279-TOKEN PROMPT | $(date +%T)"
for r in a b; do for a in mma kv2048 kv4096 kv8192; do
  l=fakv1_pf_${a}_$r; case $a in mma) E="";; kv*) E="GGML_CUDA_FA_MMA_MAX_KV=${a#kv}";; esac
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
S = {a: [L(f"fakv1_{a}_{r}") for r in "ab"] for a in ("mma", "kv4096", "nomma")}
kinds = list(S["mma"][0])
m = lambda a, k: sum(x[k]["decode_tps"] for x in S[a]) / 2
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in S}
ssp = sum(abs(S["mma"][0][k]["decode_tps"] - S["mma"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  specbench MEAN mma {M['mma']:.2f} (spread {ssp:.2f}) | kv4096 {M['kv4096']:.2f} ({100 * (M['kv4096'] / M['mma'] - 1):+.1f}%) | nomma {M['nomma']:.2f} ({100 * (M['nomma'] / M['mma'] - 1):+.1f}%)")
P = lambda a: [json.load(open(f"/ai/bench/runs/fakv1_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab"]
pm = P("mma"); psp = abs(pm[0] - pm[1]); base = sum(pm) / 2
for a in ("kv2048", "kv4096", "kv8192"):
    v = P(a); mv = sum(v) / 2
    short_ok = a != "kv4096" or M["kv4096"] >= M["mma"] - ssp
    print(f"  9.3k decode {a}: {v[0]:.2f} {v[1]:.2f} vs mma {pm[0]:.2f} {pm[1]:.2f} -> {100 * (mv / base - 1):+.1f}% (spread {psp:.2f}) -> {'WIN' if mv - base > psp and short_ok else 'NOT PROVEN'}")
PY
echo FAKV1_DONE
