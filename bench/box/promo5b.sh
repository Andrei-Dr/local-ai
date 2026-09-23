#!/bin/bash
# PROMO5B: re-run of promo5 after the plan-gate fix (d0fd493: plan only graphs >= 8 routed tokens/expert; promo5 showed the
# decode-mode reserve graph planned -> 290 MiB compute buffer -> slots could not re-allocate after prefill -> decode -20%, and
# specbench OOM at cache 26). Baseline for identity = promo5_id_new (the 2f02192 binary, same env).
# (promo5 header follows) PROMO5: lead E into STABLE — upload/compute overlap for prefill (GGML_SCHED_MOE_PREFETCH=1) with the plan-loop fix.
# ovl20 (13:38): overlap on vs unset — specbench decode -0.1% (spread 0.85), code prefill 57.7 vs 57.6, long prefill 468.2 vs 394.8,
# 9.3k prefill 499.1 vs 402.8 (+23.9%), decode after 9.3k 50.06 vs 50.01. Prefill KLD identical to 6 decimals (ovl12).
# New STABLE commit 2f021923d = c75018b + overlap (72cf575) + diag modes (c994595 b3cb7df 4e1586e 92994bf, env-gated) + loop fix (2f02192).
# Steps: 1 identity baseline (current STABLE) | 2 ff + rebuild | 3 identity with GGML_SCHED_MOE_PREFETCH unset -> IDENTICAL |
# 4 speed with the full serving env (FA_MMA_MAX_KV 4096 + 49k draft vocab, n2): specbench unset vs on (2 rounds) + 9,279-token
#   prompt at ubp 2048 unset / on and ubp 4096 on (2 rounds).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2; NEW=$V2/build75; SHA=d0fd493; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
BASEENV="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
[ -z "$(git -C $V2 status --porcelain --untracked-files=no)" ] || { echo "PROMO5B_REFUSED: $V2 has local changes"; exit 1; }
git -C $V2 merge -q --ff-only $SHA || { echo "PROMO5B_REFUSED: cannot fast-forward to $SHA"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-perplexity llama-cli llama-bench test-backend-ops > promo5b_build.log 2>&1 || { grep error promo5b_build.log | head; echo "PROMO5B_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH || { echo "PROMO5B_FAILED: no GGML_SCHED_MOE_PREFETCH"; exit 1; }
echo "    built $(git -C $V2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 3. IDENTITY (overlap unset) | $(date +%T)"
env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 $BASEENV ./specbench.sh 999 promo5b_id_new $SERVE > /dev/null 2>&1
[ -s runs/promo5_id_new.client.json ] || { echo "PROMO5B_FAILED: no baseline"; exit 1; }
$PY textdiff.py runs/promo5_id_new.client.json runs/promo5b_id_new.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PROMO5B_FAILED: not identical with the overlap unset"; exit 1; }
echo "--- 4. SPEED | $(date +%T)"
for r in a b; do for a in unset on; do
  l=promo5b_${a}_$r; E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E $BASEENV ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
done; done
for r in a b; do for a in u2048 o2048 o4096; do
  l=promo5b_pf_${a}_$r; E=""; U=2048
  case $a in o2048) E="GGML_SCHED_MOE_PREFETCH=1";; o4096) E="GGML_SCHED_MOE_PREFETCH=1"; U=4096;; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E $BASEENV GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 \
    -ub 128 -b 4096 -ubp $U > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1 | tr -d "\n"; echo " | $(grep -oE 'could not re-allocate[^\n]{0,40}|compute buffer size = +[0-9.]+ MiB' server_$l.log | head -2 | tr '\n' ' ')"
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"promo5b_{a}_{r}") for r in "ab"] for a in ("unset", "on")}
kinds = list(S["unset"][0]); m = lambda a, k, f="decode_tps": sum(x[k][f] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode unset {m('unset', k):6.2f} on {m('on', k):6.2f} ({100 * (m('on', k) / m('unset', k) - 1):+5.1f}%) | prefill {m('unset', k, 'prefill_tps'):6.1f} -> {m('on', k, 'prefill_tps'):6.1f}")
P = lambda a, f: [json.load(open(f"/ai/bench/runs/promo5b_pf_{a}_{r}.longpf.json"))[f] for r in "ab"]
for a in ("u2048", "o2048", "o4096"):
    print(f"  9.3k {a}: prefill {[round(v, 1) for v in P(a, 'prefill_tps')]} | decode after {[round(v, 2) for v in P(a, 'decode_tps')]} | wall {P(a, 'wall_s')}")
PY
echo PROMO5B_DONE
