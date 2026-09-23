#!/bin/bash
# PF2: pre-gated prefetch, copies issued off the critical path. pf1 (10:10): hit rate 48 -> 60 / 63 / 67% at budget 2 / 3 / 4, but
# decode -3 / -7 / -10%: the head op issued ~3 cudaMemcpyAsync per expert on CPU thread 0 under the cache lock while the other
# threads waited (critical path). f43cd33: head op only plans + posts; a helper thread issues while the host experts run; a tail
# node joins before the next device split. Also includes 8bdf719 (review fixes). Arms (specbench serving config, MTP n=2, top-8,
# 2 interleaved rounds): off | async b2 | async b4 | inline b4 (= pf1's behavior, same build) -> head/join us per layer logged.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=f43cd33; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "PF2_REFUSED: $T2 has local changes"; exit 1; }
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "pregate-prefetch" ] || { echo "PF2_REFUSED: $T2 not on pregate-prefetch"; exit 1; }
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "PF2_REFUSED: cannot fast-forward to $SHA"; exit 1; }; fi
cmake --build $NEW -j6 --target llama-server llama-perplexity > pf2_build.log 2>&1 || { grep error pf2_build.log | head; echo "PF2_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MOE_PREFETCH_INLINE || { echo "PF2_FAILED: build lacks f43cd33"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
cst() { grep -oE "hit rate [0-9.]+%|prefetch [0-9]+ \([^)]*\)|pre-gated prefetch on[^(]*\([^)]*\)" server_$1.log | tail -3 | tr '\n' ' '; }
for r in a b; do for a in off ab2 ab4 ib4; do
  l=pf2_${a}_$r
  case $a in off) E="";; ab2) E="LLAMA_MOE_PREFETCH=2 LLAMA_MOE_PREFETCH_TOPK=8";; ab4) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8";;
    ib4) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8 LLAMA_MOE_PREFETCH_INLINE=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-off} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
  echo "    $(cst $l)"
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = ("off", "ab2", "ab4", "ib4")
R = {a: [L(f"pf2_{a}_{r}") for r in "ab"] for a in A}
kinds = list(R["off"][0])
m = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 2
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"{a} {m(a, k):6.2f} ({100 * (m(a, k) / m('off', k) - 1):+5.1f}%)" for a in A))
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in A}
sp = sum(abs(R["off"][0][k]["decode_tps"] - R["off"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN off {M['off']:.2f} (spread {sp:.2f}) | " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['off'] - 1):+.1f}%)" for a in A[1:]))
PY
echo PF2_DONE
