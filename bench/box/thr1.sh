#!/bin/bash
# THR1 (lead C): CPU thread count with hyper-threading. The decode critical path alternates GPU phase (~15 ms/token) and the CPU
# miss window (~6.3 ms/token, GPU idle). cpu1bench2: at 6 threads ~1/3 of the miss phase waits on DRAM, ~2/3 is compute (dequant +
# dot) and barriers (prof2: 48% of CPU samples in libgomp spinning). The i5-10400F has 6 cores / 12 threads; -t 5 lost 6.2%
# (opt1), -t 6 is served. SMT may hide dequant latency or may steal the CUDA-driving thread's core.
# PRE-REGISTERED: an arm WINS if its 4-prompt mean decode beats -t 6 by more than -t 6's run-to-run spread (2 interleaved rounds).
# STABLE binary, serving config.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$V2
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for t in 6 8 12; do
  l=thr1_t${t}_$r
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  # specbench hardcodes -t 6; a later -t on the command line overrides it
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE -t $t 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
R = {t: [L(f"thr1_t{t}_{r}") for r in "ab"] for t in (6, 8, 12)}
kinds = list(R[6][0])
m = lambda t, k: sum(x[k]["decode_tps"] for x in R[t]) / 2
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"t{t} {m(t, k):6.2f} ({100 * (m(t, k) / m(6, k) - 1):+5.1f}%)" for t in (6, 8, 12)))
M = {t: sum(m(t, k) for k in kinds) / len(kinds) for t in R}
sp = sum(abs(R[6][0][k]["decode_tps"] - R[6][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN t6 {M[6]:.2f} (spread {sp:.2f}) | t8 {M[8]:.2f} ({100 * (M[8] / M[6] - 1):+.1f}%) | t12 {M[12]:.2f} ({100 * (M[12] / M[6] - 1):+.1f}%)"
      f" -> {', '.join(f't{t} WINS' for t in (8, 12) if M[t] - M[6] > sp) or 'no winner'}")
PY
echo THR1_DONE
