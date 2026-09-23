#!/bin/bash
# FR1 (lead B): FR-Spec draft vocabulary for the MTP head (combo 854cbce, LLAMA_MTP_VOCAB_FILE). pf3: the draft head reads ~0.81
# ms/token at VRAM roofline; restricting it to the K most frequent tokens cuts that ~proportionally; the target verifies against the
# full vocabulary, so only acceptance can change.
# PRE-REGISTERED: a K WINS if its 4-prompt mean decode beats full by more than full's spread AND its mean acceptance drops <= 2 points.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=854cbce; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "FR1_REFUSED: $T2 has local changes"; exit 1; }
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "FR1_REFUSED: $T2 not on combo"; exit 1; }
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "FR1_REFUSED: cannot fast-forward"; exit 1; }; fi
cmake --build $NEW -j6 --target llama-server llama-tokenize > fr1_build.log 2>&1 || { grep error fr1_build.log | head; echo "FR1_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MTP_VOCAB_FILE || { echo "FR1_FAILED: build lacks the FR-Spec head"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 0. VOCAB LISTS | $(date +%T)"
env -u LD_LIBRARY_PATH $PY fr1_vocab.py $NEW/bin/llama-tokenize $K2 /ai/bench/fr1_vocab 2>&1 | sed 's/^/    /'
[ -s fr1_vocab_32768.bin ] || { echo "FR1_FAILED: no vocab list"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "--- 1. SPEED + ACCEPTANCE (2 interleaved rounds) | $(date +%T)"
for r in a b; do for k in full 8192 16384 32768; do
  l=fr1_${k}_$r; E=""; [ $k != full ] && E="LLAMA_MTP_VOCAB_FILE=/ai/bench/fr1_vocab_$k.bin"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  grep -hoE "MTP draft head restricted[^\n]*" server_$l.log | head -1 | sed 's/^/    /'
done; done
$PY - <<'PY'
import json, re
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = ("full", "8192", "16384", "32768")
R = {a: [L(f"fr1_{a}_{r}") for r in "ab"] for a in A}
kinds = list(R["full"][0])
m = lambda a, k, f="decode_tps": sum(x[k][f] for x in R[a]) / 2
acc = lambda a: sum(m(a, k, "acceptance") for k in kinds) / len(kinds)
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"{a} {m(a, k):6.2f} ({100 * (m(a, k) / m('full', k) - 1):+5.1f}%) acc {m(a, k, 'acceptance'):.3f}" for a in A))
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in A}
sp = sum(abs(R["full"][0][k]["decode_tps"] - R["full"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN full {M['full']:.2f} (spread {sp:.2f}, acc {acc('full'):.3f}) | " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['full'] - 1):+.1f}%, acc {acc(a):.3f})" for a in A[1:]))
wins = [a for a in A[1:] if M[a] - M["full"] > sp and acc("full") - acc(a) <= 0.02]
print(f"  -> {'WIN: ' + ', '.join(wins) if wins else 'NOT PROVEN'}")
PY
echo FR1_DONE
