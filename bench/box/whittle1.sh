#!/bin/bash
# WHITTLE1 (SPEC S4): first contact with Whittle-Qwen-3.8-35B-A3B = dense Qwen3.8-27B DISTILLED into an A3B MoE body (8 of 180
# routed experts + shared) plus a 10 B hashed n-gram memory; arch qwen4exp (in our mainline tree). This is the "dense model run as
# MoE" idea already built, by someone else, the only form of it the 2026-09-21 scout found that exists for the Qwen3 family.
# File: mradermacher i1-Q2_K, 12,968,342,080 bytes (the only quant that fits 16 GB RAM; Q3_K_M is 16.7 GB).
# Questions, in order: does it load and fit (which tensors are big, where do they go), decode t/s with our expert cache, and the
# kq1h-comparable quality row (GSM8K-200 + MMLU-Pro-280, thinking off): served Qwen3.6 IQ2_M = 96.0 / 71.4, K2 = 95.5 / 73.6.
# Kill: GSM8K or MMLU-Pro more than 2 sigma under those rows, or < 30 t/s. Pass => an hq1 arm (hard sets) decides.
# Self-reported numbers (GSM8K 88, MATH 77) are one author's, unverified.
set -u
M=/ai/models; COLD=/mnt/md0/models-cold; B=/ai/src/llama.cpp-mainline/build75
URL=https://huggingface.co/mradermacher/Whittle-Qwen-3.8-35B-A3B-i1-GGUF/resolve/main/Whittle-Qwen-3.8-35B-A3B.i1-Q2_K.gguf
F=$M/Whittle-Qwen-3.8-35B-A3B.i1-Q2_K.gguf; SIZE=12968342080
f=Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf    # S1 closed: IQ3_M stays the Gemma default
if [ -f $M/$f ] && [ ! -L $M/$f ]; then s0=$(stat -c %s $M/$f); mv $M/$f $COLD/$f && [ "$(stat -c %s $COLD/$f)" = "$s0" ] && ln -s $COLD/$f $M/$f && echo "moved to cold + symlinked: $f" || { echo "WHITTLE1_FAILED: move of $f"; exit 1; }; fi
if [ ! -f $F ] || [ "$(stat -c %s $F)" != "$SIZE" ]; then
  free=$(df --output=avail -B1 $M | tail -1); [ $free -gt $((SIZE + 12000000000)) ] || { echo "WHITTLE1_FAILED: $((free / 1000000000)) GB free, not enough room"; exit 1; }
  for try in 1 2 3 4 5; do curl -sSfL -C - --retry 5 --retry-delay 10 -o $F.part "$URL"; [ "$(stat -c %s $F.part 2>/dev/null)" = "$SIZE" ] && break; echo "  try $try: have $(stat -c %s $F.part 2>/dev/null) of $SIZE"; sleep 20; done
  [ "$(stat -c %s $F.part)" = "$SIZE" ] || { echo "WHITTLE1_FAILED: size mismatch"; exit 1; }
  mv $F.part $F
fi
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
# tensors over 400 MiB that are not fused experts (the n-gram memory, embeddings, output) stay on the CPU: names from the file itself
OT=$(PYTHONPATH=/ai/src/llama.cpp-mainline/gguf-py /ai/.venv/bin/python - $F <<'PY'
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
big = [(t.name, int(t.n_bytes)) for t in r.tensors if t.n_bytes > 400 * 2**20 and "_exps" not in t.name]
for n, b in big: print(f"    big tensor: {n} {b / 2**20:.0f} MiB", file=sys.stderr)
arch = [f for k, f in r.fields.items() if k == "general.architecture"]
print("    arch:", bytes(arch[0].parts[-1]).decode() if arch else "?", "| tensors:", len(r.tensors), file=sys.stderr)
print("|".join(n.replace(".", r"\.") for n, _ in big))
PY
) || { echo "WHITTLE1_FAILED: GGUF unreadable"; exit 1; }
OTARG=(-ot "exps=CPU"); [ -n "$OT" ] && OTARG+=(-ot "^($OT)\$=CPU")
echo "placement: ${OTARG[*]}"
export BUILD=$B EDIT=1 LONG=1 GEN=300
for C in 0 16; do
  L=whittle_q2k_c$C; echo "##### $L"
  MODEL=$F OFFLOAD=32 ./specbench.sh 999 $L "${OTARG[@]}" --moe-expert-cache $C -ub 128 -b 256 2>&1 | grep -vE "^\s*$" | cut -c1-230
  grep -hE "MoE expert cache enabled|moe-cache: steps|out of memory|error loading|unknown model architecture|failed" server_$L.log | sort -u | tail -4 | cut -c1-230 | sed 's/^/    /'
done
grep -qE "tok \|" whittle1.log 2>/dev/null || true
export MMLU_CAP=2048 QARGS="--data data_h2 --sets gsm8k,mmlu_pro"
MODEL=$F ./qualbench.sh whittle_q2k "${OTARG[@]}" --moe-expert-cache 16
echo WHITTLE1_DONE
