#!/bin/bash
# DL_STOCK: fetch the CONTROL for hq1 — stock Qwen3.6-35B-A3B at the same quant class as the served file (bartowski imatrix IQ2_M,
# 12,964,402,816 bytes). hq1 on the served file (Uncensored-HauhauCS-Aggressive IQ2_M): MATH-L5 57.5% with 16 of 40 chains cut at
# the card's own 32k budget, none of them a loop. Quantization damage or the fine-tune? Same items, same sampler, same seeds on the
# stock weights answers it. Runs as a queue job so nothing downloads or moves files beside a benchmark.
# Room first (Andrei 2026-09-21: mv + symlink): two files whose decisions are closed go to the cold store on /mnt/md0.
set -u
M=/ai/models; COLD=/mnt/md0/models-cold
URL=https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-IQ2_M.gguf
OUT=$M/Qwen3.6-35B-A3B-stock-bartowski-IQ2_M.gguf; SIZE=12964402816
for f in Qwen3.8-35B-A3B-Distill-IQ2_M.gguf Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf; do   # S2: keep Qwen3.6; S1: IQ3_M stays
  [ -L $M/$f ] && { echo "cold already: $f"; continue; }
  [ -f $M/$f ] || { echo "missing, skipped: $f"; continue; }
  s0=$(stat -c %s $M/$f)
  mv $M/$f $COLD/$f && [ "$(stat -c %s $COLD/$f)" = "$s0" ] && ln -s $COLD/$f $M/$f && echo "moved to cold + symlinked: $f ($s0 bytes)" || { echo "DL_STOCK_FAILED: move of $f"; exit 1; }
done
free=$(df --output=avail -B1 $M | tail -1); echo "free on $M: $((free / 1000000000)) GB"
[ -f $OUT ] && [ "$(stat -c %s $OUT)" = "$SIZE" ] && { echo "already complete: $OUT"; echo DL_STOCK_DONE; exit 0; }
[ $free -gt $((SIZE + 15000000000)) ] || { echo "DL_STOCK_FAILED: not enough room"; exit 1; }
for try in 1 2 3 4 5; do
  curl -sSfL -C - --retry 5 --retry-delay 10 -o $OUT.part "$URL"; rc=$?
  [ "$(stat -c %s $OUT.part 2>/dev/null)" = "$SIZE" ] && break
  echo "  try $try: curl rc=$rc, have $(stat -c %s $OUT.part 2>/dev/null) of $SIZE bytes"; sleep 20
done
[ "$(stat -c %s $OUT.part)" = "$SIZE" ] || { echo "DL_STOCK_FAILED: size mismatch"; exit 1; }
mv $OUT.part $OUT
/ai/.venv/bin/python - <<PY || { echo "DL_STOCK_FAILED: not a readable GGUF"; exit 1; }
import struct
f = open("$OUT", "rb"); magic = f.read(4); ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
assert magic == b"GGUF", magic
print(f"GGUF v{ver}: {nt} tensors, {nkv} metadata keys")
PY
echo DL_STOCK_DONE
