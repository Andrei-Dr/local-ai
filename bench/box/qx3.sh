#!/bin/bash
# QX3: is a HOT/COLD per-expert precision split worth runtime C++? Priced by KL divergence before anyone writes loader code.
# Facts it stands on: both served ~2.5-bit files sit at mean KLD 0.218 vs Q6 (1 token in 5 changes its argmax; kld1), hq1 shows
# what that costs on hard reasoning, and routing is concentrated: the top 25% of experts carry 72-78% of the routing mass. RAM is
# the binding resource (16 GB), so precision should follow heat: hot experts at ~4.5 bpw, cold ones stay at ~2.6.
#   A. X4 = the same recipe as K2 with Q4_K experts (~19.8 GB, streams from NVMe like the Q6 reference did) -> KLD = the CEILING
#      (what all-4-bit experts would buy = also the price tag of RAM1, the 32 GB upgrade).
#   B. hot sets from the code + prose routing traces (top 25% and top 50% per layer), expert_mix.py: HI = X4, LO = K2, Q8_0
#      container (~36 GB each => /mnt/md0, read from there; slow, unattended) -> KLD per mix.
# Read: mean / 99% KLD and same-top-token of K2 (0.2181 / 79.9%) -> mix25 -> mix50 -> X4. GO for the runtime split (two expert
# tensors per layer + id remap) only if mix25 cuts mean KLD by >= 25% (SPEC QX); if X4 itself is not far below K2, QX3 is dead.
# Caveat: wikitext is prose and the hot set is workload-specific (code vs prose overlap = chance); a code text is the follow-up.
source /ai/bench/preflight.sh || exit 1
D=/ai/src/llama.cpp-mainline; B=$D/build75; M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K=/ai/bench/kld; S=/mnt/md0/qx3; T=/ai/bench/traces; PY=/ai/.venv/bin/python
export PYTHONPATH=$D/gguf-py   # expert_mix.py needs gguf-py; the venv has numpy only
SRC=$M/$N-Q6_K_P.gguf; K2=$M/$N-K2-expQ2K-downQ3K.gguf; IM=$M/imatrix-$N-from-IQ2_M.dat; X4=$M/$N-X4-expQ4K.gguf
for f in $SRC $K2 $IM $K/q6.kld $T/q36_code_tok.bin $T/q36_prose_tok.bin /ai/bench/expert_mix.py; do [ -s $f ] || { echo "QX3_REFUSED: $f missing"; exit 1; }; done
mkdir -p $S; cd $K || exit 1
CH=${CH:-40}
PP="$B/bin/llama-perplexity -f $K/wiki.test.raw -c 512 -b 512 -ub 512 --chunks $CH -ngl 999 -ot exps=CPU -t 6 -fa on"
stats() { grep -E "Mean +KLD|Maximum KLD|99\.0% +KLD|99\.9% +KLD|Median +KLD|Same top p|Mean PPL\(Q\)|PPL\(Q\)/PPL\(base\)|out of memory|error" "$1" | cut -c1-160 | sed 's/^/    /'; }
# Resume per arm: a finished arm leaves qx3_kld_LABEL.done holding the key of its INPUTS (size + mtime of every file it was built
# from, content hashes of the small ones, the chunk count); a restart with the same key reprints the measured stats instead of re-running.
key() { # small files (hot sets, the mix tool) by CONTENT (the hot set is regenerated every run), model files by size + mtime
  for f in "$@"; do
    if [ ! -f "$f" ]; then echo "$f"
    elif [ "$(stat -Lc %s "$f")" -lt 10000000 ]; then echo "$f $(sha256sum < "$f")"
    else stat -Lc '%n %s %Y' "$f"; fi
  done | sha256sum | cut -c1-16; }
arm_done() { [ -f qx3_kld_$1.done ] && [ "$(cat qx3_kld_$1.done)" = "$2" ] && grep -q "Same top p" qx3_kld_$1.log 2>/dev/null; }
kld() { # label file key
  echo "##### qx3_$1 | $(date +%T) | $(ls -laL $2 | awk '{print $5}') bytes"
  rm -f qx3_kld_$1.done
  GGML_OP_OFFLOAD_MIN_BATCH=32 $PP -m $2 --kl-divergence-base $K/q6.kld --kl-divergence > qx3_kld_$1.log 2>&1; stats qx3_kld_$1.log
  grep -q "Same top p" qx3_kld_$1.log && echo "$3" > qx3_kld_$1.done
}
if [ ! -s $X4 ]; then
  free=$(df -BG --output=avail $M | tail -1 | tr -dc 0-9); [ "$free" -ge 30 ] || { echo "QX3_DISK_REFUSED: ${free}G free on $M, need 30G"; exit 1; }
  echo "##### quantize X4 | $(date +%T)"
  $B/bin/llama-quantize --allow-requantize --imatrix $IM --tensor-type ffn_gate_exps=q4_k --tensor-type ffn_up_exps=q4_k --tensor-type ffn_down_exps=q4_k \
    $SRC $X4.part IQ2_M 6 > /ai/bench/quantize_qx3.log 2>&1 || { tail -8 /ai/bench/quantize_qx3.log; echo QX3_QUANTIZE_FAILED; exit 1; }
  mv $X4.part $X4; echo "    $(ls -la $X4 | awk '{print $5}') bytes | $(grep -E "model size|quant size" /ai/bench/quantize_qx3.log | tr '\n' ' ')"
fi
KX4=$(key $X4 $K/q6.kld CH=$CH)
if arm_done X4 $KX4; then echo "##### qx3_X4 | resumed: measured earlier on the same inputs"; stats qx3_kld_X4.log; else kld X4 $X4 $KX4; fi
for F in 25 50; do
  HOT=$S/hot$F.json; MIX=$S/mix.gguf   # one scratch file, overwritten per arm (36 GB each)
  $PY /ai/bench/expert_mix.py hot --trace $T/q36_code_tok.bin --trace $T/q36_prose_tok.bin --frac 0.$F --out $HOT || { echo "QX3_FAILED: hot set $F"; exit 1; }
  KM=$(key $X4 $K2 $HOT $K/q6.kld CH=$CH /ai/bench/expert_mix.py)
  if arm_done mix$F $KM; then echo "##### qx3_mix$F | resumed: measured earlier on the same inputs"; stats qx3_kld_mix$F.log; continue; fi
  echo "##### mix$F | $(date +%T) | $($PY /ai/bench/expert_mix.py --hi $X4 --lo $K2 --hot $HOT --out $MIX --dry-run 2>&1 | grep -iE "effective|bits per" | tail -1 | cut -c1-160)"
  freeS=$(df -BG --output=avail $S | tail -1 | tr -dc 0-9); [ "$freeS" -ge 60 ] || { echo "QX3_DISK_REFUSED: ${freeS}G free on $S"; exit 1; }
  $PY /ai/bench/expert_mix.py --hi $X4 --lo $K2 --hot $HOT --out $MIX > /ai/bench/qx3_mix$F.log 2>&1 || { tail -5 /ai/bench/qx3_mix$F.log; echo "QX3_FAILED: mix $F"; exit 1; }
  kld mix$F $MIX $KM
done
echo "    reference rows (kld1): IQ2_M mean KLD 0.2188 same-top 79.8% | K2 0.2181 / 79.9%"
echo QX3_DONE
