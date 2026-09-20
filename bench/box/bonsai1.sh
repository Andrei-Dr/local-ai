#!/bin/bash
# BONSAI1: the accuracy run we dropped. On 2026-09-19 Ternary-Bonsai-2-27B (PrismML QAT ternary of Qwen3.8-27B, PQ2_0 2.13 bpw)
# was written off at 5.15 tok/s ("dense is dead here") and its quality run was killed at GSM8K item 7. Under the accuracy-first
# rule (SPEC 8.x) speed does not settle that: the vendor claims 95% of full precision and AIME 87+ where a conventional 2-bit quant
# of the same model gets 57.5 — i.e. exactly the failure our served ~2.5-bit Qwen3.6 may have. Speed was already tuned to the PCIe
# limit (FFN streamed, MTP n=2, 3.7 GiB per pass at ~10 GB/s), so this job only measures accuracy.
# usage: bonsai1.sh easy | hard
#   easy = the standard sets, thinking off, label bonsai27_pq2 (RESUMES the 9 items of the killed run), ~3.5 h. Paired against the
#          q36 rows on the same ids (bench/qual/paired.py).
#   hard = AIME first 10 (2024-I, a prefix of hq1's set => paired with q36_iq2m_hard), thinking on, -c 16384 (a chain longer than
#          ~16k tokens is cut and counts wrong: read `truncated`), ~5-7 h. Lowest priority: queue it last.
# Build = the PrismML llama.cpp fork (PQ2_0 kernels), config = the best measured placement.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
F=/ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf
[ -e "$F" ] || { echo "BONSAI1_REFUSED: $F missing (cold-storage symlink broken?)"; exit 1; }
export BUILD=/ai/src/llama.cpp/build75
PLACE=(-ngl 99 -ot 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU' -ub 128 -b 256 --spec-type draft-mtp --spec-draft-n-max 2)
case "$1" in
  easy) MMLU_CAP=2048 MODEL=$F OFFLOAD=2 ./qualbench.sh bonsai27_pq2 "${PLACE[@]}" ;;
  hard) grep -q '"aime"' /ai/bench/qual/qual.py || { echo "BONSAI1_REFUSED: deploy the hard-set qual.py first"; exit 1; }
        QARGS="--data data_hard --sets aime --think --limit 10" MODEL=$F OFFLOAD=2 ./qualbench.sh bonsai27_pq2_hard "${PLACE[@]}" -c 16384 ;;
  *) echo "usage: bonsai1.sh easy|hard"; exit 2 ;;
esac
echo "BONSAI1_DONE $1"
