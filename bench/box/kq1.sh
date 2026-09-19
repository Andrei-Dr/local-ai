#!/bin/bash
# KQ1: cheaper CPU misses for Qwen3.6 by changing the EXPERT quant, not the code (SPEC section 4, lever "cheaper miss").
# Ledger fit 2026-09-20: a Qwen MTP n=2 round = 58 ms = 5.4 ms draft + ~21 ms fixed + ~31 ms CPU expert matvecs
# (IQ2_S vec_dot, compute-bound). On Gemma the same move (IQ3_S -> Q2_K/Q4_0 experts) cut the miss cost ~2.7x.
# HauhauCS ships no K-quant that fits 16 GB RAM (Q2_K_P = 13.67 GB of experts), so build one: requantize their
# Q6_K_P with an imatrix computed here. Recipe K2: gate/up Q2_K, down Q3_K (experts ~11.7 GB); non-expert tensors
# follow the stock IQ2_M recipe so the GPU-resident part matches the file we serve today.
# Kill: MemAvailable min < 1.5 GB or swap moving => fall back to K1 (down Q2_K, ~10.7 GB); quality gate = within 1 sigma of IQ2_M.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; D=/ai/src/llama.cpp-mainline; B=$D/build75
N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
SRC=$M/$N-Q6_K_P.gguf; IQ=$M/$N-IQ2_M.gguf; IM=$M/imatrix-$N-from-IQ2_M.dat; K2=$M/$N-K2-expQ2K-downQ3K.gguf
URL=https://huggingface.co/HauhauCS/$N/resolve/main/$N-Q6_K_P.gguf
if [ ! -x $B/bin/llama-quantize ] || [ ! -x $B/bin/llama-imatrix ]; then
  cmake --build $B -j6 --target llama-quantize llama-imatrix > build_kq1.log 2>&1 || { grep -E "error" -A3 build_kq1.log | head; echo KQ1_BUILD_FAILED; exit 1; }
fi
if [ ! -s $K2 ]; then
  if [ ! -s $SRC ]; then
    free=$(df -BG --output=avail /ai | tail -1 | tr -dc 0-9); [ "$free" -ge 50 ] || { echo "KQ1_DISK_REFUSED: ${free}G free, need 50G"; exit 1; }
    want=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}' | tr -d '\r')
    curl -sL -C - --retry 30 --retry-delay 20 -o $SRC.part "$URL"; got=$(stat -c %s $SRC.part 2>/dev/null || echo 0)
    [ "$want" -gt 1000000000 ] && [ "$got" = "$want" ] || { echo "KQ1_DOWNLOAD_INCOMPLETE: got $got want $want (rerun resumes)"; exit 1; }
    mv $SRC.part $SRC; echo "downloaded $SRC ($got bytes)"
  fi
  if [ ! -s $IM ]; then
    cat corpus/code_big.txt corpus/prose_big.txt > corpus/kq_calib.txt
    GGML_OP_OFFLOAD_MIN_BATCH=32 $B/bin/llama-imatrix -m $IQ -f corpus/kq_calib.txt -o $IM -c 512 -b 512 --chunks 200 -ngl 999 -ot "exps=CPU" -t 6 > imatrix_kq1.log 2>&1 || { tail -5 imatrix_kq1.log; echo KQ1_IMATRIX_FAILED; exit 1; }
    echo "imatrix: $(ls -la $IM | awk '{print $5}') bytes, $(grep -ciE "no data|partial" imatrix_kq1.log) partial-data warnings"
  fi
  $B/bin/llama-quantize --allow-requantize --imatrix $IM --tensor-type ffn_gate_exps=q2_k --tensor-type ffn_up_exps=q2_k --tensor-type ffn_down_exps=q3_k $SRC $K2.part IQ2_M 6 > quantize_kq1.log 2>&1 || { tail -8 quantize_kq1.log; echo KQ1_QUANTIZE_FAILED; exit 1; }
  mv $K2.part $K2; echo "quantized: $(ls -la $K2 | awk '{print $5}') bytes | $(grep -E "model size|quant size" quantize_kq1.log | tr '\n' ' ')"
fi
export BUILD=$B EDIT=1 LONG=1 GEN=300
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps|statistics +draft|out of memory" server_$l.log | sort -u | tail -5 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; python3 - "$l" <<'PY'
import json,sys
try:
    t=json.load(open(f"/ai/bench/runs/{sys.argv[1]}.mon.json")); print("    mem:", {k:v for k,v in t.items() if "mem" in k.lower() or "swap" in k.lower()})
except Exception as e: print("    mem: n/a", e)
PY
}
# miss cost with no cache (k=1), IQ2_M reference vs K2 on the same build
run $IQ kq_q36iq2m_off -ot exps=CPU
run $K2 kq_q36k2_off   -ot exps=CPU
# K2 slots are ~13% larger than IQ2_M slots: 26 K2 slots ~= the VRAM of 30 IQ2_M slots
run $K2 kq_q36k2_c26_mtp2 -ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 $QH --spec-draft-n-max 2
run $IQ kq_q36iq2m_c30_mtp2 -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $QH --spec-draft-n-max 2
# cheaper verify tokens may make n=3 pay
run $K2 kq_q36k2_c24_mtp3 -ot exps=CPU --moe-expert-cache 24 -ub 128 -b 256 $QH --spec-draft-n-max 3
# quality gate (full set, MMLU-Pro at cap 2048), comparable to q36_iq2m_cache48
MMLU_CAP=2048 MODEL=$K2 ./qualbench.sh q36_k2_cache26 -ngl 999 -ot "exps=CPU" --moe-expert-cache 26
echo KQ1_DONE
