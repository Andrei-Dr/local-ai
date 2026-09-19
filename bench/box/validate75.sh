#!/bin/bash
# Queue step 1: validate build75 against the 61-virtual build, then the PTQ1_0 PCIe pre-test (step 4).
cd /ai/bench
FFN="blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU"
MTP=/ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf
PQ=/ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf
PTQ=/ai/models/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf

echo "########## 1a. best Bonsai config on build75 (61-virtual gave 5.24/5.05) ##########"
pkill -x llama-server; sleep 1
BUILD=/ai/src/llama.cpp/build75 ./specbench.sh 99 b75_best -ot "$FFN" -ub 128 -b 256 \
  --spec-type draft-mtp --spec-draft-n-max 2 2>&1 | grep -E "tok \||telemetry|DIED"
pkill -x llama-server; sleep 1

echo "########## 1b. streamed verify ms/pass (61-virtual: 517/563/647/693) ##########"
for b in build build75; do
  printf "  %-8s " $b
  GGML_OP_OFFLOAD_MIN_BATCH=2 /ai/src/llama.cpp/$b/bin/llama-bench -m $PQ -ngl 24 -fa 1 -t 6 \
    -p 2,4,8,16 -n 0 -r 2 --load-mode none -o md 2>&1 | grep -E "^\| q" \
    | sed -E "s/.*\| +(pp[0-9]+) \| +([0-9.]+) .*/\1 \2/" \
    | awk '{n=substr($1,3)+0; printf "b%-2d %4.0fms  ", n, 1000*n/$2} END{print ""}'
done

echo "########## 4-pre. PTQ1_0 vs PQ2_0 streamed verify (18% fewer bytes?) ##########"
for m in $PQ $PTQ; do
  printf "  %-46s " "$(basename $m)"
  GGML_OP_OFFLOAD_MIN_BATCH=2 /ai/src/llama.cpp/build75/bin/llama-bench -m $m -ngl 24 -fa 1 -t 6 \
    -p 2,4,8 -n 8 -r 2 --load-mode none -o md 2>&1 | grep -E "^\| q" \
    | sed -E "s/.*\| +(pp[0-9]+|tg[0-9]+) \| +([0-9.]+) .*/\1 \2/" \
    | awk '{n=substr($1,3)+0; if ($1 ~ /^pp/) printf "b%-2d %4.0fms  ", n, 1000*n/$2; else printf "tg %.2f t/s", $2} END{print ""}'
done
echo VALIDATE75_DONE
