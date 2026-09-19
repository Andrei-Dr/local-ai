#!/bin/bash
cd /ai/bench; pkill -x llama-server
try() { # try LABEL "NGL list" args...  -- first ngl that completes a generation wins
  local label=$1 ngls=$2; shift 2
  for ngl in $ngls; do ./specbench.sh $ngl ${label}_ngl$ngl "$@" > /tmp/sb.out 2>&1; pkill -x llama-server
    if grep -q "decode" /tmp/sb.out; then grep -E "tok \||telemetry" /tmp/sb.out; return; fi; done
  echo "$label: no ngl in [$ngls] completed"; }
try mtp_n3        "24 22"    --spec-type draft-mtp --spec-draft-n-max 3
try ngram_mod     "28 26 24" --spec-type ngram-mod
try ngram+mtp_n2  "24 22"    --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2
OMP_WAIT_POLICY=passive try mtp_n2_omppassive "24 22" --spec-type draft-mtp --spec-draft-n-max 2
try mtp_n2_kq8    "26 25 24" --spec-type draft-mtp --spec-draft-n-max 2 -ctk q8_0
echo ROUND2_DONE
