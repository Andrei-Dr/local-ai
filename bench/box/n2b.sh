#!/bin/bash
# N2B: N2-lite n-gram cap repeat with NOISE CONTROL (M1). Cap 3 vs cap 2 was +4.5% ALL / +9% edit at n=1 —
# inside the 3-5% identical-run jitter we measure, so it may have been nothing. Hypothesis: cap 3 (n_rs_seq 3,
# one slot less) is genuinely faster on repetition workloads; verified here at REPEATS=3 WARMUP=1 per server
# start, arms interleaved a/b/a/b (ABAB) so drift hits both sides. Same binaries as p4c1's N2 rows, one flag apart.
# Kill: ALL delta <+2% at 3x3 samples => the N2-lite win was noise, cap stays at 2, close N2.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=300 REPEATS=3 WARMUP=1
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
QN="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --spec-ngram-mod-n-min 2"
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "n_rs_seq +=|RS buffer size|MoE expert cache (enabled|prompt)|moe-cache: (steps|prompt)|draft acceptance|statistics +(ngram|draft)" server_$l.log | sort -u | tail -12 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
for i in a b; do
  run $Q n2b_q36_c30_ngmod2_$i -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 $QN --spec-ngram-mod-n-max 2
  run $Q n2b_q36_c29_ngmod3_$i -ot exps=CPU --moe-expert-cache 29 -ub 128 -b 256 $QN --spec-ngram-mod-n-max 3
done
echo "--- temp-0 text identity between caps (divergence = greedy tie-break from different accepted drafts, not an error)"
for i in a b; do python3 /ai/bench/textdiff.py runs/n2b_q36_c30_ngmod2_$i.client.json runs/n2b_q36_c29_ngmod3_$i.client.json | sed "s/^/    $i /"; done
echo "--- per-run spread is in the ledger rows (*_runs lists); judge deltas with: python3 /ai/bench/abreport.py ledger.jsonl '^n2b_q36_c30_ngmod2_' '^n2b_q36_c29_ngmod3_' --metric decode_tps"
echo N2B_DONE
