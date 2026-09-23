#!/bin/bash
# PROF18: mechanism check for 0018 (GGML_SCHED_NO_COPY_SYNC=1: no host ggml_backend_synchronize before a split input copy that is
# already ordered on the GPU stream). Speed was inconclusive: tu1 +3.4% (kcompactd-contaminated), tu2 -2.0% with 2x wider spread.
# Question: does the per-layer host sync actually disappear, and where does the time go instead (a later sync, a longer CPU
# phase, a longer graph launch)? Build: the promotion candidate /ai/src/llama.cpp-v2 (promo1), serving config.
# Per arm (sync = default, nosync = 0018 on), two captures each (a/b) for the noise floor: nsys API trace (graphs on) of one
# 300-token decode, then (1) hand1_phases.py per-layer phases, (2) per decode token: count + total time of cudaStreamSynchronize /
# cudaDeviceSynchronize / cudaStreamWaitEvent / cudaMemcpyAsync on the graph thread, (3) the capture's decode t/s.
# Read: 0018 is WORTH KEEPING only if nosync's per-layer round (D launch -> next D launch) is shorter than sync's by more than the
# a/b difference; if the sync merely moves (same round time), it is dead and gets dropped from the series.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
NEW=/ai/src/llama.cpp-v2/build75
strings $NEW/bin/libggml-base.so* 2>/dev/null | grep -q GGML_SCHED_NO_COPY_SYNC || { echo "PROF18_REFUSED: v2 build missing (promo1 first)"; exit 1; }
ARGS="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
cap() { # label envs
  local l=$1 envs=$2
  rm -f prof18_$l.nsys-rep prof18_$l.qdstrm prof18_$l.sqlite
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none \
    --cpuctxsw=none -f true -o /ai/bench/prof18_$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none \
    --jinja --parallel 1 --port 8099 --cache-ram 0 $ARGS > server_prof18_$l.log 2>&1 &
  local NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  EDIT=0 LONG=0 OUT=/tmp python3 specclient.py prof18_$l x 2>&1 | grep -E "decode" | head -1 | cut -c1-120 | sed "s/^/    [$l] /"
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s prof18_$l.nsys-rep ] || /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i prof18_$l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output prof18_$l.sqlite prof18_$l.nsys-rep > /dev/null 2>&1
  echo "  $l phases:"; $PY hand1_phases.py prof18_$l.sqlite --skip-s 0 2>&1 | sed 's/^/    /'
  $PY - prof18_$l.sqlite <<'PY' | sed 's/^/    /'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
tid = c.execute("""select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
    where s.value like 'cudaGraphLaunch%' group by globalTid order by count(*) desc limit 1""").fetchone()[0]
n_launch = c.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
    where r.globalTid=? and s.value like 'cudaGraphLaunch%'""", (tid,)).fetchone()[0]
print(f"graph thread: {n_launch} graph launches")
for name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaStreamWaitEvent", "cudaEventSynchronize", "cudaMemcpyAsync"):
    r = c.execute("""select count(*), coalesce(sum(r.end - r.start), 0) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s
        on s.id=r.nameId where r.globalTid=? and s.value like ?""", (tid, name + "%")).fetchone()
    print(f"{name:24s} calls {r[0]:7d} ({r[0] / max(n_launch, 1):5.2f} per launch) | total {r[1] / 1e6:9.1f} ms | mean {r[1] / max(r[0], 1) / 1e3:7.1f} us")
PY
}
for r in a b; do
  cap sync_$r   ""
  cap nosync_$r "GGML_SCHED_NO_COPY_SYNC=1"
done
rm -f prof18_*.qdstrm
echo PROF18_DONE
