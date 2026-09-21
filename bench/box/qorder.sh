#!/bin/bash
# qorder.sh ID [ID ...]: put the PENDING jobs of /ai/bench/queue.tsv in the given order (pending ids not named keep their relative
# order after the named ones). Rows that are not pending never move. Under queue.lock, with a row-count + id-set check and a backup;
# the runner re-reads the file on every loop, so no restart is needed. This replaces the hand-rolled awk-over-ssh reorders.
set -eu
Q=/ai/bench/queue.tsv; cd /ai/bench
exec 9>queue.lock; flock 9
cp $Q $Q.bak-reorder
python3 - "$@" <<'PY'
import sys
Q = "/ai/bench/queue.tsv"
want = sys.argv[1:]
rows = [l for l in open(Q).read().split("\n") if l]
pend = [r for r in rows if r.split("\t")[1] == "pending"]
ids = [r.split("\t")[0] for r in pend]
missing = [w for w in want if w not in ids]
if missing:
    sys.exit(f"QORDER_REFUSED: not pending: {missing}")
order = want + [i for i in ids if i not in want]
bypid = {r.split("\t")[0]: r for r in pend}
it = iter(order)
out = [bypid[next(it)] if r.split("\t")[1] == "pending" else r for r in rows]
assert len(out) == len(rows) and sorted(out) == sorted(rows), "row set changed"
open(Q + ".new", "w").write("\n".join(out) + "\n")
PY
mv $Q.new $Q
cut -f1,2 $Q | grep -E "running|pending" | tr '\n' ' '; echo
