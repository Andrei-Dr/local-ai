#!/bin/bash
# HAND1 step 1: export the prof2 nsys timeline (short-context decode, expert cache on) to sqlite so the CPU<->GPU handoff gap
# (server 0.78 ms per layer vs 0.49 ms for the bare expert ops) can be analyzed off the box. No server, no GPU work: ~1 min.
set -u
cd /ai/bench; set +u; source ./preflight.sh; set -u
for f in prof2_nsys prof4_gqa1; do
  [ -f $f.nsys-rep ] || { echo "missing $f.nsys-rep"; continue; }
  rm -f $f.sqlite
  nsys export --type sqlite --output $f.sqlite $f.nsys-rep > /dev/null 2>&1 || { echo "HAND1_FAILED: export $f"; exit 1; }
  echo "$f.sqlite $(stat -c %s $f.sqlite) bytes | tables: $(sqlite3 $f.sqlite .tables 2>/dev/null | tr -s ' \n' ' ' | cut -c1-300)"
done
echo HAND1_DONE
