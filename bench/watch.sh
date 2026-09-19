#!/bin/bash
# watch.sh LOG DONE_REGEX PROC_REGEX [TIMEOUT_MIN] -- wait on a box job WITHOUT ever waiting forever:
# exits on the done/fail marker, when no matching process has been alive for ~60 s, or on timeout. Always prints why.
LOG=$1; DONE=$2; PROC=$3; TMO=${4:-120}; dead=0; t0=$(date +%s)
while :; do
  # remote side: marker count (0 when the log does not exist yet), then live processes EXCLUDING the remote shell itself,
  # whose command line contains the pattern
  out=$(ssh -o ConnectTimeout=20 root@i5.local "h=\$(grep -cE '$DONE' $LOG 2>/dev/null); echo \${h:-0}; pgrep -f '$PROC' | grep -vx \$\$ | wc -l" 2>/dev/null) || { sleep 30; continue; }
  hits=$(sed -n 1p <<<"$out"); procs=$(sed -n 2p <<<"$out")
  [ "${hits:-0}" -gt 0 ] && { echo "WATCH: marker matched in $LOG"; break; }
  if [ "${procs:-0}" -eq 0 ]; then dead=$((dead+1)); else dead=0; fi
  [ $dead -ge 2 ] && { echo "WATCH: JOB DIED - no process matching '$PROC' and no marker in $LOG"; break; }
  [ $(( ($(date +%s) - t0) / 60 )) -ge $TMO ] && { echo "WATCH: TIMEOUT after $TMO min"; break; }
  sleep 30
done
ssh -o ConnectTimeout=20 root@i5.local "tail -25 $LOG | cut -c1-220; echo; free -m | sed -n 2,3p; journalctl --since '-10 min' --no-pager 2>/dev/null | grep -iE 'oomd.*Killed|Out of memory' | tail -2"
