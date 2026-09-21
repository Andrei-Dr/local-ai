#!/bin/bash
# watch.sh -- the ONE durable job/queue waiter for the i5 box. Exits on success/drain, failure, stall, OR timeout --
# never silent on death (see feedback_death_aware_waiters: a marker-only loop left the box idle 6h once).
# Two modes:
#   watch.sh queue [TIMEOUT_MIN]            -- exit when queue.tsv has 0 pending AND 0 running (whole queue drained),
#                                              or the service died with work still pending (stall), or a job failed.
#                                              Order/count-agnostic: survives reorders and added jobs.
#   watch.sh job LOG DONE_RE PROC_RE [MIN]  -- exit when DONE_RE hits LOG, or a failure marker hits it, or no PROC_RE
#                                              process is alive for ~60s, or timeout.
# Always prints WHY it exited + tail(log) + free mem + recent OOM/oomd journal lines. Poll 30s. Meant to be run
# under ssh from a local one-shot background task: the local side just relays this script's stdout.
set -u
mode=${1:-queue}; cd /ai/bench || exit 9
FAIL_RE='_FAILED|_REFUSED|GIVING UP|CTX1_REFUSED|out of memory'
diag() { echo "--- free ---"; free -m | awk '/Mem:/{print "  mem_avail="$7"MiB"}'; echo "--- recent OOM/oomd ---"; journalctl -k --since "-30min" 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" | tail -3; journalctl -u ai-queue --since "-30min" 2>/dev/null | tail -2; }
case "$mode" in
queue)
  tmin=${2:-960}; end=$(( $(date +%s) + tmin*60 ))
  while :; do
    p=$(cut -f2 queue.tsv 2>/dev/null | grep -c pending)
    r=$(cut -f2 queue.tsv 2>/dev/null | grep -c running)
    act=$(systemctl is-active ai-queue 2>/dev/null)
    done=$(awk -F"\t" '$2=="done"{print $1}' queue.tsv | tr "\n" " ")
    failed=$(awk -F"\t" '$2=="failed"{print $1}' queue.tsv | tr "\n" " ")
    bad=$(grep -hE "$FAIL_RE" ./*.log 2>/dev/null | tail -3)
    if [ "${p:-1}" -eq 0 ] && [ "${r:-1}" -eq 0 ]; then echo "EXIT queue: DRAINED. done=[$done] failed=[$failed]"; [ -n "$bad" ] && echo "badlog: $bad"; diag; exit 0; fi
    if [ "$act" != active ] && [ "${p:-0}" -gt 0 ]; then echo "EXIT queue: STALLED (service=$act, $p pending, $r running). done=[$done] failed=[$failed]"; diag; exit 2; fi
    if [ -n "$failed" ]; then echo "EXIT queue: JOB FAILED [$failed]. done=[$done]"; echo "badlog: $bad"; diag; exit 3; fi
    if [ $(date +%s) -ge $end ]; then echo "EXIT queue: TIMEOUT ${tmin}min. pending=$p running=$r done=[$done] failed=[$failed]"; diag; exit 4; fi
    sleep 30
  done ;;
job)
  log=$2; done_re=$3; proc_re=$4; tmin=${5:-180}; t0=$(date +%s); end=$(( t0 + tmin*60 )); gone=0
  # A log left over from an EARLIER run still holds that run's markers (queue jobs truncate the log only when they start):
  # markers count only once the log has been written after this watcher started.
  fresh() { [ "$(stat -c %Y "$log" 2>/dev/null || echo 0)" -ge "$t0" ]; }
  while :; do
    if ! fresh; then
      if [ $(date +%s) -ge $end ]; then echo "EXIT job: TIMEOUT ${tmin}min ($log never written after the watcher started)"; diag; exit 4; fi
      # the queue runner waits ~1-2 min for an idle box before it starts a job: no verdict on a stale log in the first 5 min
      [ $(( $(date +%s) - t0 )) -lt 300 ] || pgrep -af "$proc_re" | grep -vE "watch\.sh|pgrep|grep" | grep -q . || { echo "EXIT job: PROCESS GONE ($proc_re) and $log is stale"; tail -4 "$log" 2>/dev/null; diag; exit 5; }
      sleep 30; continue
    fi
    if grep -qE "$done_re" "$log" 2>/dev/null; then echo "EXIT job: DONE ($done_re in $log)"; tail -6 "$log"; exit 0; fi
    if grep -qE "$FAIL_RE" "$log" 2>/dev/null; then echo "EXIT job: FAILURE marker in $log"; grep -E "$FAIL_RE" "$log" | tail -3; diag; exit 3; fi
    if pgrep -af "$proc_re" | grep -vE "watch\.sh|pgrep|grep" | grep -q .; then gone=0; else gone=$((gone+1)); fi
    if [ $gone -ge 2 ]; then echo "EXIT job: PROCESS GONE ($proc_re absent ~60s) without DONE"; tail -8 "$log"; diag; exit 5; fi
    if [ $(date +%s) -ge $end ]; then echo "EXIT job: TIMEOUT ${tmin}min"; tail -6 "$log"; diag; exit 4; fi
    sleep 30
  done ;;
*) echo "usage: watch.sh queue [MIN] | watch.sh job LOG DONE_RE PROC_RE [MIN]"; exit 9 ;;
esac
