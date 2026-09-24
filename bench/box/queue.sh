#!/bin/bash
# queue.sh -- durable, file-backed job queue for the i5 box. One job at a time (the box is a serial machine).
# State lives in $QDIR/queue.tsv (tab-separated: id, status, tries, started, ended, rc, cmd) and survives reboots,
# tmux deaths and systemd-oomd; the runner is ai-queue.service (system.slice, so oomd's user-slice pressure kills miss it).
# The service is enabled at boot (resumes the queue after a poweroff) but has Restart=no, so it never respawns after a crash.
#   systemctl start|stop|disable ai-queue   (status: queue.sh list; journal: journalctl -u ai-queue; log: queue.log)
# A clean stop / shutdown pauses the running job (back to pending, the try is not counted); a job left `running` by a crash or
# power loss is requeued on the next start, up to MAX_TRIES crashed starts.
#   queue.sh add ID 'CMD'     append a pending job (CMD runs under bash -c in $QDIR; rc 0 = done, else failed)
#   queue.sh list             show the queue
#   queue.sh retry ID | skip ID
#   queue.sh run              the runner loop (started by systemd)
# A job only starts once the box has been idle for two checks 30 s apart: no llama-server / quality client / build /
# download, no foreign /ai/bench/*.sh script alive, AND nothing loaded on the GPU (nvidia-smi lists no compute process and
# VRAM in use is under GPU_IDLE_MIB). That is also how it hands over from jobs started by hand in tmux.
QDIR=${QDIR:-/ai/bench}
Q=$QDIR/queue.tsv; LOG=$QDIR/queue.log; LOCK=$QDIR/queue.lock
BUSY_RE=${BUSY_RE:-'llama-server|llama-bench|qual\.py|specclient\.py|nvcc|cmake --build|hf download|huggingface-cli|wget |/ai/bench/[a-z0-9_]+\.sh'}
MAX_TRIES=${MAX_TRIES:-3}; IDLE_SLEEP=${IDLE_SLEEP:-30}; POLL=${POLL:-60}
now() { date '+%F %T'; }
log() { echo "$(now) $*" >> "$LOG"; }
locked() { ( flock 9; "$@" ) 9>"$LOCK"; }
_set() { # id field(2=status 3=tries 4=started 5=ended 6=rc) value
    awk -F'\t' -v OFS='\t' -v id="$1" -v f="$2" -v v="$3" '$1==id{$f=v} {print}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"; }
_get() { awk -F'\t' -v id="$1" -v f="$2" '$1==id{print $f}' "$Q"; }
GPU_CHECK=${GPU_CHECK:-1}; GPU_IDLE_MIB=${GPU_IDLE_MIB:-200}
gpu_busy() { # busy if any compute process holds the GPU, if VRAM in use is above the idle floor, or if nvidia-smi itself fails
    [ "$GPU_CHECK" = 1 ] || return 1
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null) || { log "nvidia-smi failed: treating the GPU as busy"; return 0; }
    [ -n "$apps" ] && return 0
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -gt "$GPU_IDLE_MIB" ] && return 0
    return 1; }
# wrapper shells (ssh one-liners, tmux "bash -c" waiters) only QUOTE the pattern in their command line, and watch.sh only waits on
# a row; real work shows up as its own process
busy() { pgrep -af "$BUSY_RE" | grep -vE "^[0-9]+ (/usr)?(/bin/)?(bash|sh|zsh|timeout|ssh|sshd:)( |$).*-c |queue\.sh|watch\.sh|pgrep" | grep -q . && return 0; gpu_busy; }

case "$1" in
add)
    [ -n "$2" ] && [ -n "$3" ] || { echo "usage: queue.sh add ID 'CMD'"; exit 2; }
    touch "$Q"; [ -n "$(_get "$2" 1)" ] && { echo "id $2 already queued"; exit 1; }
    locked bash -c "printf '%s\tpending\t0\t-\t-\t-\t%s\n' \"\$1\" \"\$2\" >> '$Q'" _ "$2" "$3"; log "ADD $2" ;;
list)
    touch "$Q"; awk -F'\t' '{printf "%-16s %-8s tries=%s  %s -> %s  rc=%s\n    %s\n", $1, $2, $3, $4, $5, $6, $7}' "$Q" ;;
retry) locked _set "$2" 2 pending; locked _set "$2" 3 0; log "RETRY $2" ;;
skip)  locked _set "$2" 2 skipped; log "SKIP $2" ;;
run)
    touch "$Q"; log "RUNNER START pid $$"
    # systemctl stop (and a shutdown) TERMs the whole control group: the job dies with rc != 0. bash runs this trap as soon as
    # the killed job returns, before the row is touched. A CLEAN stop is a pause, not a failure: the job goes back to pending
    # WITHOUT spending a try (long jobs resume per work unit and must survive any number of nightly downtimes). Only a runner
    # that dies without this trap (crash, SIGKILL, power loss) leaves the row `running`, which the next start counts below.
    id=""
    trap 'if [ -n "$id" ] && [ "$(_get "$id" 2)" = running ]; then locked _set "$id" 2 pending; locked _set "$id" 3 $(( $(_get "$id" 3) - 1 )); log "PAUSE $id (runner stopped by signal; not counted as a try)"; else log "RUNNER STOP (signal)"; fi; exit 0' TERM INT
    # a job left in 'running' means the runner died under it (reboot, kill): requeue it, bounded by MAX_TRIES
    for id in $(awk -F'\t' '$2=="running"{print $1}' "$Q"); do
        if [ "$(_get "$id" 3)" -lt "$MAX_TRIES" ]; then locked _set "$id" 2 pending; log "REQUEUE $id (interrupted)"
        else locked _set "$id" 2 failed; locked _set "$id" 6 interrupted; log "FAILED $id (interrupted, out of tries)"; fi
    done
    while true; do
        id=$(awk -F'\t' '$2=="pending"{print $1; exit}' "$Q")
        [ -z "$id" ] && { sleep "$POLL"; continue; }
        idle=0; while [ $idle -lt 2 ]; do if busy; then idle=0; else idle=$((idle+1)); fi; sleep "$IDLE_SLEEP"; done
        [ "$(_get "$id" 2)" = pending ] || continue   # skipped or edited while we waited
        cmd=$(_get "$id" 7)
        locked _set "$id" 2 running; locked _set "$id" 3 $(( $(_get "$id" 3) + 1 )); locked _set "$id" 4 "$(now)"
        log "START $id: $cmd"
        ( cd "$QDIR" && bash -c "$cmd" ); rc=$?
        locked _set "$id" 5 "$(now)"; locked _set "$id" 6 "$rc"
        if [ $rc -eq 0 ]; then locked _set "$id" 2 done; else locked _set "$id" 2 failed; fi
        log "END $id rc=$rc"
    done ;;
*) echo "usage: queue.sh add|list|retry|skip|run"; exit 2 ;;
esac
