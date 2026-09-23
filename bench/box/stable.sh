#!/bin/bash
# shellcheck disable=SC2034,SC2153  # STABLE_* are read by the sourcing job script; the eval defines STABLE_MODEL etc.
# Source me: the STABLE configuration from stable.env (repo stable/stable.env, deployed to /ai/bench/stable.env), so a job's
# STABLE arm can never lag behind a promotion. Every stable.env KEY becomes STABLE_KEY (no clash with a job's own MODEL etc.).
#   stable_export_env        export STABLE's GGML_* / LLAMA_* switches that the caller has NOT already set (an arm overrides by
#                            exporting its own value first); includes LLAMA_MTP_VOCAB_FILE
#   stable_args CTX          llama-server flags for context CTX, without -m / -c (specbench sets those): -md HEAD + SERVE_ARGS +
#                            SERVE_CTX_<CTX>; append an arm's own flags AFTER these (llama.cpp keeps the last value of a flag)
# Also sets STABLE_BUILD, STABLE_MODEL_PATH, STABLE_HEAD_PATH, STABLE_VOCAB_PATH.
STABLE_ENV_FILE=${STABLE_ENV_FILE:-/ai/bench/stable.env}
STABLE_MODELS=${STABLE_MODELS:-/ai/models}
[ -r "$STABLE_ENV_FILE" ] || { echo "stable.sh: $STABLE_ENV_FILE missing (deploy stable/stable.env)" >&2; return 1; }
eval "$(sed -n 's/^\([A-Z0-9_]*\)=\(".*"\)$/STABLE_\1=\2/p' "$STABLE_ENV_FILE")"
STABLE_BUILD=$STABLE_STABLE_BOX_BUILD
STABLE_MODEL_PATH=$STABLE_MODELS/$STABLE_MODEL
STABLE_HEAD_PATH=$STABLE_MODELS/$STABLE_MTP_HEAD
STABLE_VOCAB_PATH=$STABLE_MODELS/$STABLE_MTP_VOCAB

stable_export_env() {
    local kv name
    for kv in $STABLE_SERVE_ENV "LLAMA_MTP_VOCAB_FILE=$STABLE_VOCAB_PATH"; do
        name=${kv%%=*}
        [ -n "${!name+set}" ] || export "${kv?}"
    done
}

stable_args() {
    local var="STABLE_SERVE_CTX_$1"
    [ -n "${!var:-}" ] || { echo "stable.sh: no STABLE setting for -c $1" >&2; return 1; }
    echo "-md $STABLE_HEAD_PATH $STABLE_SERVE_ARGS ${!var}"
}

# stable_server CTX LABEL [arm flags...]: start STABLE llama-server on :8099 in the background (log server_LABEL.log in
# STABLE_LOGDIR, pid in STABLE_PID); MODEL / BUILD in the caller's environment override the STABLE files. The caller waits for
# /health as usual. Exports LEDGER_STABLE so the ledger records which STABLE the arm was built on.
stable_server() {
    local ctx=$1 label=$2 args
    shift 2
    args=$(stable_args "$ctx") || return 1
    export LEDGER_STABLE=$STABLE_STABLE_SINCE
    # shellcheck disable=SC2086  # args is a flag list
    ( stable_export_env; exec env -u LD_LIBRARY_PATH "${BUILD:-$STABLE_BUILD}/bin/llama-server" -m "${MODEL:-$STABLE_MODEL_PATH}" \
        -c "$ctx" --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 $args "$@" ) \
        > "${STABLE_LOGDIR:-/ai/bench}/server_$label.log" 2>&1 &
    STABLE_PID=$!
}
