#!/bin/bash
# Run the STABLE llama-server (model files, environment switches and flags from stable/stable.env).
# usage: stable/serve.sh [--ctx 4096|12288] [extra llama-server args...]
# env:   LLAMA_BIN (default ./llama.cpp/build/bin), MODELS (default ./models), MODEL (override the model file name)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_OVERRIDE=${MODEL:-}
source "$ROOT/stable/stable.env"
MODEL=${MODEL_OVERRIDE:-$MODEL}
ctx=$SERVE_CTX_DEFAULT
if [ "${1:-}" = "--ctx" ]; then ctx=$2; shift 2; fi
var="SERVE_CTX_$ctx"
[ -n "${!var:-}" ] || { echo "no STABLE setting for -c $ctx (stable.env has: $(compgen -v SERVE_CTX_ | grep -v DEFAULT | sed 's/SERVE_CTX_//' | tr '\n' ' '))"; exit 2; }
LLAMA_BIN=${LLAMA_BIN:-$ROOT/llama.cpp/build/bin}; MODELS=${MODELS:-$ROOT/models}
for f in "$LLAMA_BIN/llama-server" "$MODELS/$MODEL" "$MODELS/$MTP_HEAD"; do [ -e "$f" ] || { echo "missing: $f (see QUICKSTART.md)"; exit 1; }; done
vocab=(); [ -e "$MODELS/$MTP_VOCAB" ] && vocab=("LLAMA_MTP_VOCAB_FILE=$MODELS/$MTP_VOCAB") || echo "note: $MTP_VOCAB missing, draft uses the full vocabulary (slower, same output)"
# shellcheck disable=SC2086  # SERVE_* are flag lists
exec env $SERVE_ENV ${vocab[@]+"${vocab[@]}"} "$LLAMA_BIN/llama-server" -m "$MODELS/$MODEL" -md "$MODELS/$MTP_HEAD" -c "$ctx" \
  $SERVE_ARGS ${!var} "$@"
