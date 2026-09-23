#!/bin/bash
# Build STABLE llama-server: upstream llama.cpp at LLAMA_CPP_BASE + research/patches/mainline-series, with CMAKE_FLAGS
# (all from stable/stable.env). Verifies the patched source tree equals LLAMA_CPP_STABLE_TREE before building.
# usage: stable/build.sh [DIR]    (default DIR: ./llama.cpp next to this repo's root; an existing clone is reused)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/stable/stable.env"
DIR=${1:-$ROOT/llama.cpp}
[ -d "$DIR/.git" ] || git clone "$LLAMA_CPP_REPO" "$DIR"
cd "$DIR"
git fetch -q origin 2>/dev/null || true
git checkout -q --detach "$LLAMA_CPP_BASE"
git -c user.name=stable -c user.email=stable@localhost am -q "$ROOT"/research/patches/mainline-series/*.patch
tree=$(git rev-parse "HEAD^{tree}")
[ "$tree" = "$LLAMA_CPP_STABLE_TREE" ] || { echo "source tree $tree != STABLE $LLAMA_CPP_STABLE_TREE"; exit 1; }
echo "source tree matches STABLE ($tree)"
# shellcheck disable=SC2086  # CMAKE_FLAGS is a flag list
cmake -B build $CMAKE_FLAGS
cmake --build build -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" --target llama-server
echo "built: $DIR/build/bin/llama-server"
