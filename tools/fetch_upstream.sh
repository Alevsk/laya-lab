#!/usr/bin/env bash
# Clone or update the Laya library into ./upstream, pinned to a known-good commit.
set -euo pipefail

REPO_URL="https://github.com/NandhaKishorM/laya.git"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/upstream"
PIN_FILE="$(dirname "$DIR")/UPSTREAM_PIN"
PIN="$(cat "$PIN_FILE" 2>/dev/null || true)"

if [ ! -d "$DIR/.git" ]; then
  echo "cloning $REPO_URL -> upstream/"
  git clone --quiet "$REPO_URL" "$DIR"
else
  echo "upstream/ already present; fetching"
  git -C "$DIR" fetch --quiet origin
fi

if [ -n "$PIN" ]; then
  echo "checking out pinned commit $PIN"
  git -C "$DIR" checkout --quiet "$PIN"
else
  git -C "$DIR" checkout --quiet main 2>/dev/null || true
fi

echo "upstream/ at $(git -C "$DIR" rev-parse --short HEAD) — $(git -C "$DIR" log -1 --format=%s)"
