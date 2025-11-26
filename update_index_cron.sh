#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_PY="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

"$VENV_PY" src/update_index.py >> "$LOG_DIR/cron.log" 2>&1
