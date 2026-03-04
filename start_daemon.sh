#!/bin/bash
# Start the PKA daemon on localhost:8741
# Usage: ./start_daemon.sh [--reload]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RELOAD_FLAG=""
if [[ "${1:-}" == "--reload" ]]; then
    RELOAD_FLAG="--reload"
fi

echo "Starting PKA daemon on http://127.0.0.1:8741 ..."
exec uvicorn daemon.server:app --host 127.0.0.1 --port 8741 $RELOAD_FLAG
