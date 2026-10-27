#!/usr/bin/env bash
# Start Photool at login, in the background (macOS):
#   ./startup.sh install    start it now and at every login
#   ./startup.sh remove     stop starting it at login
#   ./startup.sh            say whether it is set up
set -e
cd "$(dirname "$0")"
PY="$HOME/photool-env/venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "[ERROR] Python venv not found. Run ./setup.sh first."
    exit 1
fi
if [ ! -f frontend/dist/index.html ]; then
    echo "Building the page first..."
    (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
fi
exec "$PY" backend/login_item.py "$@"
