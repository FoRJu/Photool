#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "===================================="
echo " Photool - Stopping..."
echo "===================================="
echo

# Try PID file first
if [ -f .pids ]; then
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PID $pid..."
            kill "$pid" 2>/dev/null
        fi
    done < .pids
    rm -f .pids
    echo "Done."
    exit 0
fi

# Fallback: kill by port
BACKEND_PID=$(lsof -ti :8000 2>/dev/null || true)
FRONTEND_PID=$(lsof -ti :5173 2>/dev/null || true)

if [ -n "$BACKEND_PID" ]; then
    echo "[1/2] Stopping backend (PID $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
else
    echo "[1/2] Backend not running."
fi

if [ -n "$FRONTEND_PID" ]; then
    echo "[2/2] Stopping frontend (PID $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null
else
    echo "[2/2] Frontend not running."
fi

echo
echo "Done."
