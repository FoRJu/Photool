#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "===================================="
echo " Photool - Starting (development)..."
echo "===================================="
echo

# The venv lives outside the project folder on purpose (see dev.bat); setup.sh makes it.
VENV="$HOME/photool-env/venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[ERROR] Python venv not found at $VENV. Run ./setup.sh first."
    exit 1
fi

# Check frontend node_modules
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Node modules not found. Run: cd frontend && npm install"
    exit 1
fi

# Start backend
echo "[1/2] Starting backend on http://localhost:8000 ..."
cd backend
source "$VENV/bin/activate"
# --no-proxy-headers: the app reads proxy headers itself, to tell this computer
# from a device relayed through a local proxy (backend/access.py).
uvicorn main:app --reload --port 8000 --no-proxy-headers &
BACKEND_PID=$!
cd ..

sleep 2

# Start frontend
echo "[2/2] Starting frontend on http://localhost:5173 ..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Save PIDs for stop script
echo "$BACKEND_PID" > .pids
echo "$FRONTEND_PID" >> .pids

echo
echo "===================================="
echo " Ready!"
echo " Open: http://localhost:5173"
echo " Stop: ./stop.sh (or Ctrl+C)"
echo " PIDs: backend=$BACKEND_PID frontend=$FRONTEND_PID"
echo "===================================="

# Handle Ctrl+C gracefully
trap 'echo; echo "Shutting down..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .pids; exit 0' INT TERM

# Wait for either process to exit
wait
