#!/usr/bin/env bash
# Run Photool on macOS or Linux: the same as serve.bat on Windows. Builds the page
# when it is missing, then serves the page and the API at http://localhost:8000.
#   ./serve.sh            start; Ctrl+C stops it
#   ./serve.sh rebuild    build the page again first (after updating the code)
set -e
cd "$(dirname "$0")"

VENV="$HOME/photool-env/venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
    echo "[ERROR] Python venv not found at $VENV. Run ./setup.sh first."
    exit 1
fi

# Installs anything requirements.txt lists that the venv is missing (the first
# run after an update), and is quick when nothing is.
if ! "$PY" -c "import uvicorn, fastapi, rawpy" >/dev/null 2>&1; then
    echo "[1/3] Installing Python packages..."
    "$PY" -m pip install -r backend/requirements.txt
else
    echo "[1/3] Python packages OK"
fi

# Homebrew's folders too: a shell whose profile never ran `brew shellenv` lacks them.
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
if ! command -v exiftool >/dev/null 2>&1; then
    echo "[WARN] exiftool not found: capture settings, lens data and the camera's Picture Style"
    echo "       will be missing. Install it with: brew install exiftool"
fi

if [ "$1" = "rebuild" ]; then rm -rf frontend/dist; fi
if [ ! -f frontend/dist/index.html ]; then
    echo "[2/3] Building the page..."
    (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
else
    echo "[2/3] Page already built (run \"./serve.sh rebuild\" after updating the code)"
fi

echo "[3/3] Starting at http://localhost:8000 - Ctrl+C stops it."
cd backend
exec "$PY" serve.py
