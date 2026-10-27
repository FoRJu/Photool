#!/usr/bin/env bash
# First setup on macOS or Linux: the same steps as setup.bat on Windows.
set -e
cd "$(dirname "$0")"

echo "===================================="
echo " Photool - First Setup"
echo "===================================="
echo

# Python 3.11 or newer. macOS's own python3 can be older: install one from
# python.org or with Homebrew (brew install python@3.12).
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "[ERROR] Python 3.11+ not found. Install it (brew install python@3.12),"
    echo "        or name one: PYTHON=python3.12 ./setup.sh"
    exit 1
fi
echo "[OK] $("$PYTHON" --version)"

if ! command -v node >/dev/null 2>&1; then
    echo "[ERROR] Node.js not found. Install it from nodejs.org or with: brew install node"
    exit 1
fi
echo "[OK] Node.js $(node --version)"

if command -v exiftool >/dev/null 2>&1; then
    echo "[OK] exiftool found"
else
    echo "[WARN] exiftool not found. It makes RAW thumbnails fast and reads more metadata."
    echo "       Install it with: brew install exiftool   (Linux: apt install libimage-exiftool-perl)"
fi
echo

# The venv lives OUTSIDE this project folder on purpose - see dev.bat.
VENV="$HOME/photool-env/venv"
echo "[1/3] Setting up the Python backend..."
if [ ! -x "$VENV/bin/python" ]; then
    echo "    Creating venv at $VENV ..."
    mkdir -p "$(dirname "$VENV")"
    "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r backend/requirements.txt
echo "[OK] Backend ready ($VENV)"

echo
echo "[2/3] Setting up the React frontend..."
(cd frontend && npm install --silent)
echo "[OK] Frontend ready"

echo
if [ ! -f backend/config.yaml ]; then
    echo "[3/3] Creating backend/config.yaml..."
    cp backend/config.example.yaml backend/config.yaml
    echo "[ACTION REQUIRED] Edit backend/config.yaml: set library_root to your photo folder."
else
    echo "[3/3] backend/config.yaml already exists, skipping."
fi

echo
echo "===================================="
echo " Setup complete. Start with ./serve.sh"
echo "===================================="
