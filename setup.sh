#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "idol.cpp setup"
echo "========================================================"

# 1. Check for Python (>= 3.10)
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.10 or newer is required, but was not found in PATH."
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

# 2. Check architecture and OS
OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
    echo "Detected macOS ($ARCH)."
    if [ "$ARCH" = "arm64" ]; then
        echo "[NOTE] Apple Silicon (arm64) detected."
        echo "The idol.cpp prebuilt binaries use AVX2 vector instructions and run under Rosetta 2."
        echo "macOS 15 (Sequoia) or newer is required for Rosetta 2 AVX2 support."
    fi
fi

# 3. Create virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "[1/4] Existing virtual environment found (.venv)."
fi

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

# 4. Upgrade pip
echo "[2/4] Upgrading pip..."
"$VENV_PY" -m pip install --quiet --upgrade pip

# 5. Install PyTorch
echo "[3/4] Installing PyTorch..."
if [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA..."
    "$VENV_PIP" install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple || \
    "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
else
    echo "Installing standard/CPU PyTorch..."
    "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
fi

# 6. Install idol-sdk and requirements
echo "[4/4] Installing idol-sdk and dependencies..."
if [ -f "idol-sdk/setup.py" ]; then
    echo "Found local idol-sdk directory, installing from local copy..."
    "$VENV_PIP" install ./idol-sdk -r requirements-runtime.txt -r requirements-server.txt
else
    echo "Installing pinned idol-sdk from GitHub..."
    "$VENV_PIP" install "git+https://github.com/speridlabs/idol-sdk.git@4702a7f00850e34b6e83a2170cc123c3e9570744" -r requirements-runtime.txt -r requirements-server.txt
fi

# 7. Check native binaries
echo ""
BIN_FOUND=0
for path in idol_mesh bin/idol_mesh build/idol_mesh; do
    if [ -f "$path" ]; then
        chmod +x "$path" 2>/dev/null || true
        BIN_FOUND=1
    fi
done
for path in idol_decode bin/idol_decode build/idol_decode; do
    if [ -f "$path" ]; then
        chmod +x "$path" 2>/dev/null || true
    fi
done

if [ "$BIN_FOUND" -eq 1 ]; then
    echo "Prebuilt native binaries detected (idol_mesh, idol_decode)."
else
    echo "[NOTE] Native idol_mesh was not found in root, bin/, or build/."
    echo "If you cloned from git without prebuilt binaries, compile with:"
    echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
fi

echo ""
echo "========================================================"
echo "Setup complete! Run ./run.sh to start the server."
echo "========================================================"
