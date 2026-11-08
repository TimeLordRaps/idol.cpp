#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/python" ]; then
    echo "Virtual environment not found. Running ./setup.sh first..."
    ./setup.sh
fi

# Ensure prebuilt binaries in bin/ or root are executable
chmod +x idol_mesh idol_decode idol_octree_compare bin/idol_mesh bin/idol_decode bin/idol_octree_compare 2>/dev/null || true

echo "Starting idol.cpp local server..."
echo "Web UI will be available at http://127.0.0.1:8127"
echo ""

exec .venv/bin/python tools/idol_server.py --repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf "$@"
