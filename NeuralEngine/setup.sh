#!/usr/bin/env bash
# Create a virtualenv and install dependencies. Run once on the VPS (or laptop) after scp.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete."
echo "CPU torch is installed by default. On an NVIDIA GPU, install the matching CUDA build instead, e.g.:"
echo "  source .venv/bin/activate && pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
echo
echo "Verify the install end-to-end with:  source .venv/bin/activate && python smoke_test.py"
