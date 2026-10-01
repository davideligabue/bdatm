#!/usr/bin/env bash
# Create the virtual environment and install the dependencies
#
#   ./setup.sh
#   export HF_TOKEN=hf_...      # an account with access to the gated datasets
#   ./run_all.sh --fresh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "Python 3.10 or newer is required"; exit 1; }

if [ ! -d .venv ]; then
  echo "creating .venv"
  "$PYTHON" -m venv .venv
fi

echo "installing dependencies"
./.venv/bin/pip install --upgrade pip --quiet
./.venv/bin/pip install -r requirements.txt

echo
echo "done. Next:"
echo "  export HF_TOKEN=...      # the two datasets are gated; your account"
echo "                           # must have been GRANTED access, not just"
echo "                           # hold a valid token"
echo "  ./run_all.sh --fresh"
