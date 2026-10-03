#!/usr/bin/env bash
# End-to-end check in about 20 minutes on one GPU: data, baselines, a short
# training run for each output mode, scoring, constrained decoding, metrics
#
#   ./tools/smoke_test.sh
#   MODEL=Qwen/Qwen3.5-0.8B ./tools/smoke_test.sh
#
# QUICK=1 makes train.py and score.py use a handful of steps and write to
# runs/_smoke_* and results/_smoke_*; the leading underscore keeps them out of
# the notebook and of the run_all.sh summary. The numbers are meaningless: only
# "does every stage run" is being tested.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PY:-}" ]; then
  if [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; else PY=python3; fi
fi
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TAG="${MODEL##*/}"
export QUICK=1

echo ">>> data and baselines"
$PY -m src.data
$PY -m src.baselines > /dev/null
$PY -m src.metrics

for MODE in id text picto; do
  NAME="${MODE}_${TAG}"
  [ "$MODE" = picto ] && NAME="picto_text_${TAG}"
  echo ">>> train $MODE (30 steps)"
  $PY -m src.train --model "$MODEL" --mode "$MODE"
  echo ">>> score $MODE (100 test steps)"
  $PY -m src.score --run "_smoke_$NAME"
done

echo ">>> constrained decoding"
$PY -m src.score --run "_smoke_picto_text_${TAG}" --constrain

echo
echo "smoke test passed: every stage ran"
