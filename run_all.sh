#!/usr/bin/env bash
# Run every experiment for one model, in order. A stage is skipped if its output
# already exists, so the script can be stopped and restarted at any time.
#
#   ./run_all.sh                                  # Qwen3.5-2B, default setting
#   MODEL=Qwen/Qwen3.5-0.8B ./run_all.sh          # another model
#   FULL_SENTENCE=1 ./run_all.sh                  # the full-sentence upper bound
#   ./run_all.sh --fresh                          # delete all outputs first
#   nohup ./run_all.sh > logs/run_all.log 2>&1 &
#
# Default setting: the prompt holds the pictograms chosen so far (plus the pool
# for id and text). FULL_SENTENCE=1 also shows the whole target sentence.

set -uo pipefail
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
  if [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; else PY=python3; fi
fi
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TAG="${MODEL##*/}"

if [ "${FULL_SENTENCE:-0}" = 1 ]; then
  SETTING=(--full-sentence); SUFFIX="_fullsent"
else
  SETTING=(); SUFFIX=""
fi

# Only one copy at a time; two would run out of GPU memory
exec 9>.run_all.lock
if command -v flock >/dev/null && ! flock -n 9; then
  echo "another run_all.sh holds .run_all.lock. Stop it first, or wait."
  exit 1
fi

FRESH=0
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    *) echo "unknown argument: $arg"; exit 1 ;;
  esac
done
if [ "$FRESH" = 1 ]; then
  echo "--fresh: removing results/, runs/ and cache/ so every stage recomputes"
  rm -rf results runs cache
fi
mkdir -p logs runs results

# run <marker-path> <label> <command...>
# Skips the command if the marker exists; logs to logs/<label>.log
SKIPPED=0
run () {
  local marker="$1"; local label="$2"; shift 2
  if [ -e "$marker" ]; then
    echo ">>> SKIP  $label  (found $marker)"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi
  echo ">>> START $label  $(date '+%F %T')"
  local log="logs/${label}.log"
  local status=0
  "$@" > "$log" 2>&1 || status=$?
  if [ "$status" = 0 ]; then
    echo ">>> DONE  $label  $(date '+%F %T')"
  else
    echo "!!! FAIL  $label  (exit $status) -- see $log"
    tail -20 "$log"
  fi
}

train () { $PY -u -m src.train --model "$MODEL" "${SETTING[@]}" "$@"; }
score () { $PY -u -m src.score "$@"; }

echo "=============================================================="
echo " model:   $MODEL"
echo " setting: ${SUFFIX:-pictograms so far (default)}"
echo " start:   $(date '+%F %T')"
if [ "$FRESH" = 0 ] && [ -n "$(ls -A results 2>/dev/null)" ]; then
  echo
  echo " NOTE: results/ already contains output, so the matching stages will be"
  echo "       skipped and their numbers will be the ones already on disk."
  echo "       Run './run_all.sh --fresh' to recompute everything."
fi
echo "=============================================================="

# --- stage 0: dataset cache and baselines (CPU) ------------------------------ #
run cache/catalogue.parquet          data        $PY -m src.data
run results/baselines.json           baselines   $PY -m src.baselines

# --- stage 1: output = ARASAAC id -------------------------------------------- #
R="id_${TAG}${SUFFIX}"
run "runs/$R/adapter"                "train_$R"  train --mode id
run "results/$R/predictions.jsonl"   "score_$R"  score --run "$R"

# --- stage 2: output = picto token, plus constrained decoding ---------------- #
R="picto_text_${TAG}${SUFFIX}"
run "runs/$R/adapter"                "train_$R"  train --mode picto --init text
run "results/$R/predictions.jsonl"   "score_$R"  score --run "$R"
run "results/${R}_constrained/predictions.jsonl" "score_${R}_constrained" \
    score --run "$R" --constrain

# --- stage 3: output = textual description ----------------------------------- #
R="text_${TAG}${SUFFIX}"
run "runs/$R/adapter"                "train_$R"  train --mode text
run "results/$R/predictions.jsonl"   "score_$R"  score --run "$R"

# --- stage 4: picto tokens initialised from the images ----------------------- #
# Needs an image encoder; not every model has one
HAS_VISION=$($PY -c "from transformers import AutoConfig as C; \
print(int(getattr(C.from_pretrained('$MODEL'), 'vision_config', None) is not None))" \
  2>/dev/null || echo 0)
if [ "$HAS_VISION" = 1 ]; then
  run "cache/vision_$TAG.npz"        "vision_$TAG" $PY -m src.vision_features --model "$MODEL"
  R="picto_image_${TAG}${SUFFIX}"
  run "runs/$R/adapter"              "train_$R"  train --mode picto --init image
  run "results/$R/predictions.jsonl" "score_$R"  score --run "$R"
else
  echo ">>> SKIP  vision stage  ($MODEL has no image encoder)"
fi

# --- full-sentence setting only: zero-shot control and text+image init ------- #
if [ -n "$SUFFIX" ]; then
  run "results/zeroshot_id_${TAG}_fullsent/predictions.jsonl" "zeroshot_$TAG" \
      score --zero-shot "$MODEL" --full-sentence
  if [ "$HAS_VISION" = 1 ]; then
    R="picto_textimage_${TAG}${SUFFIX}"
    run "runs/$R/adapter"              "train_$R"  train --mode picto --init text+image
    run "results/$R/predictions.jsonl" "score_$R"  score --run "$R"
  fi
fi

echo "=============================================================="
echo " finished: $(date '+%F %T')"
echo " stages skipped because their output already existed: $SKIPPED"
$PY - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.data import load_corpus
from src.metrics import load_predictions, compute, table
corpus = load_corpus()
rows = {}
for d in sorted(Path("results").iterdir()):
    f = d / "predictions.jsonl"
    if f.exists() and not d.name.startswith(("randneg_", "_")):
        rows[d.name] = compute(load_predictions(f), corpus.equivalence)
print(table(rows, ["hit@1", "hit@3", "mrr", "relaxed_hit@1", "relaxed_mrr", "gen_exact"]))
EOF
