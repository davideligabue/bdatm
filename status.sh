#!/usr/bin/env bash
# Show what is running, which stages are done, and the results so far
#   ./status.sh
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  if [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; else PY=python3; fi
fi

echo "=============================================================="
echo " $(date '+%F %T')"
# bracket in the pattern so this check cannot match its own command line
if pgrep -f "bash .*(run_al[l]|queu[e])\.sh" > /dev/null; then
  echo " experiments: RUNNING"
else
  echo " experiments: not running   (start with: nohup ./run_all.sh > logs/run_all.log 2>&1 &)"
fi
command -v nvidia-smi >/dev/null && \
  echo " GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
echo "=============================================================="

# The stage running now: the one whose log was written most recently, if that
# was in the last 10 minutes. Each stage writes logs/<stage>.log
NEWEST=$(ls -t logs/*.log 2>/dev/null | grep -vE "/(run_all|queue|smoke_test)[^/]*\.log$" | head -1)
RUNNING=""
if [ -n "$NEWEST" ] && [ -n "$(find "$NEWEST" -mmin -10 2>/dev/null)" ]; then
  RUNNING=$(basename "$NEWEST" .log)
fi

echo
echo "NOW"
if [ -z "$RUNNING" ]; then
  echo "  (nothing running)"
else
  f="logs/$RUNNING.log"
  # training progress bar and loss; skip the checkpoint loading bar
  bar=$(tr '\r' '\n' < "$f" | grep -v "Loading weights" \
        | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1)
  loss=$(tr '\r' '\n' < "$f" | grep -oE "'loss': '[0-9.]+'" | tail -1 | grep -oE "[0-9.]+")
  # scoring progress, printed by score.py
  scored=$(tr '\r' '\n' < "$f" | grep -E "^  scored " | tail -1 | sed 's/^  //')
  phase=$(tr '\r' '\n' < "$f" | grep -E "^  (open-set|free-form)" | tail -1 | sed 's/^  //')
  echo "  stage   : $RUNNING"
  [ -n "$bar" ] && [ -z "$scored" ] && echo "  training: $bar]   loss ${loss:-?}"
  [ -n "$scored" ] && echo "  scoring : $scored"
  [ -n "$phase" ] && echo "  phase   : $phase"
fi

RUNNING="$RUNNING" $PY - <<'EOF' 2>/dev/null || echo "  (results unavailable)"
import os, sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from src.data import load_corpus
from src.metrics import load_predictions, compute

pd.set_option("display.width", 200, "display.max_columns", 30)
running = os.environ.get("RUNNING", "")
R, RUNS = Path("results"), Path("runs")
MODELS = ["LFM2.5-350M", "Qwen3.5-2B", "Qwen3.5-0.8B"]
VISION = {"Qwen3.5-2B", "Qwen3.5-0.8B"}

# --- plan: every stage run_all.sh produces, default setting ------------------ #
def state(train, score):
    """done / training / scoring / trained / waiting for one stage"""
    if (R / score / "predictions.jsonl").exists():
        return "done"
    if running == f"score_{score}":
        return "scoring"
    if running == f"train_{train}":
        return "training"
    if (RUNS / train / "adapter").exists():
        return "trained"
    return "-"

plan = {}
for tag in MODELS:
    plan[tag] = {
        "id": state(f"id_{tag}", f"id_{tag}"),
        "picto": state(f"picto_text_{tag}", f"picto_text_{tag}"),
        "picto+constraints": state(f"picto_text_{tag}", f"picto_text_{tag}_constrained"),
        "text": state(f"text_{tag}", f"text_{tag}"),
        "picto image init": state(f"picto_image_{tag}", f"picto_image_{tag}")
                            if tag in VISION else "n/a",
    }
print("\nPLAN  (default setting: pictograms chosen so far;  - = not started)")
print(pd.DataFrame(plan).T.to_string())

# --- results ------------------------------------------------------------------ #
corpus = load_corpus()

def row(name):
    recs = load_predictions(R / name / "predictions.jsonl")
    m = compute(recs, corpus.equivalence, k_values=(1, 3, 5))
    out = {"hit@1": m["hit@1"], "±": (m["hit@1_ci"][1] - m["hit@1_ci"][0]) / 2,
           "hit@3": m["hit@3"], "hit@5": m["hit@5"], "mrr": m["mrr"]}
    if "open_ranked" in recs[0]:
        o = compute(recs, corpus.equivalence, k_values=(1, 5, 10), rank_key="open_ranked")
        out.update({"all@1": o["hit@1"], "all@5": o["hit@5"], "all@10": o["hit@10"]})
    return out

names = sorted(d.name for d in R.iterdir()
               if (d / "predictions.jsonl").exists() and not d.name.startswith(("_", "randneg_")))
groups = {
    "BASELINES (never read the sentence)": [n for n in names if n.startswith("baseline_")],
    "RESULTS  pictograms chosen so far": [n for n in names if not n.startswith("baseline_")],
}
print("\nhit@k: right pictogram in the top k of the pool of 8.  all@k: in the top k of all")
print("pictograms (picto, frequency, n-gram).  Strict metrics; relaxed ones are in the notebook")
for title, group in groups.items():
    if group:
        print(f"\n{title}")
        df = pd.DataFrame({n: row(n) for n in group}).T
        print(df.round(3).to_string(na_rep=""))
EOF
