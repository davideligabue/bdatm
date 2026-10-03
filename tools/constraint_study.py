#!/usr/bin/env python3
"""Study constrained decoding by re-ranking saved predictions. CPU only

score.py stores the model's log-probability for every candidate, so any
constraint can be applied afterwards without touching the GPU. Three sweeps:

  train coverage  the trie is built from 10 / 25 / 50 / 100 % of the non-test
                  sentences (honest: no test data)
  test coverage   the trie is built from all non-test sentences plus 0 to 100 %
                  of the test sentences, in nested subsets. Leaky by construction:
                  it measures how the gain grows with coverage, from the honest
                  setting (0 %) to the oracle (100 %)
  soft prior      a bonus lambda on allowed candidates instead of a hard mask

Writes results/constraint_study.json.

Usage:
    python3 tools/constraint_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.constraints import PrefixTrie
from src.data import load_corpus, load_sentences
from src.metrics import compute, load_predictions

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The picto run on Qwen3.5-2B
RUNS = ("picto_text_Qwen3.5-2B",)

TRAIN_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
TEST_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
TEST_SEEDS = (0, 1, 2, 3, 4)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 5.0)


def steps_from_sentences(df) -> list[dict]:
    """Turn sentences into (prefix, next) pairs for the trie"""
    out = []
    for row in df.itertuples(index=False):
        ids = [int(p) for p in row.picto_ids]
        for t, gold in enumerate(ids):
            out.append({"prefix_ids": ids[:t], "gold_id": gold})
    return out


def rerank(records, steps_by_id, trie, hard: bool, lam: float = 1.0):
    """Re-rank each pool under a constraint

    Args:
        hard: suppress disallowed candidates outright, else add `lam` to allowed ones
    Returns:
        (records with new rankings, fraction of steps where the gold was disallowed)
    """
    out, removed = [], 0
    for rec in records:
        step = steps_by_id[rec["step_id"]]
        pool = {int(p) for p in rec["scores"]}
        ok = trie.allowed(step["prefix_ids"], pool)
        if rec["gold_id"] not in ok:
            removed += 1
        scored = [
            ((score - 1e4) if hard and int(pid) not in ok
             else (score + lam * (int(pid) in ok) if not hard else score), int(pid))
            for pid, score in rec["scores"].items()
        ]
        scored.sort(key=lambda x: -x[0])
        out.append({**rec, "ranked_ids": [p for _, p in scored]})
    return out, removed / max(1, len(records))


def study(run: str, corpus, sentences) -> list[dict]:
    """All three sweeps for one run

    Returns:
        One row per setting
    """
    records = load_predictions(RESULTS / run / "predictions.jsonl")
    steps_by_id = {s["step_id"]: s for s in corpus.steps["test"]}
    sentence_of = {s["step_id"]: s["sentence_idx"] for s in corpus.steps["test"]}
    eq = corpus.equivalence

    test_df = corpus.splits["test"]
    test_sentences = set(test_df["sentence"])
    non_test = sentences[~sentences["sentence"].isin(test_sentences)].reset_index(drop=True)
    non_test_steps = steps_from_sentences(non_test)

    base = compute(records, eq)
    rows = [{"setting": "no constraint", "sweep": "none", "hit@1": base["hit@1"],
             "mrr": base["mrr"], "gold_removed": 0.0}]
    print(f"\n=== {run}\nno constraint: hit@1 {base['hit@1']:.3f}  mrr {base['mrr']:.3f}")

    # --- train coverage: no test data -------------------------------------- #
    order = np.random.default_rng(0).permutation(len(non_test))
    for frac in TRAIN_FRACTIONS:
        subset = non_test.iloc[order[: int(len(non_test) * frac)]]
        trie = PrefixTrie.from_steps(steps_from_sentences(subset))
        ranked, removed = rerank(records, steps_by_id, trie, hard=True)
        m = compute(ranked, eq)
        rows.append({"setting": f"{frac:.0%} of non-test data", "sweep": "train",
                     "fraction": frac, "n_sentences": len(subset), "hit@1": m["hit@1"],
                     "mrr": m["mrr"], "gold_removed": removed})
        print(f"train {frac:>5.0%}  hit@1 {m['hit@1']:.3f}  gold removed {removed:.3f}")

    # --- test coverage: all non-test + nested fractions of the test set ----- #
    for frac in TEST_FRACTIONS:
        per_seed = []
        for seed in TEST_SEEDS:
            chosen = set(np.random.default_rng(seed).permutation(len(test_df))
                         [: int(round(len(test_df) * frac))].tolist())
            trie = PrefixTrie.from_steps(
                non_test_steps + steps_from_sentences(test_df.iloc[sorted(chosen)]))
            ranked, removed = rerank(records, steps_by_id, trie, hard=True)
            covered = [r for r in ranked if sentence_of[r["step_id"]] in chosen]
            uncovered = [r for r in ranked if sentence_of[r["step_id"]] not in chosen]
            per_seed.append({
                "hit@1": compute(ranked, eq)["hit@1"],
                "gold_removed": removed,
                "hit@1_covered": compute(covered, eq)["hit@1"] if covered else None,
                "hit@1_uncovered": compute(uncovered, eq)["hit@1"] if uncovered else None,
            })
        row = {"setting": f"non-test + {frac:.0%} of test", "sweep": "test",
               "fraction": frac, "seeds": len(TEST_SEEDS)}
        for key in ("hit@1", "gold_removed", "hit@1_covered", "hit@1_uncovered"):
            vals = [p[key] for p in per_seed if p[key] is not None]
            row[key] = float(np.mean(vals)) if vals else None
            row[f"{key}_sd"] = float(np.std(vals)) if vals else None
        rows.append(row)
        print(f"test  {frac:>5.0%}  hit@1 {row['hit@1']:.3f} ±{row['hit@1_sd']:.3f}  "
              f"covered {row['hit@1_covered'] or float('nan'):.3f}  "
              f"uncovered {row['hit@1_uncovered'] or float('nan'):.3f}")

    # --- soft prior on the train trie --------------------------------------- #
    trie_train = PrefixTrie.from_steps(corpus.steps["train"])
    for lam in LAMBDAS:
        ranked, _ = rerank(records, steps_by_id, trie_train, hard=False, lam=lam)
        m = compute(ranked, eq)
        rows.append({"setting": f"soft prior lambda={lam}", "sweep": "soft", "lambda": lam,
                     "hit@1": m["hit@1"], "mrr": m["mrr"]})
        print(f"soft  lambda={lam:<4}  hit@1 {m['hit@1']:.3f}")
    return rows


def main() -> None:
    corpus = load_corpus()
    sentences = load_sentences()
    out = {}
    for run in RUNS:
        if (RESULTS / run / "predictions.jsonl").exists():
            out[run] = study(run, corpus, sentences)
        else:
            print(f"skip {run}: no predictions yet")
    path = RESULTS / "constraint_study.json"
    path.write_text(json.dumps({"runs": out}, indent=2))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
