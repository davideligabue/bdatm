"""Non-neural baselines, for reference against the fine-tuned models.

    random          pick uniformly from the pool
    frequency       rank by how often each pictogram appears in training
    ngram           prefix -> next-pictogram counts with back-off

Each writes results/<name>/predictions.jsonl in the same format as score.py, so
metrics.py treats them identically. A zero-shot LLM baseline needs a GPU and
lives in score.py --zero-shot.

All three see only the pool and the pictograms chosen so far, never the
sentence, and learn only from the training split.

Every baseline is run twice: on the retrieval pools used by the models, and on
pools with random distractors, which shows why retrieval matters.

Usage:
    python -m src.baselines
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import NO_DESCRIPTION, load_corpus
from .metrics import compute, table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# --------------------------------------------------------------------------- #
# Rankers. Each takes a decision step and returns its pool, ranked
# --------------------------------------------------------------------------- #


def rank_random(step: dict, rng: random.Random, **_) -> list[int]:
    """Shuffle the pool"""
    pool = list(step["pool_ids"])
    rng.shuffle(pool)
    return pool


def rank_frequency(step: dict, freq: Counter, **_) -> list[int]:
    """Rank by training frequency"""
    return sorted(step["pool_ids"], key=lambda p: -freq.get(p, 0))


def rank_ngram(step: dict, ngram: dict, **_) -> list[int]:
    """Rank by how often each candidate followed this prefix in training.

    Backs off to shorter suffixes when the full prefix is unseen.
    """
    prefix = tuple(step["prefix_ids"])
    scores: dict[int, float] = {}
    for length in range(len(prefix), -1, -1):
        counts = ngram.get(prefix[len(prefix) - length :] if length else ())
        if not counts:
            continue
        weight = 10.0**length  # longer context dominates
        for pid, c in counts.items():
            scores[pid] = scores.get(pid, 0.0) + weight * c
        break
    unigram = ngram.get(())
    return sorted(
        step["pool_ids"],
        key=lambda p: (-scores.get(p, 0.0), -(unigram.get(p, 0) if unigram else 0)),
    )


# --------------------------------------------------------------------------- #


def build_ngram(steps: list[dict], max_order: int = 3) -> dict[tuple[int, ...], Counter]:
    """Count next-pictogram frequencies per prefix suffix.

    Args:
        steps: training decision steps.
    Returns:
        {prefix suffix -> Counter of next pictogram}
    """
    counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for step in steps:
        prefix = tuple(step["prefix_ids"])
        gold = step["gold_id"]
        for length in range(0, min(max_order, len(prefix)) + 1):
            key = prefix[len(prefix) - length :] if length else ()
            counts[key][gold] += 1
    return dict(counts)


def write_predictions(name: str, records: list[dict]) -> Path:
    """Write records to results/<name>/predictions.jsonl"""
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.jsonl"
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def run(negatives: str = "retrieval", split: str = "test") -> dict[str, dict]:
    """Run every baseline and save its predictions and metrics.

    Returns:
        {baseline name -> metrics dict}
    """
    corpus = load_corpus(negatives=negatives)
    steps = corpus.steps[split]
    train_steps = corpus.steps["train"]

    freq = Counter(s["gold_id"] for s in train_steps)
    ngram = build_ngram(train_steps)
    rng = random.Random(0)

    # Keep the two negative-sampling regimes in separate folders
    prefix = "" if negatives == "retrieval" else "randneg_"

    rankers = {
        "baseline_random": lambda s: rank_random(s, rng=rng),
        "baseline_frequency": lambda s: rank_frequency(s, freq=freq),
        "baseline_ngram": lambda s: rank_ngram(s, ngram=ngram),
    }

    # Frequency and n-gram never look at the pool, so they can also rank every
    # pictogram, like picto mode: the top 20 is saved as "open_ranked"
    vocab = list(corpus.vocab_ids)
    by_frequency = sorted(vocab, key=lambda p: -freq.get(p, 0))[:20]
    open_rankers = {
        "baseline_frequency": lambda s: by_frequency,
        "baseline_ngram": lambda s: rank_ngram({**s, "pool_ids": vocab}, ngram=ngram)[:20],
    }

    results: dict[str, dict] = {}
    for name, ranker in rankers.items():
        records = [
            {
                "step_id": s["step_id"],
                "gold_id": s["gold_id"],
                "ranked_ids": ranker(s),
                "t": s["t"],
                "n_pictos": s["n_pictos"],
                "has_description": corpus.description.get(s["gold_id"], NO_DESCRIPTION)
                != NO_DESCRIPTION,
            }
            for s in steps
        ]
        if name in open_rankers and negatives == "retrieval":
            for rec, s in zip(records, steps):
                rec["open_ranked"] = open_rankers[name](s)
        write_predictions(prefix + name, records)
        results[name.replace("baseline_", "")] = compute(records, corpus.equivalence)

    (RESULTS / f"baselines{'' if not prefix else '_random'}.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    for negatives in ("retrieval", "random"):
        res = run(negatives)
        print(f"\nBASELINES on the test split ({res['random']['n']:,} decision steps, "
              f"pool of 8, {negatives} negatives)\n")
        print(table(res))
