"""Ranking metrics computed from saved predictions. CPU only.

score.py writes one JSON line per decision step:

    {"step_id": ..., "gold_id": 11118, "ranked_ids": [11118, 2389, ...], ...}

This module turns those files into Hit@k and MRR. Keeping it separate from
scoring means new metrics and breakdowns cost no GPU time.

Each metric has two variants:
    strict   the prediction must be the gold pictogram
    relaxed  any pictogram sharing a keyword with the gold counts as correct

Usage:
    python -m src.metrics     # sanity check against a random ranking
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

K_VALUES = (1, 3, 5)


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Returns:
        (estimate, lower bound, upper bound)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    """Normal confidence interval on a mean, used for MRR.

    Returns:
        (mean, lower bound, upper bound)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = z * math.sqrt(var / n)
    return mean, mean - half, mean + half


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _first_rank(ranked: Sequence[int], accept: set[int]) -> int | None:
    """1-based rank of the first accepted id, or None if absent"""
    for i, pid in enumerate(ranked, start=1):
        if pid in accept:
            return i
    return None


def compute(
    records: Iterable[dict],
    equivalence: dict[int, set[int]] | None = None,
    k_values: Sequence[int] = K_VALUES,
    rank_key: str = "ranked_ids",
) -> dict:
    """Compute Hit@k and MRR with 95% confidence intervals.

    Args:
        records: prediction records.
        equivalence: synonym groups for the relaxed variants.
        rank_key: which saved ranking to score. "ranked_ids" is the 8-candidate
            pool, "open_ranked" the full label space, "ranked_ids_sum" the pool
            ranked by total instead of per-token log-probability.
    Returns:
        Dict of metric name -> value, each with a matching "<name>_ci" entry.
    """
    records = [r for r in records if rank_key in r]
    n = len(records)
    if n == 0:
        return {"n": 0}

    strict_hits = {k: 0 for k in k_values}
    relaxed_hits = {k: 0 for k in k_values}
    strict_rr: list[float] = []
    relaxed_rr: list[float] = []

    for rec in records:
        gold = rec["gold_id"]
        ranked = rec[rank_key]

        rank = _first_rank(ranked, {gold})
        strict_rr.append(1.0 / rank if rank else 0.0)
        for k in k_values:
            if rank is not None and rank <= k:
                strict_hits[k] += 1

        accept = equivalence.get(gold, {gold}) if equivalence else {gold}
        rrank = _first_rank(ranked, accept)
        relaxed_rr.append(1.0 / rrank if rrank else 0.0)
        for k in k_values:
            if rrank is not None and rrank <= k:
                relaxed_hits[k] += 1

    out: dict = {"n": n}
    for k in k_values:
        p, lo, hi = wilson(strict_hits[k], n)
        out[f"hit@{k}"] = p
        out[f"hit@{k}_ci"] = [lo, hi]
        p, lo, hi = wilson(relaxed_hits[k], n)
        out[f"relaxed_hit@{k}"] = p
        out[f"relaxed_hit@{k}_ci"] = [lo, hi]

    m, lo, hi = mean_ci(strict_rr)
    out["mrr"], out["mrr_ci"] = m, [lo, hi]
    m, lo, hi = mean_ci(relaxed_rr)
    out["relaxed_mrr"], out["relaxed_mrr_ci"] = m, [lo, hi]

    # Free-form generation metrics, present only if score.py recorded them
    gen = [r for r in records if "generated_id" in r]
    if gen:
        valid = sum(r["generated_id"] is not None for r in gen)
        correct = sum(r["generated_id"] == r["gold_id"] for r in gen)
        p, lo, hi = wilson(valid, len(gen))
        out["gen_valid"], out["gen_valid_ci"] = p, [lo, hi]
        p, lo, hi = wilson(correct, len(gen))
        out["gen_exact"], out["gen_exact_ci"] = p, [lo, hi]
    return out


def breakdown(
    records: Iterable[dict],
    key: Callable[[dict], str],
    equivalence: dict[int, set[int]] | None = None,
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, dict]:
    """Compute the metrics separately per group.

    Args:
        key: maps a record to its group name.
    Returns:
        {group name -> metrics dict}
    """
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(key(rec), []).append(rec)
    return {g: compute(rs, equivalence, k_values) for g, rs in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Input / output
# --------------------------------------------------------------------------- #


def load_predictions(path: str | Path) -> list[dict]:
    """Read a predictions.jsonl file"""
    path = Path(path)
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fmt(metrics: dict, key: str) -> str:
    """Format one metric as 'value ±half-width'"""
    if key not in metrics:
        return "--"
    value = metrics[key]
    ci = metrics.get(f"{key}_ci")
    if not ci:
        return f"{value:.3f}"
    return f"{value:.3f} ±{(ci[1] - ci[0]) / 2:.3f}"


def table(rows: dict[str, dict], columns: Sequence[str] = ()) -> str:
    """Render {run name -> metrics} as a markdown table"""
    columns = list(columns) or [
        "hit@1", "hit@3", "mrr", "relaxed_hit@1", "relaxed_hit@3", "relaxed_mrr"
    ]
    header = "| run | n | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 2) + "|"
    lines = [header, sep]
    for label, m in rows.items():
        cells = " | ".join(fmt(m, c) for c in columns)
        lines.append(f"| {label} | {m.get('n', 0):,} | {cells} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #


def _sanity() -> None:
    """Check that a random ranking scores about 1 / pool size"""
    import random

    from .data import load_corpus

    corpus = load_corpus()
    rng = random.Random(0)
    fake = []
    for step in corpus.steps["test"]:
        ranked = list(step["pool_ids"])
        rng.shuffle(ranked)
        fake.append({"gold_id": step["gold_id"], "ranked_ids": ranked})

    m = compute(fake, corpus.equivalence)
    expected = 1 / len(corpus.steps["test"][0]["pool_ids"])
    print(f"random-shuffle hit@1 = {m['hit@1']:.4f}  (expected ~{expected:.4f})")
    print(f"random-shuffle mrr   = {m['mrr']:.4f}")
    print(f"relaxed hit@1        = {m['relaxed_hit@1']:.4f}")
    assert abs(m["hit@1"] - expected) < 0.02, "metric is miscomputed"
    assert m["relaxed_hit@1"] >= m["hit@1"], "relaxed must dominate strict"
    print("\nOK")
    print("\n" + table({"random shuffle": m}))


if __name__ == "__main__":
    _sanity()
