"""Prefix trie used to restrict which pictograms may follow a given sequence.

Built from the training split: for each sequence prefix, record which pictograms
were observed next. At inference the candidates outside that set are suppressed.

Prefixes unseen in training back off to shorter suffixes, then to the set of all
pictograms seen in training, so the allowed set is never empty.

Usage:
    python -m src.constraints     # report how much the trie restricts the pools
"""

from __future__ import annotations

from collections import defaultdict

MIN_CANDIDATES = 1
MAX_ORDER = 3


class PrefixTrie:
    """Maps a sequence suffix to the pictograms that followed it in training"""

    def __init__(self, max_order: int = MAX_ORDER):
        self.max_order = max_order
        self.children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        self.seen: set[int] = set()

    @classmethod
    def from_steps(cls, steps: list[dict], max_order: int = MAX_ORDER) -> "PrefixTrie":
        """Build a trie from decision steps (use the training split only)"""
        trie = cls(max_order)
        for step in steps:
            trie.insert(step["prefix_ids"], step["gold_id"])
        return trie

    def insert(self, prefix: list[int], nxt: int) -> None:
        """Record that `nxt` followed `prefix`, for every suffix up to max_order"""
        self.seen.add(nxt)
        prefix = tuple(prefix)
        for order in range(0, min(self.max_order, len(prefix)) + 1):
            key = prefix[len(prefix) - order :] if order else ()
            self.children[key].add(nxt)

    def allowed(self, prefix: list[int], universe: set[int]) -> set[int]:
        """Return the pictograms allowed after `prefix`, within `universe`.

        Backs off from the longest known suffix down to the unigram set, so the
        result is never empty.
        """
        prefix = tuple(prefix)
        for order in range(min(self.max_order, len(prefix)), -1, -1):
            key = prefix[len(prefix) - order :] if order else ()
            candidates = self.children.get(key)
            if candidates:
                hit = candidates & universe
                if len(hit) >= MIN_CANDIDATES:
                    return hit
        hit = self.seen & universe
        return hit if hit else set(universe)

    def coverage(self, steps: list[dict], universe: set[int]) -> dict:
        """Measure how often the constraint restricts the pool and removes the gold.

        Returns:
            Dict with n, fraction_restricted, fraction_gold_removed and
            mean_candidates_kept.
        """
        restricted = removed_gold = 0
        kept: list[int] = []
        for step in steps:
            pool = set(step["pool_ids"]) if "pool_ids" in step else set(universe)
            hit = self.allowed(step["prefix_ids"], pool)
            kept.append(len(hit))
            if len(hit) < len(pool):
                restricted += 1
            if step["gold_id"] not in hit:
                removed_gold += 1
        n = max(1, len(steps))
        return {
            "n": len(steps),
            "fraction_restricted": restricted / n,
            "fraction_gold_removed": removed_gold / n,
            "mean_candidates_kept": sum(kept) / n,
        }


def _report() -> None:
    """Print trie size and coverage on the test split"""
    from .data import load_corpus

    corpus = load_corpus()
    trie = PrefixTrie.from_steps(corpus.steps["train"])
    print(f"trie built from {len(corpus.steps['train']):,} training steps")
    print(f"  distinct prefix keys : {len(trie.children):,}")
    print(f"  pictograms ever seen : {len(trie.seen):,} / {len(corpus.vocab_ids):,}")

    print("\nPool of 8 (test split):")
    for k, v in trie.coverage(corpus.steps["test"], set(corpus.vocab_ids)).items():
        print(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v:,}")

    print(f"\nFull label space ({len(corpus.vocab_ids):,} pictograms, first 2,000 steps):")
    open_steps = [{**s, "pool_ids": list(corpus.vocab_ids)} for s in corpus.steps["test"][:2000]]
    for k, v in trie.coverage(open_steps, set(corpus.vocab_ids)).items():
        print(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v:,}")


if __name__ == "__main__":
    _report()
