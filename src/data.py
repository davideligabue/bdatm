"""Dataset loading and preprocessing.

Builds the corpus used by every other module: the pictogram catalogue, the
train/val/test splits, and the decision steps with their candidate pools.

A decision step is one training example: given a sentence, the pictograms
already selected, and 8 candidates, predict the next pictogram. A sentence with
k pictograms produces k steps.

Usage:
    python -m src.data        # build the cache and print dataset statistics
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SENTENCES_REPO = "disi-unibo-nlp-students/ARASAAC_CommonGen_new_dataset"
PICTOGRAMS_REPO = "disi-unibo-nlp-students/ARASAAC-Pictograms"

# Commit pinned so a later upload to either repo cannot silently change the
# splits, the label space, or any reported number
SENTENCES_REVISION = "7f5533ff7955875f82f16a2db2549348c68380b3"
PICTOGRAMS_REVISION = "34df552d9b3394a803d037894628a66a13d49be6"

# Bump when describe() changes, so the retrieval cache is not reused
DESCRIPTION_VERSION = 1

SEED = 42

# Number of sentences per split. Subsampled from the 54,797 available

N_TRAIN_SENTENCES = 12_000
N_VAL_SENTENCES = 1_500
N_TEST_SENTENCES = 2_000

# Candidates per decision step
POOL_SIZE = 8

NO_DESCRIPTION = "(no description)"

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_hf(repo: str, revision: str | None = None, **kwargs):
    """Load a pinned dataset split, retrying from the local cache if offline

    Both datasets are gated, so a caller without an approved HF_TOKEN gets a 401.
    That error is re-raised with a hint rather than retried, since retrying
    cannot fix it. Only connection failures fall back to the cache.

    Args:
        repo: dataset repository id
        revision: commit to pin
    Returns:
        The "train" split
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo, split="train", revision=revision, **kwargs)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if any(k in text for k in ("401", "403", "Gated", "gated", "Unauthorized",
                                   "authentication", "restricted")):
            raise RuntimeError(
                f"{repo} is gated. Set HF_TOKEN to a token whose account has been "
                f"granted access to it, then retry."
            ) from exc
        offline = any(k in text for k in ("name resolution", "ConnectError", "Connection",
                                          "Timeout", "Max retries", "502", "503"))
        if not offline:
            raise
        print(f"[hub] {type(exc).__name__}: reading {repo} from the local cache")
        return load_dataset(repo, split="train", revision=revision,
                            download_mode="reuse_cache_if_exists", **kwargs)


def _as_list(value) -> list[str]:
    """Normalise a metadata field to a list of strings"""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def _keyword_strings(keywords) -> list[str]:
    """Extract the keyword strings from a list of keyword records"""
    if keywords is None:
        return []
    out = []
    for entry in keywords:
        if not isinstance(entry, dict):
            continue
        kw = entry.get("keyword")
        if kw and str(kw).strip():
            out.append(str(kw).strip())
    return out


def _dedup(items: list[str]) -> list[str]:
    """Remove case-insensitive duplicates, preserving order"""
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def describe(keywords: list[str], categories: list[str]) -> str:
    """Render a pictogram as a single line of text for the prompt

    Args:
        keywords: pictogram keywords, e.g. ["knife grinder"].
        categories: pictogram categories, e.g. ["professional", "cutlery"].
    Returns:
        "knife grinder (professional, cutlery)", or NO_DESCRIPTION when the
        pictogram has neither.
    """
    kw = _dedup(keywords)
    cat = _dedup(categories)
    if not kw and not cat:
        return NO_DESCRIPTION
    if not kw:
        return f"({', '.join(cat)})"
    if not cat:
        return ", ".join(kw)
    return f"{', '.join(kw)} ({', '.join(cat)})"


def load_catalogue(rebuild: bool = False) -> pd.DataFrame:
    """Load the pictogram metadata.

    Returns:
        One row per pictogram with columns: pictogram_id, keywords, categories,
        tags, description, has_metadata.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "catalogue.parquet"
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    from datasets import Image as HFImage

    ds = _load_hf(PICTOGRAMS_REPO, PICTOGRAMS_REVISION)
    # Skip PNG decoding; only the metadata columns are needed here
    ds = ds.cast_column("image", HFImage(decode=False))
    df = ds.remove_columns(["image"]).to_pandas()

    rows = []
    for rec in df.to_dict("records"):
        kws = _keyword_strings(rec.get("keywords"))
        cats = _as_list(rec.get("categories"))
        tags = _as_list(rec.get("tags"))
        rows.append(
            {
                "pictogram_id": int(rec["pictogram_id"]),
                "keywords": kws,
                "categories": cats,
                "tags": tags,
                "description": describe(kws, cats),
                "n_metadata": len(kws) + len(cats) + len(tags),
            }
        )
    cat = pd.DataFrame(rows)

    # A few ids appear twice, once with metadata and once empty; keep the richer row
    cat = (
        cat.sort_values(["pictogram_id", "n_metadata"], ascending=[True, False])
        .drop_duplicates(subset="pictogram_id", keep="first")
        .reset_index(drop=True)
    )
    cat["has_metadata"] = cat["n_metadata"] > 0
    cat = cat.drop(columns=["n_metadata"])
    cat.to_parquet(cache_path, index=False)
    return cat


def load_sentences(rebuild: bool = False) -> pd.DataFrame:
    """Load the sentences and their ordered pictogram sequences.

    Drops null pictogram ids and sentences left with fewer than 2 pictograms.

    Returns:
        One row per sentence with columns: sentence, concepts, picto_ids, n_pictos.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "sentences.parquet"
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    ds = _load_hf(SENTENCES_REPO, SENTENCES_REVISION)
    raw = ds.to_pandas()

    rows = []
    for rec in raw.to_dict("records"):
        concepts = list(rec["concept"]) if rec["concept"] is not None else []
        ids = list(rec["best_id"]) if rec["best_id"] is not None else []
        kept_c, kept_i = [], []
        for concept, pid in zip(concepts, ids):
            if pid is None:
                continue
            try:
                kept_i.append(int(pid))
            except (TypeError, ValueError):
                continue
            kept_c.append(str(concept))
        if len(kept_i) < 2:
            continue
        rows.append(
            {
                "sentence": str(rec["sentence"]).strip(),
                "concepts": kept_c,
                "picto_ids": kept_i,
            }
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    df["n_pictos"] = df["picto_ids"].apply(len)
    df.to_parquet(cache_path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def split_sentences(
    sentences: pd.DataFrame,
    seed: int = SEED,
    n_train: int = N_TRAIN_SENTENCES,
    n_val: int = N_VAL_SENTENCES,
    n_test: int = N_TEST_SENTENCES,
) -> dict[str, pd.DataFrame]:
    """Split at sentence level so all steps of a sentence stay in one split.

    Returns:
        {"train": df, "val": df, "test": df}
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    shuffled = sentences.iloc[order].reset_index(drop=True)

    total = n_train + n_val + n_test
    if total > len(shuffled):
        raise ValueError(f"asked for {total} sentences, only {len(shuffled)} available")

    splits = {
        "train": shuffled.iloc[:n_train].reset_index(drop=True),
        "val": shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": shuffled.iloc[n_train + n_val : total].reset_index(drop=True),
    }

    seen: set[str] = set()
    for name, part in splits.items():
        this = set(part["sentence"])
        overlap = seen & this
        assert not overlap, f"{name} overlaps an earlier split on {len(overlap)} sentences"
        seen |= this
    return splits


# --------------------------------------------------------------------------- #
# Candidate retrieval
# --------------------------------------------------------------------------- #


def retrieve_candidates(
    sentences: pd.DataFrame,
    catalogue: pd.DataFrame,
    top_k: int = 60,
    cache_key: str = "",
    rebuild: bool = False,
) -> dict[int, list[int]]:
    """Retrieve plausible pictograms per sentence with TF-IDF cosine similarity.

    Used to fill candidate pools with distractors that are related to the
    sentence, as a retrieval stage would produce.

    Args:
        sentences: dataframe with a "sentence" column.
        catalogue: pictogram catalogue.
        top_k: candidates to keep per sentence.
    Returns:
        {row index in `sentences` -> list of pictogram ids, best first}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    CACHE_DIR.mkdir(exist_ok=True)
    # The key covers everything the result depends on. Without the seed and the
    # split sizes, changing either would silently reuse hits computed for a
    # different set of sentences
    cache_path = CACHE_DIR / f"retrieved_{cache_key or 'all'}_{top_k}.parquet"
    if cache_path.exists() and not rebuild:
        cached = pd.read_parquet(cache_path)
        return {int(i): [int(p) for p in ids] for i, ids in
                zip(cached["sentence_idx"], cached["retrieved"])}

    ids = [int(p) for p in catalogue["pictogram_id"]]
    corpus_text = [d if d != NO_DESCRIPTION else "" for d in catalogue["description"]]
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
    doc_matrix = normalize(vec.fit_transform(corpus_text))
    query_matrix = normalize(vec.transform(sentences["sentence"].tolist()))

    out: dict[int, list[int]] = {}
    chunk = 512
    for start in range(0, query_matrix.shape[0], chunk):
        sims = (query_matrix[start : start + chunk] @ doc_matrix.T).toarray()
        for offset, row in enumerate(sims):
            top = np.argpartition(-row, min(top_k, len(row) - 1))[:top_k]
            top = top[np.argsort(-row[top])]
            out[start + offset] = [ids[i] for i in top]

    pd.DataFrame(
        {"sentence_idx": list(out), "retrieved": [out[i] for i in out]}
    ).to_parquet(cache_path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Decision steps
# --------------------------------------------------------------------------- #


def build_steps(
    sentences: pd.DataFrame,
    negative_universe: list[int],
    pool_size: int = POOL_SIZE,
    seed: int = SEED,
    retrieved: dict[int, list[int]] | None = None,
) -> list[dict]:
    """Expand sentences into decision steps.

    At step t of a sentence with pictograms [p0 ... pk-1]:
        prefix = p0 ... p(t-1)
        gold   = pt
        pool   = the still-needed pictograms plus distractors, `pool_size` total

    Keeping the still-needed pictograms in the pool makes the task sequential:
    the model has to pick which of them comes next.

    Args:
        negative_universe: pictogram ids distractors may be sampled from.
        retrieved: per-sentence retrieval results; distractors are taken from
            here when given, otherwise sampled uniformly.
    Returns:
        List of step dicts with keys: step_id, sentence, sentence_idx, t,
        n_pictos, prefix_ids, prefix_concepts, gold_id, gold_concept, pool_ids.
    """
    rng = random.Random(seed)
    steps: list[dict] = []

    for row_idx, row in enumerate(sentences.itertuples(index=False)):
        # parquet returns numpy arrays; cast to plain ints for JSON output
        picto_ids: list[int] = [int(p) for p in row.picto_ids]
        concepts: list[str] = [str(c) for c in row.concepts]
        in_sentence = set(picto_ids)

        for t, gold in enumerate(picto_ids):
            remaining = _dedup_ints(picto_ids[t:])[:pool_size]
            pool = list(remaining)

            n_negatives = pool_size - len(pool)
            negatives: list[int] = []
            taken: set[int] = set()

            for cand in retrieved.get(row_idx, ()) if retrieved else ():
                if len(negatives) >= n_negatives:
                    break
                if cand in in_sentence or cand in taken:
                    continue
                negatives.append(cand)
                taken.add(cand)

            # Top up if retrieval returned too few usable distractors
            guard = 0
            while len(negatives) < n_negatives and guard < n_negatives * 50:
                guard += 1
                cand = rng.choice(negative_universe)
                if cand in in_sentence or cand in taken:
                    continue
                negatives.append(cand)
                taken.add(cand)
            pool.extend(negatives)
            rng.shuffle(pool)

            steps.append(
                {
                    "step_id": f"{row_idx}:{t}",
                    "sentence": str(row.sentence),
                    "sentence_idx": row_idx,
                    "t": t,
                    "n_pictos": len(picto_ids),
                    "prefix_ids": picto_ids[:t],
                    "prefix_concepts": concepts[:t],
                    "gold_id": int(gold),
                    "gold_concept": concepts[t] if t < len(concepts) else "",
                    "pool_ids": [int(p) for p in pool],
                }
            )

    return steps


def _dedup_ints(values: list[int]) -> list[int]:
    """Remove duplicates, preserving order"""
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Synonym groups
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def normalise_keyword(kw: str) -> str:
    """Lowercase and collapse whitespace"""
    return _WS.sub(" ", kw.strip().lower())


def build_equivalence(catalogue: pd.DataFrame) -> dict[int, set[int]]:
    """Group pictograms that share at least one keyword.

    Used by the relaxed metrics, where a synonymous pictogram counts as correct.

    Returns:
        {pictogram id -> set of equivalent ids, always including itself}
    """
    by_keyword: dict[str, set[int]] = {}
    for pid, kws in zip(catalogue["pictogram_id"], catalogue["keywords"]):
        for kw in kws:
            by_keyword.setdefault(normalise_keyword(kw), set()).add(int(pid))

    equivalence: dict[int, set[int]] = {}
    for pid, kws in zip(catalogue["pictogram_id"], catalogue["keywords"]):
        pid = int(pid)
        group = {pid}
        for kw in kws:
            group |= by_keyword.get(normalise_keyword(kw), set())
        equivalence[pid] = group
    return equivalence


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


@dataclass
class Corpus:
    """Everything the other modules need, built once and passed around"""

    catalogue: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    steps: dict[str, list[dict]]
    equivalence: dict[int, set[int]]
    description: dict[int, str]
    vocab_ids: list[int]  # label space: pictograms the model can predict

    def pool_descriptions(self, pool_ids: list[int]) -> list[tuple[int, str]]:
        """Pair each candidate id with its description"""
        return [(pid, self.description.get(pid, NO_DESCRIPTION)) for pid in pool_ids]


def load_corpus(
    seed: int = SEED,
    pool_size: int = POOL_SIZE,
    n_train: int = N_TRAIN_SENTENCES,
    n_val: int = N_VAL_SENTENCES,
    n_test: int = N_TEST_SENTENCES,
    rebuild: bool = False,
    negatives: str = "retrieval",
) -> Corpus:
    """Build the full corpus.

    Args:
        negatives: "retrieval" to draw distractors from TF-IDF hits,
            "random" to sample them uniformly from the catalogue.
    Returns:
        A Corpus.
    """
    catalogue = load_catalogue(rebuild=rebuild)
    sentences = load_sentences(rebuild=rebuild)
    splits = split_sentences(sentences, seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)

    description = {
        int(pid): desc for pid, desc in zip(catalogue["pictogram_id"], catalogue["description"])
    }
    negative_universe = sorted(description)

    # Identifies the exact split configuration these retrieval hits belong to
    config_key = f"s{seed}_t{n_train}_v{n_val}_e{n_test}_d{DESCRIPTION_VERSION}"

    steps = {}
    for i, (name, part) in enumerate(splits.items()):
        retrieved = (
            retrieve_candidates(part, catalogue, cache_key=f"{name}_{config_key}",
                                rebuild=rebuild)
            if negatives == "retrieval"
            else None
        )
        steps[name] = build_steps(
            part, negative_universe, pool_size=pool_size, seed=seed + i, retrieved=retrieved
        )

    # Label space: every pictogram appearing in any split, so it is the same
    # across train, val and test
    vocab = sorted({pid for part in splits.values() for ids in part["picto_ids"] for pid in ids})

    return Corpus(
        catalogue=catalogue,
        splits=splits,
        steps=steps,
        equivalence=build_equivalence(catalogue),
        description=description,
        vocab_ids=[int(p) for p in vocab],
    )


# --------------------------------------------------------------------------- #


def _summary() -> None:
    """Print dataset statistics"""
    corpus = load_corpus()
    cat = corpus.catalogue

    print("=" * 72)
    print("CATALOGUE")
    print(f"  pictograms                 {len(cat):,}")
    print(f"  with no metadata at all    {(~cat['has_metadata']).sum():,}"
          f" ({(~cat['has_metadata']).mean():.1%})")
    print(f"  rendering '(no description)' {(cat['description'] == NO_DESCRIPTION).sum():,}")
    print(f"  unique descriptions        {cat['description'].nunique():,}")

    print("\nSENTENCE SPLITS")
    for name, part in corpus.splits.items():
        steps = corpus.steps[name]
        print(f"  {name:<6} {len(part):>7,} sentences   {len(steps):>8,} decision steps"
              f"   mean len {part['n_pictos'].mean():.2f}")

    print("\nLABEL SPACE")
    print(f"  distinct pictograms used   {len(corpus.vocab_ids):,}")
    missing = [p for p in corpus.vocab_ids
               if corpus.description.get(p, NO_DESCRIPTION) == NO_DESCRIPTION]
    print(f"  of which no description    {len(missing):,} "
          f"({len(missing)/len(corpus.vocab_ids):.1%})")

    sizes = {len(s["pool_ids"]) for s in corpus.steps["test"]}
    golds_in_pool = sum(s["gold_id"] in s["pool_ids"] for s in corpus.steps["test"])
    print("\nPOOLS (test split)")
    print(f"  pool sizes observed        {sorted(sizes)}")
    print(f"  gold present in pool       {golds_in_pool:,}/{len(corpus.steps['test']):,}")

    eq_sizes = np.array([len(corpus.equivalence[p]) for p in corpus.vocab_ids])
    print("\nSYNONYM GROUPS")
    print(f"  mean equivalent pictograms {eq_sizes.mean():.2f}   median "
          f"{np.median(eq_sizes):.0f}   max {eq_sizes.max()}")
    print(f"  singletons (only itself)   {(eq_sizes == 1).sum():,}")

    ex = corpus.steps["test"][1]
    print("\nEXAMPLE STEP")
    print(json.dumps({**ex, "pool": corpus.pool_descriptions(ex["pool_ids"])},
                     indent=2, default=str)[:1400])
    print("=" * 72)


if __name__ == "__main__":
    _summary()
